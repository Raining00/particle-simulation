#ifndef _FLUID_KERNEL_H
#define _FLUID_KERNEL_H

#include <math.h>
#include <cooperative_groups.h>
#include <thrust/tuple.h>

#include "math_constants.h"
#include "SimulateParams.cuh"

#include "../Common/helper_math.h"

using namespace cooperative_groups;

__constant__ SimulateParams params;
__device__
inline float Q_rsqrt(float number)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5f;

	x2 = number * 0.5f;
	y = number;
	i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);		// what the fuck?
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));
	return y;
}
__device__
float wPoly6(const float3 &r)
{
	const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
	if (lengthSquared > params.m_sphRadiusSquared || lengthSquared <= 0.00000001f)
		return 0.0f;
	float iterm = params.m_sphRadiusSquared - lengthSquared;
	return params.m_poly6Coff * iterm * iterm * iterm;
}

__device__
float3 wSpikyGrad(const float3 &r)
{
	const float lengthSquared = r.x * r.x + r.y * r.y + r.z * r.z;
	float3 ret = { 0.0f, 0.0f, 0.0f };
	if (lengthSquared > params.m_sphRadiusSquared || lengthSquared <= 0.00000001f)
		return ret;
	const float length = sqrtf(lengthSquared);
	float iterm = params.m_sphRadius - length;
	float coff = params.m_spikyGradCoff * iterm * iterm / length;
	ret.x = coff * r.x;
	ret.y = coff * r.y;
	ret.z = coff * r.z;
	return ret;
}

__device__
int3 calcGridPosKernel(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.m_worldOrigin.x) / params.m_cellSize.x);
	gridPos.y = floor((p.y - params.m_worldOrigin.y) / params.m_cellSize.y);
	gridPos.z = floor((p.z - params.m_worldOrigin.z) / params.m_cellSize.z);
	return gridPos;
}

__device__
unsigned int calcGridHashKernel(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.m_gridSize.x - 1);
	gridPos.y = gridPos.y & (params.m_gridSize.y - 1);
	gridPos.z = gridPos.z & (params.m_gridSize.z - 1);
	return gridPos.z * params.m_gridSize.x * params.m_gridSize.y + gridPos.y * params.m_gridSize.x + gridPos.x;
}

__global__
void calcParticlesHashKernel(
	unsigned int *gridParticleHash,
	float4 *pos,
	unsigned int numParticles
)
{
	unsigned int index = blockIdx.x* blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	volatile float4 curPos = pos[index];
	int3 gridPos = calcGridPosKernel(make_float3(curPos.x, curPos.y, curPos.z));
	unsigned int hashValue = calcGridHashKernel(gridPos);
	gridParticleHash[index] = hashValue;
}

__global__
void findCellRangeKernel(
	unsigned int *cellStart,			// output: cell start index
	unsigned int *cellEnd,				// output: cell end index
	unsigned int *gridParticleHash,		// input: sorted grid hashes
	unsigned int numParticles)
{
	thread_block cta = this_thread_block();
	extern __shared__ unsigned int sharedHash[];
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int hashValue;

	if (index < numParticles)
	{
		hashValue = gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hashValue;

		// first thread in block must load neighbor particle hash
		if (index > 0 && threadIdx.x == 0)
			sharedHash[0] = gridParticleHash[index - 1];
	}

	sync(cta);

	if (index < numParticles)
	{
		if (index == 0 || hashValue != sharedHash[threadIdx.x])
		{
			cellStart[hashValue] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
			cellEnd[hashValue] = index + 1;
	}
}
/*
	float4 pos(x, y, z, phase)
	float4 vel(x, y, z, mass/lagrange multiple)
	for granuar phase, mass = 1.0f
	for fluid phase = 0, the w component of vel is lagrange multiple
*/
__global__
void advect(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float* particlePhase,
	float deltaTime,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float4 readVel = velocity[index];
	float4 readPos = position[index];
	float3 nVel = make_float3(readVel);
	float3 nPos = make_float3(readPos);
	float  phase = particlePhase[index];
	nVel += deltaTime * params.m_gravity;
	if(phase == 0) nPos += deltaTime * nVel;

	bool contact = false;
	float3 contact_normal = make_float3(0.f);
	//float3 PosBeforeCollision = nPos;
	//float3 diffPos = make_float3(0.f);
	// collision with walls.
	if (nPos.x > 40.0f - params.m_particleRadius)
	{
		nPos.x = 40.0f - params.m_particleRadius;
	}
	if (nPos.x < params.m_leftWall + params.m_particleRadius)
	{
		nPos.x = params.m_leftWall + params.m_particleRadius;
	}
	if (nPos.y > 20.0f - params.m_particleRadius)
	{
		nPos.y = 20.0f - params.m_particleRadius;
	}
	if (nPos.y < -20.0f + params.m_particleRadius)
	{
		contact = true;
		if (phase == 1.f)
		{
			contact_normal = make_float3(0.f, 1.f, 0.f);
			nVel = make_float3(0.f);
		}
		nPos.y = -20.0f + params.m_particleRadius;
	}
	if (nPos.z > 20.0f - params.m_particleRadius)
	{
		nPos.z = 20.0f - params.m_particleRadius;
	}
	if (nPos.z < -20.0f + params.m_particleRadius)
	{
		nPos.z = -20.0f + params.m_particleRadius;
	}

	if (contact && phase == 1.f)
	{
		// Computes the vertical component of the velocity relative to the normal vector
		float3 normVel = dot(nVel, contact_normal) * contact_normal;

		// Calculate tangential component of velocity relative to the normal vector
		float3 tangVel = nVel - normVel;

		// Static friction threshold, usually proportional to normal force
		float staticFrictionThreshold = params.m_staticFrictonCoeff * length(normVel);

		if (length(tangVel) < staticFrictionThreshold)
		{
			// If the tangential velocity is less than the threshold, apply static friction
			tangVel = make_float3(0.f);
		}
		else
		{
			// Otherwise, apply kinetic friction
			tangVel = tangVel * (1.0f - params.m_dynamicFrictonCoeff);
		}

		// Compute the new velocity by subtracting part of the normal velocity
		nVel = normVel * params.m_damp + tangVel;
		//diffPos = nPos - PosBeforeCollision;
		//position[index] += make_float4(diffPos, 0.f);

	}
	
	float invMass = 1.f;
	//granular advect
	if (phase == 1.f)
	{
		nPos += nVel * deltaTime;
		float stakeHeight = nPos.y - params.m_worldOrigin.y;
		invMass = 1.f / expf(params.m_stackHeightCoeff * stakeHeight);
	}

	predictedPos[index] = make_float4(nPos, invMass);
}

__global__
void calcLagrangeMultiplier(
	float4 *predictedPos,
	float4 *velocity,
	float  *particlePhase,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells)
{
	// calculate current particle's density and lagrange multiplier.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float phase = particlePhase[index];
	if (phase == 1.0) return;

	float3 readVel = make_float3(velocity[index]);
	float3 curPos = make_float3(predictedPos[index]);
	int3 gridPos = calcGridPosKernel(curPos);
	
	float density = 0.0f;
	float gradSquaredSum_j = 0.0f;
	float gradSquaredSumTotal = 0.0f;
	float3 curGrad, gradSum_i = { 0.0f,0.0f,0.0f };
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				// empty cell.
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 neighbour = predictedPos[i];
					float3 r = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };
					density += wPoly6(r);
					curGrad = wSpikyGrad(r);
					curGrad.x *= params.m_invRestDensity;
					curGrad.y *= params.m_invRestDensity;
					curGrad.z *= params.m_invRestDensity;

					gradSum_i.x += curGrad.x;
					gradSum_i.y += curGrad.y;
					gradSum_i.y += curGrad.y;
					if (i != index)
						gradSquaredSum_j += curGrad.x * curGrad.x + curGrad.y * curGrad.y + curGrad.z * curGrad.z;
				}
			}
		}
	}
	gradSquaredSumTotal = gradSquaredSum_j + gradSum_i.x * gradSum_i.x + gradSum_i.y * gradSum_i.y + gradSum_i.z * gradSum_i.z;
	
	// density constraint.
	predictedPos[index].w = density;
	float constraint = density * params.m_invRestDensity - 1.0f;
	float lambda = -(constraint) / (gradSquaredSumTotal + params.m_lambdaEps);
	velocity[index] = {readVel.x, readVel.y, readVel.z, lambda};
}

__global__
void calcDeltaPosition(
	float4 *predictedPos,
	float4 *velocity,
	float3 *deltaPos,
	float *particlePhase,
	unsigned int *cellStart,
	unsigned int *cellEnd,
	unsigned int *gridParticleHash,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float phase = particlePhase[index];

	float4 readPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 curPos = { readPos.x, readPos.y, readPos.z };
	int3 gridPos = calcGridPosKernel(curPos);
	float curLambda = readVel.w;
	float3 deltaP = { 0.0f, 0.0f, 0.0f };
	float3 frictdeltaP = make_float3(0.f);
	float invMass1 = readPos.w;
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					float4 readneighbour = predictedPos[i];
					float neighbourLambda = velocity[i].w;
					float phase2 = particlePhase[i];
					if (phase == 0.f) {
						float3 r = { curPos.x - readneighbour.x, curPos.y - readneighbour.y, curPos.z - readneighbour.z };
						float corrTerm = wPoly6(r) * params.m_oneDivWPoly6;
						float coff = curLambda + neighbourLambda - 0.1f * corrTerm * corrTerm * corrTerm * corrTerm;
						float3 grad = wSpikyGrad(r);
						deltaP += coff * grad;
					}
					if (phase == 1.f)
					{
						if (i == index) continue;
						float3 deltaP2 = make_float3(0.f);
						float3 neighbour = make_float3(readneighbour);
						float invMass2 = readneighbour.w;
						float wSum = invMass1 + invMass2;
						float weight1 = invMass1 / wSum;
						float3 r = curPos - neighbour;
						float len = length(r);
						r /= len;
						if (len < params.m_particleRadius * 2)
						{
							float3 corr = params.m_stiffness * r * (params.m_particleRadius * 2 - len) / wSum;
							deltaP += invMass1 * corr;
							deltaP2 -= invMass2 * corr;
						}
						if (phase2 == 0.f) continue;
						// friction model
						float3 relativedeltaP = deltaP - deltaP2;
						frictdeltaP = relativedeltaP - dot(relativedeltaP, r) * r;
						float d_frictdeltaP = length(frictdeltaP);
						if (d_frictdeltaP < params.m_staticFrictThreshold)
						{
							frictdeltaP *= weight1 * params.m_stiffness;
						}
						else
						{
							frictdeltaP *= weight1 * min(1.f, params.m_dynamicFrictThreshold / d_frictdeltaP) * params.m_stiffness;
						}
					}
				}
			}
		}
	}
	float3 ret = make_float3(0.f);
	if(phase == 0.0)
		ret = {deltaP.x * params.m_invRestDensity, deltaP.y * params.m_invRestDensity,
		deltaP.z * params.m_invRestDensity };
	if (phase == 1.0)
		 ret = deltaP + frictdeltaP;
	deltaPos[index] = ret;
}

__global__
void addDeltaPosition(
	float4 *predictedPos,
	float3 *deltaPos,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float3 readPos = make_float3(predictedPos[index]);
	float3 deltapos = deltaPos[index];
	readPos += deltapos;

	predictedPos[index] = make_float4(readPos, 1.0f);
}

__global__
void distanceConstrain(
	float4* predictedPos,
	float3* deltaPos,
	float*  particlePhase,
	unsigned int* cellStart,
	unsigned int* cellEnd,
	unsigned int* gridParticleHash,
	unsigned int numParticles,
	unsigned int numCells
)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	float phase = particlePhase[index];
	float4 readPos = predictedPos[index];
	float3 curPos = make_float3(readPos);
	int3 gridPos = calcGridPosKernel(curPos);
	float3 deltaP = make_float3(0.f);
	float3 frictdeltaP = make_float3(0.f);
	float invMass1 = readPos.w;
#pragma unroll 3
	for (int z = -1; z <= 1; ++z)
	{
#pragma unroll 3
		for (int y = -1; y <= 1; ++y)
		{
#pragma unroll 3
			for (int x = -1; x <= 1; ++x)
			{
				int3 neighbourGridPos = { gridPos.x + x, gridPos.y + y, gridPos.z + z };
				unsigned int neighbourGridIndex = calcGridHashKernel(neighbourGridPos);
				unsigned int startIndex = cellStart[neighbourGridIndex];
				if (startIndex == 0xffffffff)
					continue;
				unsigned int endIndex = cellEnd[neighbourGridIndex];
#pragma unroll 32
				for (unsigned int i = startIndex; i < endIndex; ++i)
				{
					if (i == index) continue;
					float3 deltaP2 = make_float3(0.f);
					float4 readNeighbour = predictedPos[i];
					float3 neighbour = make_float3(readNeighbour);
					float invMass2 = readNeighbour.w;
					float wSum = invMass1 + invMass2;
					float weight1 = invMass1 / wSum;
					/*float3 r = { curPos.x - neighbour.x, curPos.y - neighbour.y, curPos.z - neighbour.z };*/
					float3 r = curPos - neighbour;
					float len = length(r);
					r /= len;
					if (len < params.m_particleRadius * 2)
					{
						float3 corr = params.m_stiffness * r * (params.m_particleRadius * 2 - len) / wSum;
						deltaP += invMass1 * corr;
						deltaP2 -= invMass2 * corr;
					}
					// friction model
					float3 relativedeltaP = deltaP - deltaP2;
					frictdeltaP = relativedeltaP - dot(relativedeltaP, r) * r;
					float d_frictdeltaP = length(frictdeltaP);
					if (d_frictdeltaP < params.m_staticFrictThreshold)
					{
						frictdeltaP *= weight1 * params.m_stiffness;
					}
					else
					{
						frictdeltaP *= weight1 * min(1.f, params.m_dynamicFrictThreshold / d_frictdeltaP) * params.m_stiffness;
					}
				}
			}
		}
	}
	deltaPos[index] = deltaP + frictdeltaP;
}

__global__
void updateVelocityAndPosition(
	float4 *position,
	float4 *velocity,
	float4 *predictedPos,
	float invDeltaTime,
	unsigned int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;

	float4 oldPos = position[index];
	float4 newPos = predictedPos[index];
	float4 readVel = velocity[index];
	float3 posDiff = { newPos.x - oldPos.x, newPos.y - oldPos.y, newPos.z - oldPos.z };
	posDiff *= invDeltaTime;
	velocity[index] = { posDiff.x, posDiff.y, posDiff.z, readVel.w };
	if(1.0f/ Q_rsqrt(dot(posDiff, posDiff)) > params.m_sleepThreshold)
		position[index] = { newPos.x, newPos.y, newPos.z, 1.0f };
}

#endif