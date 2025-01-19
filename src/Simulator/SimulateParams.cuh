#ifndef SIMULATE_PARAMS_H
#define SIMULATE_PARAMS_H

#include "vector_types.h"

struct SimulateParams
{
	uint3 m_gridSize;
	float3 m_gravity;
	float3 m_cellSize;
	float3 m_worldOrigin;
	float m_damp;
	float m_leftWall;
	float m_poly6Coff;
	float m_spikyGradCoff;
	float m_sphRadius;
	float m_sphRadiusSquared;
	float m_viscosity;
	float m_vorticity;
	float m_lambdaEps;
	float m_restDensity;
	float m_invRestDensity;
	float m_particleRadius;
	float m_oneDivWPoly6;
	float m_stiffness;
	float m_staticFrictonCoeff;
	float m_dynamicFrictonCoeff;
	float m_stackHeightCoeff;
	float m_staticFrictThreshold;
	float m_dynamicFrictThreshold;
	float m_sleepThreshold;
	unsigned int m_maxIterNums;
	unsigned int m_numGridCells;
	unsigned int m_numParticles;
};


#endif