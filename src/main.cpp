#include <string>
#include <iostream>
#include "asserts.h"

#include "Renderer/RenderDevice.h"
#include "Renderer/Camera/TPSCamera.h"
#include "Renderer/Manager/Geometry.h"
#include "Renderer/Drawable/InstanceDrawable.h"
#include "Renderer/Drawable/StaticModelDrawable.h"
#include "Renderer/Drawable/LiquidDrawable.h"
#include "Renderer/Drawable/ParticleInstanceDrawable.h"
#include "Renderer/Drawable/ParticlePointSpriteDrawable.h"
#include "Renderer/Voxelization.h"
#include "Simulator/FluidDemo.h"

using namespace std;
using namespace Renderer;
using namespace Simulator;

int main(int argc, char *argv[])
{
	// initialization.
	const int width = 1920, height = 1080;
	std::shared_ptr<RenderDevice> window = RenderDevice::getSingleton();
	window->initialize("Fluid Simulation", width, height, false);
	std::shared_ptr<RenderSystem> renderSys = window->getRenderSystem();
	renderSys->setGlowBlur(true);

	// resource loading.
	MeshMgr::ptr meshMgr = renderSys->getMeshMgr();
	ShaderMgr::ptr shaderMgr = renderSys->getShaderMgr();
	TextureMgr::ptr textureMgr = renderSys->getTextureMgr();
	// shaders.
	unsigned int simpleColorShader = shaderMgr->loadShader("simple_color",
		ASSERTS_PATH "glsl/simple_color.vert", ASSERTS_PATH "glsl/simple_color.frag");
	unsigned int simpleTextShader = shaderMgr->loadShader("simple_texture",
		ASSERTS_PATH "glsl/simple_texture.vert", ASSERTS_PATH "glsl/simple_texture.frag");
	unsigned int blinnPhongShader = shaderMgr->loadShader("blinn_phong",
		ASSERTS_PATH "glsl/blinn_phong.vert", ASSERTS_PATH "glsl/blinn_phong.frag");
	// textures.
	unsigned int containerTex = textureMgr->loadTexture2D("cube", ASSERTS_PATH "res/OIP.jpg");
	unsigned int fluidTex = textureMgr->loadTexture2D("blue", ASSERTS_PATH "res/blue.png");
	//renderSys->setSkyDome(ASSERTS_PATH "res/skybox/", ".jpg");
	// meshes.
	unsigned int planeMesh = meshMgr->loadMesh(new Plane(1.0, 1.0));
	unsigned int containerMesh = meshMgr->loadMesh(new Container(1.0f, 1.0f, 1.0f));
	unsigned int particleSphereMesh = meshMgr->loadMesh(new Sphere(1.0f, 5, 5));
	unsigned int particleCubeMesh = meshMgr->loadMesh(new Cube(1.0f, 1.0f, 1.0f));
	// sunlight.
	renderSys->setSunLight(glm::vec3(1.0f, 0.5f, 0.5f),
		glm::vec3(0.05f), glm::vec3(0.6f), glm::vec3(0.6f));
	// shadow setting.
	renderSys->createShadowDepthBuffer(4096, 4096);
	renderSys->createSunLightCamera(glm::vec3(0.0f), -600.0f, +600.0f,
		-600.0f, +600.0f, 1.0f, 500.0f);

	StaticModelDrawable *island = new StaticModelDrawable(blinnPhongShader,
		ASSERTS_PATH "res/Small_Tropical_Island/Small_Tropical_Island.obj");
	renderSys->addDrawable(island);

	glm::vec3 scaleSize = glm::vec3(80.0f, 40.0f, 40.0f);
	glm::vec3 center = glm::vec3(0, scaleSize.y/2 + 2.0f, 150.0f);
	SimpleDrawable *container[6];
	container[0] = new SimpleDrawable(blinnPhongShader);
	container[1] = new SimpleDrawable(blinnPhongShader);
	container[2] = new SimpleDrawable(blinnPhongShader);
	container[3] = new SimpleDrawable(blinnPhongShader);
	container[4] = new SimpleDrawable(blinnPhongShader);
	container[5] = new SimpleDrawable(blinnPhongShader);
	container[0]->addMesh(planeMesh);
	container[1]->addMesh(planeMesh);
	container[2]->addMesh(planeMesh);
	container[3]->addMesh(planeMesh);
	container[4]->addMesh(planeMesh);
	container[5]->addMesh(planeMesh);
	container[0]->addTexture(containerTex);
	container[1]->addTexture(containerTex);
	container[2]->addTexture(containerTex);
	container[3]->addTexture(containerTex);
	container[4]->addTexture(containerTex);
	container[5]->addTexture(containerTex);
	container[0]->setReceiveShadow(false);
	container[1]->setReceiveShadow(false);
	container[2]->setReceiveShadow(false);
	container[3]->setReceiveShadow(false);
	container[4]->setReceiveShadow(false);
	container[5]->setReceiveShadow(false);
	container[0]->setProduceShadow(true);
	container[1]->setProduceShadow(true);
	container[2]->setProduceShadow(true);
	container[3]->setProduceShadow(true);
	container[4]->setProduceShadow(true);
	container[5]->setProduceShadow(true);
	container[0]->getTransformation()->setScale(glm::vec3(scaleSize.y, scaleSize.y, scaleSize.z));
	container[1]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[2]->getTransformation()->setScale(glm::vec3(scaleSize.y, scaleSize.y, scaleSize.z));
	container[3]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[4]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[5]->getTransformation()->setScale(glm::vec3(scaleSize.x, scaleSize.y, scaleSize.z));
	container[0]->getTransformation()->rotate(glm::vec3(0, 0, 1), 270.0f);
	container[1]->getTransformation()->rotate(glm::vec3(0, 0, 1), 180.0f);
	container[2]->getTransformation()->rotate(glm::vec3(0, 0, 1), 90.00f);
	container[4]->getTransformation()->rotate(glm::vec3(1, 0, 0), 270.0f);
	container[5]->getTransformation()->rotate(glm::vec3(1, 0, 0), 90.00f);
	container[0]->getTransformation()->setTranslation(glm::vec3(center.x - scaleSize.x/2.0f, center.y, center.z));
	container[1]->getTransformation()->setTranslation(glm::vec3(center.x, center.y + scaleSize.y / 2.0f, center.z));
	container[2]->getTransformation()->setTranslation(glm::vec3(center.x + scaleSize.x / 2.0f, center.y, center.z));
	container[3]->getTransformation()->setTranslation(glm::vec3(center.x, center.y - scaleSize.y / 2.0f, center.z));
	container[4]->getTransformation()->setTranslation(glm::vec3(center.x , center.y, center.z + scaleSize.z / 2.0f));
	container[5]->getTransformation()->setTranslation(glm::vec3(center.x, center.y, center.z - scaleSize.z / 2.0f));
	renderSys->addDrawable(container[0]);
	renderSys->addDrawable(container[1]);
	renderSys->addDrawable(container[2]);
	renderSys->addDrawable(container[3]);
	renderSys->addDrawable(container[4]);
	renderSys->addDrawable(container[5]);

	Camera3D::ptr camera = renderSys->createTPSCamera(
		glm::vec3(0, 0, 0),
		glm::vec3(center.x, center.y, center.z));
	camera->setPerspectiveProject(45.0f, static_cast<float>(width) / height, 0.1f, 3000.0f);
	TPSCamera *tpsCamera = reinterpret_cast<TPSCamera*>(camera.get());
	tpsCamera->setPitch(15.0f);
	tpsCamera->setDistance(90.0f);
	tpsCamera->setDistanceLimt(0.01f, 1000.0f);
	tpsCamera->setWheelSensitivity(5.0f);

	FluidDemo demo;
	demo.fluidDeamBreak(128, scaleSize.x, 0.30f);
	FluidSystem::ptr fluid = demo.getSimulator();

	ParticlePointSpriteDrawable *particleDrawable = new ParticlePointSpriteDrawable(4);
	LiquidDrawable *liquidDrawable = new LiquidDrawable(width, height);

	particleDrawable->setBaseColor(glm::vec3(0.2863, 0.8157, 1.0));
	particleDrawable->setProduceShadow(false);
	particleDrawable->setParticleRadius(fluid->getSimulateParams().m_particleRadius);
	particleDrawable->getTransformation()->setTranslation(center);
	particleDrawable->setParticleVBO(fluid->getPosVBO(), fluid->getSimulateParams().m_numParticles);

	liquidDrawable->setParticleRadius(fluid->getSimulateParams().m_particleRadius);
	liquidDrawable->getTransformation()->setTranslation(center);
	liquidDrawable->setParticleVBO(fluid->getPosVBO(), fluid->getSimulateParams().m_numParticles);

	renderSys->addDrawable(particleDrawable);
	renderSys->setLiquidRenderer(std::shared_ptr<LiquidDrawable>(liquidDrawable));

	renderSys->setGlowBlur(false);
	bool run = false;
	bool reset = false;
	bool wallMove = false;
	float wallMoveSpeed = +10.0f;
	float curWallPos = -40.0f;
	bool fluidRendering = true;
	int curDemo = 0, item = 0;
	int curBlur = 0, blurItem = 0;
	const char *blurItems[] = { "Bilateral Blur", "Bilateral Seperate Blur", "Gaussian Blur", "Curvature Flow Blur" };
	const char *demoItems[] = { "Deam Break", "Sphere Fluid Drop", "Cube Fluid Drop" };

	while (window->run())
	{
		// fluid rendering switch.
		particleDrawable->setVisiable(!fluidRendering);
		liquidDrawable->setVisiable(fluidRendering);

		// wall movement.
		if (wallMove)
		{
			if (curWallPos >= -15.0f)
			{
				curWallPos = -15.0f;
				wallMoveSpeed = -wallMoveSpeed;
			}
			if (curWallPos < -40.0f)
			{
				curWallPos = -40.0f;
				wallMoveSpeed = -wallMoveSpeed;
			}
			curWallPos += wallMoveSpeed * window->m_deltaTime;
			fluid->getSimulateParams().m_leftWall = curWallPos;
			container[0]->getTransformation()->setTranslation(glm::vec3(curWallPos, center.y, center.z));
		}

		// simulation.
		if(run)
			fluid->simulate(1.f/ 60.f);

		window->beginFrame();

		renderSys->render();

		// Demo setting window.
		{
			ImGui::Begin("Fluid simulation");
			ImGui::Combo("Fluid Demo", &item, demoItems, 3);
			ImGui::Combo("Blur Algorithm", &blurItem, blurItems, 4);
			ImGui::Text("Particle Number: %d", fluid->getSimulateParams().m_numParticles);
			ImGui::Checkbox("use viscosity", &fluid->getSimulateParams().m_useXSPH);
			ImGui::SliderFloat("Viscosity", &fluid->getSimulateParams().m_viscosity, 0.0f, 0.2f);
			if (ImGui::Button("Wall Movement"))
				wallMove = !wallMove;
			if (ImGui::Button("Fluid Rendering"))
				fluidRendering = !fluidRendering;
			if (ImGui::Button("Reset"))
				reset = true;
			if (ImGui::Button("Run"))
				run = !run;
			ImGui::End();
		}

		if (item != curDemo || reset)
		{
			reset = false;
			curDemo = item;

			switch (curDemo)
			{
			case 0:
				demo.fluidDeamBreak(128, scaleSize.x, 0.30f);
				break;
			case 1:
				demo.sphereFluidDrop(128, scaleSize.x, 0.25f);
				break;
			case 2:
				demo.boxFluidDrop(65536, 128, scaleSize.x, 0.30f);
				break;
			}

			curWallPos = -40.0f;
			fluid->getSimulateParams().m_leftWall = curWallPos;
			container[0]->getTransformation()->setTranslation(glm::vec3(curWallPos, center.y, center.z));
			fluid = demo.getSimulator();
			particleDrawable->setParticleVBO(fluid->getPosVBO(), fluid->getSimulateParams().m_numParticles);
			liquidDrawable->setParticleVBO(fluid->getPosVBO(), fluid->getSimulateParams().m_numParticles);
		}

		if (curBlur != blurItem)
		{
			curBlur = blurItem;
			liquidDrawable->setLiquidBlur(curBlur);
		}

		window->endFrame();
	}

	fluid->finalize();
	window->shutdown();

	return 0;
}