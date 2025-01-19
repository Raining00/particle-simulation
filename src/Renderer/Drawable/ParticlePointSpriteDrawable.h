#pragma once

#include "Drawable.h"
#include "../Manager/ShaderMgr.h"
#include "../Manager/TextureMgr.h"

namespace Renderer
{
	class ParticlePointSpriteDrawable : public Drawable
	{
	private:
		glm::vec3 m_baseColor;
		float m_particleRadius;
		bool m_vboCreateBySelf;
		bool m_phsevboCreateBySelf;
		unsigned int m_particleTex;
		unsigned int m_particleVAO;
		unsigned int m_particleVBO;
		unsigned int m_phaseVAO;
		unsigned int m_phaseVBO;
		unsigned int m_numParticles;
		unsigned int m_posChannel;
		ShaderMgr::ptr m_shaderMgr;
		TextureMgr::ptr m_textureMgr;

	public:
		ParticlePointSpriteDrawable(unsigned int posChannel = 4);

		~ParticlePointSpriteDrawable();

		void setParticleRadius(float radius);
		void setPositions(std::vector<glm::vec4> &position);
		void setPositions(std::vector<glm::vec3> &position);
		void setPhase(std::vector<float> &phase);
		void setParticleVBO(unsigned int vbo, int numParticles);
		void setPhaseVBO(unsigned int vbo, int numParticles);

		glm::vec3 &getBaseColor() { return m_baseColor; }
		unsigned int getParticleVBO()const { return m_particleVBO; }
		void setBaseColor(glm::vec3 target) { m_baseColor = target; }

		virtual void render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera,
			std::shared_ptr<Shader> shader = nullptr);

		virtual void renderDepth(std::shared_ptr<Shader> shader, Camera3D::ptr lightCamera);

	private:
		void generateGaussianMap(int resolution);
	};

}