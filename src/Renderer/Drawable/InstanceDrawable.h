#pragma once

#include "Drawable.h"

namespace Renderer
{
	class InstanceDrawable : public Drawable
	{
	private:
		unsigned int m_instanceVBO;

	public:
		InstanceDrawable(unsigned int shaderIndex);
		~InstanceDrawable();

		void setInstanceMatrix(const std::vector<glm::mat4> &instanceMatrice,
			GLenum drawWay = GL_STATIC_DRAW);

		virtual void render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera, std::shared_ptr<Shader> shader = nullptr);
		virtual void renderDepth(std::shared_ptr<Shader> shader, Camera3D::ptr lightCamera);
	};

}

