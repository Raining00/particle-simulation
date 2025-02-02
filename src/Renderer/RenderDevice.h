#pragma once

#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "RenderSystem.h"
#include "Manager/Singleton.h"
#include "../ImGui/ImGuiOpenGLContext.h"

namespace Renderer
{

	class RenderDevice : public Singleton<RenderDevice>
	{
	private:
		bool m_debugMode;
		int m_width, m_height;
		GLFWwindow *m_windowHandler;
		std::shared_ptr<RenderSystem> m_renderSys;
		ImGui::ImGuiOpenGLContext::ptr m_imguiContext;

	public:
		typedef std::shared_ptr<RenderDevice> ptr;

		// key, deltatime.
		static glm::vec2 m_cursorPos;
		static glm::vec2 m_deltaCurPos;
		static bool m_keysPressed[1024];
		static float m_deltaTime, m_lastFrame;
		static bool m_buttonPressed[GLFW_MOUSE_BUTTON_LAST];

		RenderDevice() = default;
		~RenderDevice() = default;

		//! singleton instance.
		static  std::shared_ptr<RenderDevice> getSingleton();

		bool initialize(std::string title, int width, int height, bool debugEnable = false);
		bool run();
		void beginFrame();
		void endFrame();
		bool shutdown();

		int getWindowWidth()const { return m_width; }
		int getWindowHeight()const { return m_height; }

		std::shared_ptr<RenderSystem> getRenderSystem() const
		{
			return m_renderSys;
		}

	protected:
		// debug functions.
		void initializeDebugContex();
		static void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity,
			GLsizei length, const GLchar *message, const void *userParam);

		// callback functions.
		static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
		static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
		static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
		static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
		void processInput();
	};
}

