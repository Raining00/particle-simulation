#version 430 core
layout (location = 0) in vec4 position;
layout (location = 1) in float aPhase;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectMatrix;
uniform float pointScale;
uniform float pointSize;

out vec3 eyeSpacePos;
out float phase;

void main(){
	eyeSpacePos = (viewMatrix * modelMatrix * vec4(position.xyz, 1.0f)).xyz;
	gl_PointSize = -pointScale * pointSize / eyeSpacePos.z;
	phase = aPhase;
	gl_Position = projectMatrix * viewMatrix * modelMatrix * vec4(position.xyz, 1.0f);
}