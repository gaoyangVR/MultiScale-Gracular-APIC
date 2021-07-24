#version 330

layout(location=0) in vec3 pos;
uniform mat4 MVP;
uniform vec3 eyepos;		//world coordinate.

out vec3 worldpos;

void main()
{
	gl_Position = MVP*vec4(pos,1.0f);
	worldpos = pos;
}
