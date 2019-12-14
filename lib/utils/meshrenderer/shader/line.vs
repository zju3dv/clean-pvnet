#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (binding=0) readonly buffer SCENE_BUFFER {
	mat4 view;
	mat4 projection;
	vec3 viewPos;
};

layout (location = 0) uniform vec3 vert_min;
layout (location = 1) uniform vec3 vert_max;

out vec3 v_color;

void main() {
	const vec4 vertices[] = vec4[](	/* vec4( 0,  0,  0, 1),
									vec4( 30,  0,  0, 1),
									vec4( 0,  0,  0, 1),
									vec4( 0,  30,  0, 1),
									vec4( 0,  0,  0, 1),
									vec4( 0,  0,  30, 1), */
									vec4( vert_min[0],vert_min[1],vert_min[2],1),
									vec4( vert_min[0],vert_min[1],vert_max[2],1),
									vec4( vert_min[0],vert_min[1],vert_min[2],1),
									vec4( vert_min[0],vert_max[1],vert_min[2],1),
									vec4( vert_min[0],vert_min[1],vert_min[2],1),
									vec4( vert_max[0],vert_min[1],vert_min[2],1),
									vec4( vert_min[0],vert_min[1],vert_max[2],1),
									vec4( vert_min[0],vert_max[1],vert_max[2],1),
									vec4( vert_min[0],vert_min[1],vert_max[2],1),
									vec4( vert_max[0],vert_min[1],vert_max[2],1),
									vec4( vert_min[0],vert_max[1],vert_min[2],1),
									vec4( vert_min[0],vert_max[1],vert_max[2],1),
									vec4( vert_min[0],vert_max[1],vert_min[2],1),
									vec4( vert_max[0],vert_max[1],vert_min[2],1),
									vec4( vert_max[0],vert_min[1],vert_min[2],1),
									vec4( vert_max[0],vert_max[1],vert_min[2],1),
									vec4( vert_max[0],vert_min[1],vert_min[2],1),
									vec4( vert_max[0],vert_min[1],vert_max[2],1),
									vec4( vert_max[0],vert_max[1],vert_max[2],1),
									vec4( vert_min[0],vert_max[1],vert_max[2],1),
									vec4( vert_max[0],vert_max[1],vert_max[2],1),
									vec4( vert_max[0],vert_min[1],vert_max[2],1),
									vec4( vert_max[0],vert_max[1],vert_max[2],1),
									vec4( vert_max[0],vert_max[1],vert_min[2],1));

	const vec3 colors[] = vec3[]( 	/* vec3(1, 0, 0),
										vec3(1, 0, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 0, 1),
										vec3(0, 0, 1), */
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0),
										vec3(0, 1, 0));

	gl_Position = projection * view * vertices[gl_InstanceID*2+gl_VertexID];
	v_color = colors[gl_InstanceID*2+gl_VertexID];
}