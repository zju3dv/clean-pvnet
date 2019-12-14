#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (binding=0) readonly buffer SCENE_BUFFER {
	mat4 view;
	mat4 projection;
	vec3 viewPos;
};

layout (location = 1) uniform vec3 u_light_eye_pos;
layout (location = 0) uniform vec3 light_pos;

/* out vec3 ViewDir; */
out vec3 v_color;
out vec3 v_view;

out vec3 v_normal;
out vec3 v_light_dir;

void main(void) {
	vec4 P = view * vec4(position, 1.0);

	v_view = -P.xyz;
	gl_Position = projection * P;

	mat4 u_nm = transpose(inverse(view));

	vec3 v_eye_pos = P.xyz; // Vertex position in eye coords.
	v_normal = normalize(u_nm * vec4(normal, 1.0)).xyz; // Normal in eye coords.

	v_light_dir = light_pos - P.xyz;
}