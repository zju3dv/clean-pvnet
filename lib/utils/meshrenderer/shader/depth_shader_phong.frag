#version 450 core

in vec3 v_color;
in vec3 v_view;
in vec3 ViewDir;

in vec3 v_L;
in vec3 v_normal;

layout (location = 0) uniform float u_light_ambient_w;
layout (location = 2) uniform float u_light_diffuse_w;
layout (location = 3) uniform float u_light_specular_w;

layout (location = 0) out vec3 rgb;
layout (location = 1) out float depth;
layout (location = 2) out vec3 rgb_normals;

void main(void) {

    vec3 Normal = normalize(v_normal);
    vec3 LightDir = normalize(v_L);
    vec3 ViewDir = normalize(v_view);

    vec3 diffuse = max(dot(Normal, LightDir), 0.0) * v_color;
    vec3 R = reflect(-LightDir, Normal);
    vec3 specular = max(dot(R, ViewDir), 0.0) * v_color;


    rgb = vec3(u_light_ambient_w  * v_color + 
               u_light_diffuse_w  * diffuse +
               u_light_specular_w * specular);

    if(rgb.x > 1.0) rgb.x = 1.0;
    if(rgb.y > 1.0) rgb.y = 1.0;
    if(rgb.z > 1.0) rgb.z = 1.0;
    rgb_normals = vec3(Normal * 0.5 + 0.5); // transforms from [-1,1] to [0,1]  
    depth = v_view.z;
}

/* vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
 */
/* vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
 */

    //if(light_w > 1.0) light_w = 1.0;

    //vec3 hsv_color = rgb2hsv(v_color);
    //hsv_color.y = 0.5;
    //hsv_color.y = 0.0;
    //hsv_color.z = clamp(hsv_color.z, 0., 0.62);
/*     hsv_color.z = clamp(hsv_color.z, 0., 0.8);
    vec3 rgb_color = hsv2rgb(hsv_color); */

    //rgb = light_w * rgb_color;
    




