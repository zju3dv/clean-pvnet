#version 450 core

in vec3 v_color;
in vec3 v_view;

in vec3 v_normal;
in vec3 v_light_dir;

layout (location = 0) out vec3 rgb;
layout (location = 1) out float depth;

layout (location = 2) uniform float dirlight_ambient;
layout (location = 3) uniform float dirlight_diffuse;
layout (location = 4) uniform float dirlight_specular;

void main(void) {
    vec3 Normal = normalize(v_normal);
    vec3 LightDir = normalize(v_light_dir);
    vec3 ViewDir = normalize(v_view);

    vec3 material_ambient = vec3(223./255, 214./255, 205./255 );
    vec3 material_diffuse = vec3(223./255, 214./255, 205./255 );
    vec3 material_specular = vec3(223./255, 214./255  , 205./255 );
    
    vec3 diffuse = max(dot(Normal, LightDir), 0.0) * material_diffuse;
    vec3 R = reflect(-LightDir, Normal);
    vec3 specular = max(dot(R, ViewDir), 0.0) * material_specular;

/*     vec3 dirlight_ambient = vec3(0.5);
    vec3 dirlight_diffuse = vec3(0.6);
    vec3 dirlight_specular = vec3(0.3); */

    rgb = vec3(dirlight_ambient  * material_ambient + 
               dirlight_diffuse  * diffuse +
               dirlight_specular * specular);

    if(rgb.x > 1.0) rgb.x = 1.0;
    if(rgb.y > 1.0) rgb.y = 1.0;
    if(rgb.z > 1.0) rgb.z = 1.0;
    
	depth = v_view.z;
}




