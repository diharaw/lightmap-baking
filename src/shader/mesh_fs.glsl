// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_OUT_Color;

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec3 FS_IN_WorldPos;
in vec3 FS_IN_Normal;
in vec2 FS_IN_UV;
in vec2 FS_IN_LightmapUV;
in vec4 FS_IN_NDCFragPos;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform vec3 u_Color;
uniform vec3 u_LightColor;
uniform vec3 u_Direction;
uniform sampler2D s_Lightmap;
uniform sampler2D s_ShadowMap;
uniform float u_LightBias;
uniform float u_Roughness;
uniform float u_Metallic;

layout(std140) uniform GlobalUniforms
{
    mat4 view_proj;
    mat4 light_view_proj;
    vec4 cam_pos;
};

const float PI = 3.14159265359;

// ------------------------------------------------------------------
// FUNCTIONS  -------------------------------------------------------
// ------------------------------------------------------------------

float distribution_ggx(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// ------------------------------------------------------------------

float geometry_schlick_ggx(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;

}

// ------------------------------------------------------------------

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);

    return ggx1 * ggx2;
}

// ------------------------------------------------------------------

vec3 fresnel_schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// ------------------------------------------------------------------

vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}   

// ------------------------------------------------------------------

vec3 linear_to_srgb(in vec3 color)
{
    vec3 x = color * 12.92;
    vec3 y = 1.055 * pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.4)) - 0.055;

    vec3 clr = color;
    clr.r    = color.r < 0.0031308 ? x.r : y.r;
    clr.g    = color.g < 0.0031308 ? x.g : y.g;
    clr.b    = color.b < 0.0031308 ? x.b : y.b;

    return clr;
}

// ------------------------------------------------------------------

vec3 exposed_color(vec3 color)
{
    float exposure = -16.0;
    return exp2(exposure) * color;
}

// ------------------------------------------------------------------

float shadow_occlussion(vec3 p)
{
   // Transform frag position into Light-space.
    vec4 light_space_pos = light_view_proj * vec4(p, 1.0);

    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closest_depth = texture(s_ShadowMap, proj_coords.xy).r;
    // get depth of current fragment from light's perspective
    float current_depth = proj_coords.z;
    // check whether current frag pos is in shadow
    float bias   = u_LightBias;
    float shadow = current_depth - bias > closest_depth ? 1.0 : 0.0;

    return 1.0 - shadow;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    float frag_depth = (FS_IN_NDCFragPos.z / FS_IN_NDCFragPos.w) * 0.5 + 0.5;
    float shadow     = shadow_occlussion(FS_IN_WorldPos);

    vec3 N = normalize(FS_IN_Normal);
    vec3 V = normalize(cam_pos.xyz - FS_IN_WorldPos);
    vec3 R = reflect(-V, N); 

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, u_Color, u_Metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);

    {
        vec3 L = -u_Direction;
        vec3 H = normalize(V + L);
    
        vec3 radiance = u_LightColor;

        // Cook-Torrance BRDF
        float NDF = distribution_ggx(N, H, u_Roughness);   
        float G   = geometry_smith(N, V, L, u_Roughness);    
        vec3 F    = fresnel_schlick(max(dot(H, V), 0.0), F0);        
        
        vec3 nominator    = NDF * G * F;
        float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
        vec3 specular = nominator / denominator;
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - u_Metallic;	                
            
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * u_Color / PI + specular) * radiance * NdotL;
    }

    // ambient lighting (we now use IBL as the ambient term)
    vec3 F = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, u_Roughness);
    
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - u_Metallic;	  
    
    vec3 irradiance = texture(s_Lightmap, FS_IN_LightmapUV).rgb;
    vec3 diffuse      = irradiance * u_Color;

    vec3 ambient = (kD * diffuse);
    
    vec3 color = ambient + Lo * shadow;

    vec3 final_color = linear_to_srgb(exposed_color(color));

    FS_OUT_Color = final_color;
}

// ------------------------------------------------------------------
