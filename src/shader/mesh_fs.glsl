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

uniform int u_ShowColor;
uniform vec3 u_Color;
uniform vec3 u_Direction;
uniform sampler2D s_Lightmap;
uniform sampler2D s_ShadowMap;
uniform float u_LightBias;

layout(std140) uniform GlobalUniforms
{
    mat4 view_proj;
    mat4 light_view_proj;
    vec4 cam_pos;
};

// ------------------------------------------------------------------
// FUNCTIONS  -------------------------------------------------------
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
    vec3 n = normalize(FS_IN_Normal);
    vec3 l = -u_Direction;

    float frag_depth = (FS_IN_NDCFragPos.z / FS_IN_NDCFragPos.w) * 0.5 + 0.5;
    float shadow     = shadow_occlussion(FS_IN_WorldPos);

    vec3 color = exposed_color(texture(s_Lightmap, FS_IN_LightmapUV).rgb);

    vec3 final_color = linear_to_srgb(color);

    if (u_ShowColor == 1)
        final_color += shadow * u_Color * clamp(dot(n, l), 0.0, 1.0);

    FS_OUT_Color = final_color;
}

// ------------------------------------------------------------------
