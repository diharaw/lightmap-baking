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

layout(std140) uniform CSMUniforms
{
    mat4  texture_matrices[8];
    vec4  direction;
    int   num_cascades;
    float far_bounds[8];
};

uniform int u_ShowColor;
uniform vec3 u_Color;
uniform sampler2D s_Lightmap;
uniform sampler2DArray s_ShadowMap;
uniform float u_Bias;

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

float shadow_occlussion(float frag_depth, vec3 n, vec3 l)
{
    int   index = 0;

    // Find shadow cascade.
    for (int i = 0; i < num_cascades - 1; i++)
    {
        if (frag_depth > far_bounds[i])
            index = i + 1;
    }

    // Transform frag position into Light-space.
    vec4 light_space_pos = texture_matrices[index] * vec4(FS_IN_WorldPos, 1.0);

    float current_depth = light_space_pos.z;

    float bias = u_Bias;

    float pcfDepth = texture(s_ShadowMap, vec3(light_space_pos.xy, float(index))).r;
    float shadow = current_depth - bias > pcfDepth ? 1.0 : 0.0;

    return 1.0 - shadow;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    vec3 n = normalize(FS_IN_Normal);
    vec3 l = -direction.xyz;

    float frag_depth = (FS_IN_NDCFragPos.z / FS_IN_NDCFragPos.w) * 0.5 + 0.5;
    float shadow     = shadow_occlussion(frag_depth, n, l);

    vec3 color = exposed_color(texture(s_Lightmap, FS_IN_LightmapUV).rgb);

    vec3 final_color = linear_to_srgb(color);

    if (u_ShowColor == 1)
        final_color += shadow * u_Color;

    FS_OUT_Color = final_color;
}

// ------------------------------------------------------------------
