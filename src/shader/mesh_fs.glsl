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

layout(std140, binding = 1) buffer CSMUniforms
{
	mat4 texture_matrices[8];
    vec4 direction;
    vec4 options;
    int num_cascades;
    float far_bounds[8];
};

uniform sampler2D s_Lightmap;
uniform sampler2DArray s_ShadowMap;

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
	int index = 0;
    float blend = 0.0;
    
	// Find shadow cascade.
	for (int i = 0; i < num_cascades - 1; i++)
	{
		if (frag_depth > far_bounds[i])
			index = i + 1;
	}

	blend = clamp( (frag_depth - far_bounds[index] * 0.995) * 200.0, 0.0, 1.0);
    
    // Apply blend options.
    blend *= options.z;

	// Transform frag position into Light-space.
	vec4 light_space_pos = texture_matrices[index] * vec4(FS_IN_WorldPos, 1.0f);

	float current_depth = light_space_pos.z;
    
	float bias = max(0.0005 * (1.0 - dot(n, l)), 0.0005);  

	float shadow = 0.0;
	vec2 texelSize = 1.0 / textureSize(s_ShadowMap, 0).xy;

	for(int x = -1; x <= 1; ++x)
	{
	    for(int y = -1; y <= 1; ++y)
	    {
	        float pcfDepth = texture(s_ShadowMap, vec3(light_space_pos.xy + vec2(x, y) * texelSize, float(index))).r; 
	        shadow += current_depth - bias > pcfDepth ? 1.0 : 0.0;        
	    }    
	}

	shadow /= 9.0

    return shadow;    
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    vec3 n = normalize(FS_IN_Normal);
	vec3 l = -direction.xyz;

	float frag_depth = (FS_IN_NDCFragPos.z / FS_IN_NDCFragPos.w) * 0.5 + 0.5;
	float shadow = shadow_occlussion(frag_depth, n, l);

    vec3 color   = exposed_color(texture(s_Lightmap, FS_IN_LightmapUV).rgb);

    FS_OUT_Color = shadow * vec3(1.0) + linear_to_srgb(color);
}

// ------------------------------------------------------------------
