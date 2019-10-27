// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_OUT_Color;

// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec3 FS_IN_ObjPos;
in vec3 FS_IN_WorldPos;
in vec3 FS_IN_Normal;
in vec2 FS_IN_UV;
in vec2 FS_IN_LightmapUV;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Lightmap;
uniform int i_FromLightmap;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    vec3  light_pos = vec3(200.0, 200.0, 200.0);
    vec3  n         = normalize(FS_IN_Normal);
    vec3  l         = normalize(light_pos - FS_IN_WorldPos);
    float lambert   = max(0.0f, dot(n, l));
    vec3  diffuse   = vec3(0.5);
    vec3  ambient   = diffuse * 0.03;
    vec3  color     = diffuse * lambert + ambient;

    if (i_FromLightmap == 0)
        FS_OUT_Color = FS_IN_ObjPos;
    else
        FS_OUT_Color = texture(s_Lightmap, FS_IN_LightmapUV).rgb;//color;
}

// ------------------------------------------------------------------
