// ------------------------------------------------------------------
// INPUTS VARIABLES -------------------------------------------------
// ------------------------------------------------------------------

layout(location = 0) in vec3 VS_IN_Position;
layout(location = 1) in vec2 VS_IN_UV;
layout(location = 2) in vec2 VS_IN_LightMapUV;
layout(location = 3) in vec3 VS_IN_Normal;
layout(location = 4) in vec3 VS_IN_Tangent;
layout(location = 5) in vec3 VS_IN_Bitangent;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

out vec3 FS_IN_Position;
out vec3 FS_IN_Normal;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std140) uniform GlobalUniforms
{
    mat4 view_proj;
    mat4 light_view_proj;
    vec4 cam_pos;
};

uniform mat4 u_Model;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    vec4 world_pos = u_Model * vec4(VS_IN_Position, 1.0);
    
    FS_IN_Position = VS_IN_Position;
    FS_IN_Normal = VS_IN_Normal;

    vec2 clip_space_pos = 2.0 * VS_IN_LightMapUV - 1.0;

    gl_Position = vec4(clip_space_pos, 0.0, 1.0);
}

// ------------------------------------------------------------------
