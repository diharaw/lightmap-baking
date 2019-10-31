// ------------------------------------------------------------------
// INPUT VARIABLES  -------------------------------------------------
// ------------------------------------------------------------------

in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------
// OUTPUT VARIABLES  ------------------------------------------------
// ------------------------------------------------------------------

layout(location = 0) out vec4 FS_OUT_Position;
layout(location = 1) out vec4 FS_OUT_Normal;

// ------------------------------------------------------------------
// UNIFORMS  --------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Position;
uniform sampler2D s_Normal;

// ------------------------------------------------------------------
// MAIN  ------------------------------------------------------------
// ------------------------------------------------------------------

void main(void)
{
    ivec2 size = textureSize(s_Position, 0);
    vec2 pixel_offset = vec2(1.0/float(size.x), 1.0/float(size.y));

    {
        vec4 c = texture(s_Position, FS_IN_TexCoord);

        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord - pixel_offset);
        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord + vec2(0, -pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord + vec2(pixel_offset.x, -pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord + vec2(-pixel_offset.x, 0));
        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord + vec2(pixel_offset.x, 0));
        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord + vec2(-pixel_offset.x, pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord + vec2(0, pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Position, FS_IN_TexCoord + pixel_offset);

        FS_OUT_Position = c;
    }

    {
        vec4 c = texture(s_Normal, FS_IN_TexCoord);

        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord - pixel_offset);
        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord + vec2(0, -pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord + vec2(pixel_offset.x, -pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord + vec2(-pixel_offset.x, 0));
        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord + vec2(pixel_offset.x, 0));
        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord + vec2(-pixel_offset.x, pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord + vec2(0, pixel_offset.y));
        c = c.a > 0.0 ? c : texture(s_Normal, FS_IN_TexCoord + pixel_offset);

        FS_OUT_Normal = c;
    }
} 

// ------------------------------------------------------------------
