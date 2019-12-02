#pragma once

#include <ogl.h>
#include <vector>
#include <memory>

struct ArHosekSkyModelState;

struct Skybox
{
    ~Skybox();
    bool      initialize(glm::vec3 sun_dir, glm::vec3 ground_albedo, float turbidity);
    void      set_sun_dir(glm::vec3 sun_dir);
    void      render(std::unique_ptr<dw::Framebuffer> fbo, int w, int h, glm::mat4 proj, glm::mat4 view);
    glm::vec3 sample_sky(glm::vec3 dir);

    glm::vec3                           m_sun_dir;
    float                               m_turbidity = 0.0f;
    glm::vec3                           m_ground_albedo;
    float                               m_elevation = 0.0f;
    ArHosekSkyModelState*               m_state_r   = nullptr;
    ArHosekSkyModelState*               m_state_g   = nullptr;
    ArHosekSkyModelState*               m_state_b   = nullptr;
    std::unique_ptr<dw::TextureCube>    m_skybox_texture;
    std::unique_ptr<dw::Shader>         m_skybox_vs;
    std::unique_ptr<dw::Shader>         m_skybox_fs;
    std::unique_ptr<dw::Program>        m_skybox_program;
    std::vector<std::vector<glm::vec4>> m_skybox_data;
};