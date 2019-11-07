#define _USE_MATH_DEFINES
#include <math.h>
#include "skybox.h"
#include <ArHosekSkyModel.h>
#include <logger.h>
#include <macros.h>

#define SKYBOX_TEXTURE_SIZE 1024
#define FP16_SCALE 0.0009765625f;
static const float Pi   = 3.141592654f;
static const float Pi2  = 6.283185307f;
static const float Pi_2 = 1.570796327f;
static const float Pi_4 = 0.7853981635f;

// -----------------------------------------------------------------------------------------------------------------------------------

static float angle_between(const glm::vec3& dir0, const glm::vec3& dir1)
{
    return glm::acos(glm::max(glm::dot(dir0, dir1), 0.00001f));
}

// -----------------------------------------------------------------------------------------------------------------------------------

static glm::vec3 map_xys_to_direction(uint64_t x, uint64_t y, uint64_t s, uint64_t width, uint64_t height)
{
    float u = ((x + 0.5f) / float(width)) * 2.0f - 1.0f;
    float v = ((y + 0.5f) / float(height)) * 2.0f - 1.0f;
    v *= -1.0f;

    glm::vec3 dir = glm::vec3(0.0f);

    // +x, -x, +y, -y, +z, -z
    switch (s)
    {
		case 0:
		    dir = glm::normalize(glm::vec3(1.0f, v, -u));
		    break;
		case 1:
		    dir = glm::normalize(glm::vec3(-1.0f, v, u));
		    break;
		case 2:
		    dir = glm::normalize(glm::vec3(u, 1.0f, -v));
		    break;
		case 3:
		    dir = glm::normalize(glm::vec3(u, -1.0f, v));
		    break;
		case 4:
		    dir = glm::normalize(glm::vec3(u, v, 1.0f));
		    break;
		case 5:
		    dir = glm::normalize(glm::vec3(-u, v, -1.0f));
		    break;
    }

    return dir;
}

// -----------------------------------------------------------------------------------------------------------------------------------

Skybox::~Skybox()
{
    DW_SAFE_DELETE(m_state_r);
    DW_SAFE_DELETE(m_state_g);
    DW_SAFE_DELETE(m_state_b);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool Skybox::initialize(glm::vec3 sun_dir, glm::vec3 ground_albedo, float turbidity)
{
    m_ground_albedo = ground_albedo;
    m_turbidity     = turbidity;

    m_skybox_texture = std::make_unique<dw::TextureCube>(SKYBOX_TEXTURE_SIZE, SKYBOX_TEXTURE_SIZE, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    m_skybox_texture->set_mag_filter(GL_NEAREST);
    m_skybox_texture->set_min_filter(GL_NEAREST);

	m_skybox_data.resize(6);

	for (int i = 0; i < 6; i++)
		m_skybox_data[i].resize(SKYBOX_TEXTURE_SIZE * SKYBOX_TEXTURE_SIZE);

	set_sun_dir(sun_dir);

	m_skybox_vs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/skybox_vs.glsl"));
    m_skybox_fs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/skybox_fs.glsl"));

	if (!m_skybox_vs || !m_skybox_fs)
     {
         DW_LOG_FATAL("Failed to create Shaders");
         return false;
     }

     // Create general shader program
     dw::Shader* shaders[] = { m_skybox_vs.get(), m_skybox_fs.get() };
     m_skybox_program    = std::make_unique<dw::Program>(2, shaders);

     if (!m_skybox_program)
     {
         DW_LOG_FATAL("Failed to create Shader Program");
         return false;
     }

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Skybox::set_sun_dir(glm::vec3 sun_dir)
{
    DW_SAFE_DELETE(m_state_r);
    DW_SAFE_DELETE(m_state_g);
    DW_SAFE_DELETE(m_state_b);

	sun_dir.y       = glm::clamp(sun_dir.y, 0.0f, 1.0f);
    sun_dir         = glm::normalize(sun_dir);
    float thetaS    = angle_between(sun_dir, glm::vec3(0.0f, 1.0f, 0.0f));
    float elevation = Pi_2 - thetaS;
    m_elevation     = elevation;
    m_sun_dir       = sun_dir;

    m_state_r = arhosek_rgb_skymodelstate_alloc_init(m_turbidity, m_ground_albedo.x, m_elevation);
    m_state_g = arhosek_rgb_skymodelstate_alloc_init(m_turbidity, m_ground_albedo.y, m_elevation);
    m_state_b = arhosek_rgb_skymodelstate_alloc_init(m_turbidity, m_ground_albedo.z, m_elevation);

	for (int s = 0; s < 6; s++)
	{
		for (int y = 0; y < SKYBOX_TEXTURE_SIZE; y++)
		{
		    for (int x = 0; x < SKYBOX_TEXTURE_SIZE; x++)
		    {
                glm::vec3 dir      = map_xys_to_direction(x, y, s, SKYBOX_TEXTURE_SIZE, SKYBOX_TEXTURE_SIZE);
                glm::vec3 radiance = sample_sky(dir);
		        uint64_t idx       = (y * SKYBOX_TEXTURE_SIZE) + x;
                m_skybox_data[s][idx] = (glm::vec4(radiance, 1.0f));
		    }
		}

		m_skybox_texture->set_data(s, 0, 0, m_skybox_data[s].data());
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Skybox::render(std::unique_ptr<dw::Framebuffer> fbo, int w, int h, glm::mat4 proj, glm::mat4 view)
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    if (fbo)
		fbo->bind();
    else
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glViewport(0, 0, w, h);

    // Bind shader program.
    m_skybox_program->use();

	glm::mat4 inverse_vp = glm::inverse(proj * glm::mat4(glm::mat3(view)));

	m_skybox_program->set_uniform("u_CubemapInverseVP", inverse_vp);

    if (m_skybox_program->set_uniform("s_Skybox", 0))
        m_skybox_texture->bind(0);

    // Render fullscreen triangle
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glDepthFunc(GL_LESS);
}

// -----------------------------------------------------------------------------------------------------------------------------------

glm::vec3 Skybox::sample_sky(glm::vec3 dir)
{
    float gamma = angle_between(dir, m_sun_dir);
    float theta = angle_between(dir, glm::vec3(0.0f, 1.0f, 0.0f));

    glm::vec3 radiance;

    radiance.x = float(arhosek_tristim_skymodel_radiance(m_state_r, theta, gamma, 0));
    radiance.y = float(arhosek_tristim_skymodel_radiance(m_state_g, theta, gamma, 1));
    radiance.z = float(arhosek_tristim_skymodel_radiance(m_state_b, theta, gamma, 2));

    // Multiply by standard luminous efficacy of 683 lm/W to bring us in line with the photometric
    // units used during rendering
    radiance *= 683.0f;

    radiance *= FP16_SCALE;

    return radiance;
}

// -----------------------------------------------------------------------------------------------------------------------------------