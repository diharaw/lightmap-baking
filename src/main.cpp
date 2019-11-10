#define _USE_MATH_DEFINES
#include <application.h>
#include <mesh.h>
#include <camera.h>
#include <material.h>
#include <memory>
#include <iostream>
#include <stack>
#include <random>
#include <chrono>
#include <random>
#include <rtccore.h>
#include <rtcore_geometry.h>
#include <rtcore_common.h>
#include <rtcore_ray.h>
#include <rtcore_device.h>
#include <rtcore_scene.h>
#include <xatlas.h>
#include "skybox.h"

#define CAMERA_FAR_PLANE 1000.0f
#define LIGHTMAP_TEXTURE_SIZE 1024
#define LIGHTMAP_CHART_PADDING 6
#define LIGHTMAP_SPP 1
#define LIGHTMAP_BOUNCES 2

struct GlobalUniforms
{
    DW_ALIGNED(16)
    glm::mat4 view_proj;
    DW_ALIGNED(16)
    glm::mat4 light_view_proj;
    DW_ALIGNED(16)
    glm::vec4 cam_pos;
};

struct SubmeshConfig
{
    uint32_t lightmap_idx;
};

struct AtlasConfig
{
    uint32_t width;
    uint32_t height;
};

struct MeshLightmappingConfig
{
    std::vector<AtlasConfig>   atlas_configs;
    std::vector<SubmeshConfig> submesh_configs;
};

struct LightmapVertex
{
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec2 lightmap_uv;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;
};

struct LightmapMesh
{
    std::vector<dw::SubMesh>          submeshes;
    std::unique_ptr<dw::VertexBuffer> vbo;
    std::unique_ptr<dw::IndexBuffer>  ibo;
    std::unique_ptr<dw::VertexArray>  vao;
    std::vector<std::unique_ptr<dw::Texture2D>> lightmaps;
    MeshLightmappingConfig                      config;

	LightmapMesh()
	{

	}

	LightmapMesh(std::vector<LightmapVertex>& vertices, std::vector<uint32_t>& indices, uint32_t num_submeshes, dw::SubMesh* submeshes, MeshLightmappingConfig config)
	{

	}
};

class Lightmaps : public dw::Application
{
protected:
    // -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {
        m_distribution = std::uniform_real_distribution<float>(0.0f, 0.9999999f);

        glm::vec3 default_light_dir = glm::vec3(-0.7500f, 0.9770f, -0.4000f);
        m_light_direction           = -default_light_dir;

        create_lightmap_buffers();

        m_lightmap_texture = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);

        if (!load_cached_lightmap())
            bake_lightmap();

        // Load scene.
        if (!load_scene())
            return false;

        // Create GPU resources.
        if (!create_shaders())
            return false;

        if (!create_uniform_buffer())
            return false;

        create_textures();

        initialize_lightmap();

        // Create camera.
        create_camera();

        m_transform = glm::mat4(1.0f);
        m_transform = glm::scale(m_transform, glm::vec3(0.1f));

        return m_skybox.initialize(default_light_dir, glm::vec3(0.5f), 2.0f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
        if (m_debug_gui)
            gui();

        // Update camera.
        update_camera();

        update_global_uniforms(m_global_uniforms);

        render_lit_scene();

        m_skybox.render(nullptr, m_width, m_height, m_main_camera->m_projection, m_main_camera->m_view);

        if (m_debug_gui)
            visualize_lightmap();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {
        rtcReleaseGeometry(m_embree_triangle_mesh);
        rtcReleaseScene(m_embree_scene);
        rtcReleaseDevice(m_embree_device);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void window_resized(int width, int height) override
    {
        // Override window resized method to update camera projection.
        m_main_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));

        create_textures();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_pressed(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W)
            m_heading_speed = m_camera_speed;
        else if (code == GLFW_KEY_S)
            m_heading_speed = -m_camera_speed;

        // Handle sideways movement.
        if (code == GLFW_KEY_A)
            m_sideways_speed = -m_camera_speed;
        else if (code == GLFW_KEY_D)
            m_sideways_speed = m_camera_speed;

        if (code == GLFW_KEY_SPACE)
            m_mouse_look = true;

        if (code == GLFW_KEY_G)
            m_debug_gui = !m_debug_gui;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_released(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W || code == GLFW_KEY_S)
            m_heading_speed = 0.0f;

        // Handle sideways movement.
        if (code == GLFW_KEY_A || code == GLFW_KEY_D)
            m_sideways_speed = 0.0f;

        if (code == GLFW_KEY_SPACE)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_pressed(int code) override
    {
        // Enable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_released(int code) override
    {
        // Disable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

protected:
    // -----------------------------------------------------------------------------------------------------------------------------------

    dw::AppSettings intial_app_settings() override
    {
        dw::AppSettings settings;

        settings.resizable    = true;
        settings.maximized    = false;
        settings.refresh_rate = 60;
        settings.major_ver    = 4;
        settings.width        = 1920;
        settings.height       = 1080;
        settings.title        = "Lightmaps (c) 2019 Dihara Wijetunga";

        return settings;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    // -----------------------------------------------------------------------------------------------------------------------------------

    void gui()
    {
        if (ImGui::Checkbox("Conservative Rasterization", &m_enable_conservative_raster))
            initialize_lightmap();

        if (ImGui::Checkbox("Bilinear Filtering", &m_bilinear_filtering))
        {
            if (m_bilinear_filtering)
                m_lightmap_texture->set_mag_filter(GL_LINEAR);
            else
                m_lightmap_texture->set_mag_filter(GL_NEAREST);
        }

        ImGui::InputFloat3("Light Direction", &m_light_direction.x);

        ImGui::InputInt("SPP", &m_num_samples);

        if (ImGui::Button("Bake"))
            bake_lightmap();

        if (ImGui::Button("Save to Disk"))
            write_lightmap();

        if (ImGui::Button("Upload"))
            m_lightmap_texture->set_data(0, 0, m_framebuffer.data());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void initialize_lightmap()
    {
        if (m_enable_conservative_raster)
        {
            if (GLAD_GL_NV_conservative_raster)
                glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
            else if (GLAD_GL_INTEL_conservative_rasterization)
                glEnable(GL_INTEL_conservative_rasterization);
        }

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);

        glDisable(GL_CULL_FACE);

        m_lightmap_fbo[0]->bind();

        glViewport(0, 0, LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE);

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind shader program.
        m_lightmap_program->use();

        // Bind vertex array.
        m_unwrapped_mesh.vao->bind();

        for (uint32_t i = 0; i < m_unwrapped_mesh.submeshes.size(); i++)
        {
            dw::SubMesh& submesh = m_unwrapped_mesh.submeshes[i];

            // Issue draw call.
            glDrawElementsBaseVertex(GL_TRIANGLES, submesh.index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * submesh.base_index), submesh.base_vertex);
        }

        if (m_enable_conservative_raster)
        {
            if (GLAD_GL_NV_conservative_raster)
                glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
            else if (GLAD_GL_INTEL_conservative_rasterization)
                glDisable(GL_INTEL_conservative_rasterization);
        }

        glFinish();

        // Dialate

        m_lightmap_fbo[1]->bind();

        glViewport(0, 0, LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE);

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind shader program.
        m_dialate_program->use();

        if (m_dialate_program->set_uniform("s_Position", 0))
            m_lightmap_pos_texture[0]->bind(0);

        if (m_dialate_program->set_uniform("s_Normal", 1))
            m_lightmap_normal_texture[0]->bind(1);

        // Render fullscreen triangle
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glFinish();

        // Copy bake sample points
        GL_CHECK_ERROR(glActiveTexture(GL_TEXTURE0));
        GL_CHECK_ERROR(glBindTexture(m_lightmap_pos_texture[1]->target(), m_lightmap_pos_texture[1]->id()));
        GL_CHECK_ERROR(glGetTexImage(m_lightmap_pos_texture[1]->target(), 0, m_lightmap_pos_texture[1]->format(), m_lightmap_pos_texture[1]->type(), m_ray_positions.data()));

        GL_CHECK_ERROR(glBindTexture(m_lightmap_normal_texture[1]->target(), m_lightmap_normal_texture[1]->id()));
        GL_CHECK_ERROR(glGetTexImage(m_lightmap_normal_texture[1]->target(), 0, m_lightmap_normal_texture[1]->format(), m_lightmap_normal_texture[1]->type(), m_ray_directions.data()));
        GL_CHECK_ERROR(glBindTexture(m_lightmap_normal_texture[1]->target(), 0));

        glFinish();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_lightmap_buffers()
    {
        m_framebuffer.resize(LIGHTMAP_TEXTURE_SIZE * LIGHTMAP_TEXTURE_SIZE);
        m_ray_positions.resize(LIGHTMAP_TEXTURE_SIZE * LIGHTMAP_TEXTURE_SIZE);
        m_ray_directions.resize(LIGHTMAP_TEXTURE_SIZE * LIGHTMAP_TEXTURE_SIZE);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_lit_scene()
    {
        render_scene(nullptr, m_mesh_program, 0, 0, m_width, m_height, GL_BACK);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_shaders()
    {
        {
            // Create general shaders
            m_lightmap_fs           = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/lightmap_fs.glsl"));
            m_mesh_vs               = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/mesh_vs.glsl"));
            m_mesh_fs               = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/mesh_fs.glsl"));
            m_triangle_vs           = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/fullscreen_triangle_vs.glsl"));
            m_lightmap_vs           = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/lightmap_vs.glsl"));
            m_visualize_lightmap_fs = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/visualize_lightmap_fs.glsl"));
            m_dialate_fs            = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/dialate_fs.glsl"));

            {
                if (!m_lightmap_vs || !m_lightmap_fs)
                {
                    DW_LOG_FATAL("Failed to create Shaders");
                    return false;
                }

                // Create general shader program
                dw::Shader* shaders[] = { m_lightmap_vs.get(), m_lightmap_fs.get() };
                m_lightmap_program    = std::make_unique<dw::Program>(2, shaders);

                if (!m_lightmap_program)
                {
                    DW_LOG_FATAL("Failed to create Shader Program");
                    return false;
                }

                m_lightmap_program->uniform_block_binding("GlobalUniforms", 0);
            }

            {
                if (!m_triangle_vs || !m_dialate_fs)
                {
                    DW_LOG_FATAL("Failed to create Shaders");
                    return false;
                }

                // Create general shader program
                dw::Shader* shaders[] = { m_triangle_vs.get(), m_dialate_fs.get() };
                m_dialate_program     = std::make_unique<dw::Program>(2, shaders);

                if (!m_dialate_program)
                {
                    DW_LOG_FATAL("Failed to create Shader Program");
                    return false;
                }
            }

            {
                if (!m_triangle_vs || !m_visualize_lightmap_fs)
                {
                    DW_LOG_FATAL("Failed to create Shaders");
                    return false;
                }

                // Create general shader program
                dw::Shader* shaders[]        = { m_triangle_vs.get(), m_visualize_lightmap_fs.get() };
                m_visualize_lightmap_program = std::make_unique<dw::Program>(2, shaders);

                if (!m_visualize_lightmap_program)
                {
                    DW_LOG_FATAL("Failed to create Shader Program");
                    return false;
                }
            }

            {
                if (!m_mesh_vs || !m_mesh_fs)
                {
                    DW_LOG_FATAL("Failed to create Shaders");
                    return false;
                }

                // Create general shader program
                dw::Shader* shaders[] = { m_mesh_vs.get(), m_mesh_fs.get() };
                m_mesh_program        = std::make_unique<dw::Program>(2, shaders);

                if (!m_mesh_program)
                {
                    DW_LOG_FATAL("Failed to create Shader Program");
                    return false;
                }

                m_mesh_program->uniform_block_binding("GlobalUniforms", 0);
            }
        }

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_textures()
    {
        m_lightmap_pos_texture[0]    = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);
        m_lightmap_pos_texture[1]    = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);
        m_lightmap_normal_texture[0] = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);
        m_lightmap_normal_texture[1] = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGBA32F, GL_RGBA, GL_FLOAT);

        m_lightmap_texture->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        m_lightmap_pos_texture[0]->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        m_lightmap_pos_texture[1]->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        m_lightmap_normal_texture[0]->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        m_lightmap_normal_texture[1]->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

        m_lightmap_fbo[0] = std::make_unique<dw::Framebuffer>();
        m_lightmap_fbo[1] = std::make_unique<dw::Framebuffer>();

        {
            dw::Texture* textures[] = { m_lightmap_pos_texture[0].get(), m_lightmap_normal_texture[0].get() };

            m_lightmap_fbo[0]->attach_multiple_render_targets(2, textures);
        }

        {
            dw::Texture* textures[] = { m_lightmap_pos_texture[1].get(), m_lightmap_normal_texture[1].get() };

            m_lightmap_fbo[1]->attach_multiple_render_targets(2, textures);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_uniform_buffer()
    {
        // Create uniform buffer for global data
        m_global_ubo = std::make_unique<dw::UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(GlobalUniforms));

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_scene()
    {
        dw::Mesh* mesh = dw::Mesh::load("mesh/sponza.obj");

        if (!mesh)
        {
            DW_LOG_FATAL("Failed to load mesh!");
            return false;
        }

        if (!lightmap_uv_unwrap(mesh))
            return false;

        if (!initialize_embree(mesh))
            return false;

        dw::Mesh::unload(mesh);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_lightmap_uv_unwrapped_mesh(xatlas::Atlas* atlas, dw::Mesh* mesh)
    {
        dw::Vertex* vertex_ptr = mesh->vertices();

        std::vector<LightmapVertex> vertices;
        std::vector<uint32_t>       indices;

        for (int i = 0; i < mesh->sub_mesh_count(); i++)
            m_unwrapped_mesh.submeshes.push_back(mesh->sub_meshes()[i]);

        uint32_t index_count  = 0;
        uint32_t vertex_count = 0;

        for (int mesh_idx = 0; mesh_idx < atlas->meshCount; mesh_idx++)
        {
            dw::SubMesh& sub = m_unwrapped_mesh.submeshes[mesh_idx];

            sub.base_index  = index_count;
            sub.base_vertex = vertex_count;

            for (int i = 0; i < atlas->meshes[mesh_idx].vertexCount; i++)
            {
                int idx = atlas->meshes[mesh_idx].vertexArray[i].xref;

                LightmapVertex v;

                v.position    = vertex_ptr[idx].position;
                v.uv          = vertex_ptr[idx].tex_coord;
                v.normal      = vertex_ptr[idx].normal;
                v.tangent     = vertex_ptr[idx].tangent;
                v.bitangent   = vertex_ptr[idx].bitangent;
                v.lightmap_uv = glm::vec2(atlas->meshes[mesh_idx].vertexArray[i].uv[0] / (atlas->width - 1), atlas->meshes[mesh_idx].vertexArray[i].uv[1] / (atlas->height - 1));

                vertices.push_back(v);
            }

            for (int i = 0; i < atlas->meshes[mesh_idx].indexCount; i++)
                indices.push_back(atlas->meshes[mesh_idx].indexArray[i]);

            index_count += atlas->meshes[mesh_idx].indexCount;
            vertex_count += atlas->meshes[mesh_idx].vertexCount;
        }

        // Create vertex buffer.
        m_unwrapped_mesh.vbo = std::make_unique<dw::VertexBuffer>(GL_STATIC_DRAW, sizeof(LightmapVertex) * vertices.size(), vertices.data());

        // Create index buffer.
        m_unwrapped_mesh.ibo = std::make_unique<dw::IndexBuffer>(GL_STATIC_DRAW, sizeof(uint32_t) * indices.size(), indices.data());

        // Declare vertex attributes.
        dw::VertexAttrib attribs[] = { { 3, GL_FLOAT, false, 0 },
                                       { 2, GL_FLOAT, false, offsetof(LightmapVertex, uv) },
                                       { 2, GL_FLOAT, false, offsetof(LightmapVertex, lightmap_uv) },
                                       { 3, GL_FLOAT, false, offsetof(LightmapVertex, normal) },
                                       { 3, GL_FLOAT, false, offsetof(LightmapVertex, tangent) },
                                       { 3, GL_FLOAT, false, offsetof(LightmapVertex, bitangent) } };

        // Create vertex array.
        m_unwrapped_mesh.vao = std::make_unique<dw::VertexArray>(m_unwrapped_mesh.vbo.get(), m_unwrapped_mesh.ibo.get(), sizeof(LightmapVertex), 6, attribs);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool lightmap_uv_unwrap(dw::Mesh* mesh)
    {
        std::vector<glm::vec3> positions(mesh->vertex_count());
        std::vector<glm::vec3> normals(mesh->vertex_count());
        std::vector<glm::vec2> uvs(mesh->vertex_count());

        xatlas::Atlas* atlas = xatlas::Create();

        int         idx        = 0;
        dw::Vertex* vertex_ptr = mesh->vertices();
        uint32_t*   index_ptr  = mesh->indices();

        for (int i = 0; i < mesh->vertex_count(); i++)
        {
            positions[i] = vertex_ptr[i].position;
            normals[i]   = vertex_ptr[i].normal;
            uvs[i]       = vertex_ptr[i].tex_coord;
        }

        for (int mesh_idx = 0; mesh_idx < mesh->sub_mesh_count(); mesh_idx++)
        {
            dw::SubMesh& submesh = mesh->sub_meshes()[mesh_idx];

            xatlas::MeshDecl mesh_decl;

            mesh_decl.vertexCount          = mesh->vertex_count();
            mesh_decl.vertexPositionStride = sizeof(glm::vec3);
            mesh_decl.vertexPositionData   = &positions[0];
            mesh_decl.vertexNormalStride   = sizeof(glm::vec3);
            mesh_decl.vertexNormalData     = &normals[0];
            mesh_decl.vertexUvStride       = sizeof(glm::vec2);
            mesh_decl.vertexUvData         = &uvs[0];
            mesh_decl.indexCount           = submesh.index_count;
            mesh_decl.indexData            = &mesh->indices()[submesh.base_index];
            mesh_decl.indexOffset          = submesh.base_vertex;
            mesh_decl.indexFormat          = xatlas::IndexFormat::UInt32;

            xatlas::AddMeshError::Enum error = xatlas::AddMesh(atlas, mesh_decl);

            if (error != xatlas::AddMeshError::Success)
            {
                xatlas::Destroy(atlas);
                DW_LOG_ERROR("Failed to add UV mesh to Lightmap Atlas");
                return false;
            }
        }

        xatlas::ComputeCharts(atlas);
        xatlas::ParameterizeCharts(atlas);

        xatlas::PackOptions pack_options;

        pack_options.padding    = LIGHTMAP_CHART_PADDING;
        pack_options.resolution = LIGHTMAP_TEXTURE_SIZE;

        xatlas::PackCharts(atlas, pack_options);

        bool status = create_lightmap_uv_unwrapped_mesh(atlas, mesh);
        xatlas::Destroy(atlas);

        return status;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool initialize_embree(dw::Mesh* mesh)
    {
        m_embree_device = rtcNewDevice(nullptr);

        RTCError embree_error = rtcGetDeviceError(m_embree_device);

        if (embree_error == RTC_ERROR_UNSUPPORTED_CPU)
            throw std::runtime_error("Your CPU does not meet the minimum requirements for embree");
        else if (embree_error != RTC_ERROR_NONE)
            throw std::runtime_error("Failed to initialize embree!");

        m_embree_scene = rtcNewScene(m_embree_device);

        rtcSetSceneFlags(m_embree_scene, RTC_SCENE_FLAG_ROBUST);

        m_embree_triangle_mesh = rtcNewGeometry(m_embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

        std::vector<glm::vec3> vertices(mesh->vertex_count());
        std::vector<uint32_t>  indices(mesh->index_count());
        uint32_t               idx        = 0;
        dw::Vertex*            vertex_ptr = mesh->vertices();
        uint32_t*              index_ptr  = mesh->indices();

        for (int i = 0; i < mesh->vertex_count(); i++)
            vertices[i] = vertex_ptr[i].position;

        for (int i = 0; i < mesh->sub_mesh_count(); i++)
        {
            dw::SubMesh& submesh = mesh->sub_meshes()[i];

            for (int j = submesh.base_index; j < (submesh.base_index + submesh.index_count); j++)
                indices[idx++] = submesh.base_vertex + index_ptr[j];
        }

        void* data = rtcSetNewGeometryBuffer(m_embree_triangle_mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(glm::vec3), mesh->vertex_count());
        memcpy(data, vertices.data(), vertices.size() * sizeof(glm::vec3));

        data = rtcSetNewGeometryBuffer(m_embree_triangle_mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), mesh->index_count() / 3);
        memcpy(data, indices.data(), indices.size() * sizeof(uint32_t));

        rtcCommitGeometry(m_embree_triangle_mesh);
        rtcAttachGeometry(m_embree_scene, m_embree_triangle_mesh);
        rtcCommitScene(m_embree_scene);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_camera()
    {
        m_main_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height), glm::vec3(150.0f, 20.0f, 0.0f), glm::vec3(-1.0f, 0.0, 0.0f));
        m_main_camera->set_rotatation_delta(glm::vec3(0.0f, -90.0f, 0.0f));
        m_main_camera->update();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_mesh(LightmapMesh& mesh, glm::mat4 model, std::unique_ptr<dw::Program>& program)
    {
        program->set_uniform("u_Model", model);

        // Bind vertex array.
        mesh.vao->bind();

        for (uint32_t i = 0; i < mesh.submeshes.size(); i++)
        {
            dw::SubMesh& submesh = mesh.submeshes[i];

            if (program->set_uniform("s_Lightmap", 0))
                m_lightmap_texture->bind(0);

            // Issue draw call.
            glDrawElementsBaseVertex(GL_TRIANGLES, submesh.index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * submesh.base_index), submesh.base_vertex);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_scene(dw::Framebuffer* fbo, std::unique_ptr<dw::Program>& program, int x, int y, int w, int h, GLenum cull_face, bool clear = true)
    {
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);

        if (cull_face == GL_NONE)
            glDisable(GL_CULL_FACE);
        else
        {
            glEnable(GL_CULL_FACE);
            glCullFace(cull_face);
        }

        if (fbo)
            fbo->bind();
        else
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glViewport(x, y, w, h);

        if (clear)
        {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClearDepth(1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        // Bind shader program.
        program->use();

        // Bind uniform buffers.
        m_global_ubo->bind_base(0);

        // Draw scene.
        render_mesh(m_unwrapped_mesh, m_transform, program);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void visualize_lightmap()
    {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, 512, 512);

        // Bind shader program.
        m_visualize_lightmap_program->use();

        if (m_visualize_lightmap_program->set_uniform("s_Lightmap", 0))
            m_lightmap_texture->bind(0);

        // Render fullscreen triangle
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    float drand48()
    {
        return m_distribution(m_generator);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    glm::vec3 diffuse_lambert(glm::vec3 albedo)
    {
        return albedo;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool valid_texel(glm::vec3 t)
    {
        return !(t.x == 0.0f && t.y == 0.0f && t.z == 0.0f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    glm::vec3 sample_direction(int x, int y)
    {
        return m_ray_directions[LIGHTMAP_TEXTURE_SIZE * y + x];
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    glm::vec3 sample_position(int x, int y)
    {
        return m_ray_positions[LIGHTMAP_TEXTURE_SIZE * y + x];
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool valid_position(int x, int y)
    {
        glm::vec3 p = sample_position(x, y);
        return !(p.x == 0.0f && p.y == 0.0f && p.z == 0.0f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool is_nan(glm::vec3 v)
    {
        glm::bvec3 b = glm::isnan(v);
        return b.x || b.y || b.z;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    glm::mat3 make_rotation_matrix(glm::vec3 z)
    {
        const glm::vec3 ref = glm::abs(glm::dot(z, glm::vec3(0, 1, 0))) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

        const glm::vec3 x = glm::normalize(glm::cross(ref, z));
        const glm::vec3 y = glm::cross(z, x);

        assert(!is_nan(x));
        assert(!is_nan(y));
        assert(!is_nan(z));

        return { x, y, z };
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

#undef max

    glm::vec3 sample_cosine_lobe_direction(glm::vec3 n)
    {
        glm::vec2 sample = glm::max(glm::vec2(0.00001f), glm::vec2(drand48(), drand48()));

        const float phi = 2.0f * M_PI * sample.y;

        const float cos_theta = sqrt(sample.x);
        const float sin_theta = sqrt(1 - sample.x);

        glm::vec3 t = glm::vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

        assert(!is_nan(t));

        return glm::normalize(make_rotation_matrix(n) * t);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_ray(glm::vec3 direction, glm::vec3 position, RTCRayHit& rayhit)
    {
        rayhit.ray.dir_x = direction.x;
        rayhit.ray.dir_y = direction.y;
        rayhit.ray.dir_z = direction.z;

        rayhit.ray.org_x = position.x;
        rayhit.ray.org_y = position.y;
        rayhit.ray.org_z = position.z;

        rayhit.ray.tnear     = 0;
        rayhit.ray.tfar      = INFINITY;
        rayhit.ray.mask      = -1;
        rayhit.ray.flags     = 0;
        rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    glm::vec3 evaluate_direct_lighting(RTCIntersectContext& context, glm::vec3 p, glm::vec3 n)
    {
        const glm::vec3 l      = -m_light_direction;
        const glm::vec3 albedo = glm::vec3(0.7f);
        const glm::vec3 li     = glm::vec3(1.0f);

        RTCRay rayhit;

        rayhit.dir_x = l.x;
        rayhit.dir_y = l.y;
        rayhit.dir_z = l.z;

        rayhit.org_x = p.x;
        rayhit.org_y = p.y;
        rayhit.org_z = p.z;

        rayhit.tnear = 0;
        rayhit.tfar  = INFINITY;
        rayhit.mask  = -1;
        rayhit.flags = 0;

        rtcOccluded1(m_embree_scene, &context, &rayhit);

        // Is it visible?
        if (rayhit.tfar == INFINITY)
            return li * diffuse_lambert(albedo) * glm::max(glm::dot(n, l), 0.0f);

        return glm::vec3(0.0f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    glm::vec3 path_trace(glm::vec3 direction, glm::vec3 position)
    {
        glm::vec3       color;
        const glm::vec3 albedo = glm::vec3(0.7f);

        RTCRayHit rayhit;

        glm::vec3 p = position;
        glm::vec3 n = direction;
        glm::vec3 d = direction;

        p += glm::sign(n) * abs(p * 0.0000002f);

        color                 = glm::vec3(0.0f);
        glm::vec3 attenuation = glm::vec3(1.0f);

        for (int i = 0; i < LIGHTMAP_BOUNCES; i++)
        {
            RTCIntersectContext intersect_context;
            rtcInitIntersectContext(&intersect_context);

            d = sample_cosine_lobe_direction(n);

            create_ray(d, p, rayhit);

            rtcIntersect1(m_embree_scene, &intersect_context, &rayhit);

            // Does intersect scene
            if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            {
                float sky_dir = d.y < 0.0f ? 0.0f : 1.0f;
                return m_skybox.sample_sky(d) * sky_dir;
            }

            p = p + d * rayhit.ray.tfar;

            // Add bias to position
            p += glm::sign(n) * abs(p * 0.0000002f);

            n = glm::normalize(glm::vec3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));

            color += evaluate_direct_lighting(intersect_context, p, n) * attenuation;

            attenuation *= albedo;
        }

        return color;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_cached_lightmap()
    {
        FILE* lm = fopen("lightmap.raw", "r");

        if (lm)
        {
            size_t n = sizeof(float) * LIGHTMAP_TEXTURE_SIZE * LIGHTMAP_TEXTURE_SIZE * 4;

            fread(m_framebuffer.data(), n, 1, lm);

            m_lightmap_texture->set_data(0, 0, m_framebuffer.data());

            fclose(lm);

            return true;
        }
        else
            return false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void write_lightmap()
    {
        FILE* lm = fopen("lightmap.raw", "wb");

        size_t n = sizeof(float) * LIGHTMAP_TEXTURE_SIZE * LIGHTMAP_TEXTURE_SIZE * 4;

        fwrite(m_framebuffer.data(), n, 1, lm);

        fclose(lm);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void bake_lightmap()
    {
        glFinish();

        float w = 1.0f / float(m_num_samples);

#pragma omp parallel for
        for (int y = 0; y < LIGHTMAP_TEXTURE_SIZE; y++)
        {
            for (int x = 0; x < LIGHTMAP_TEXTURE_SIZE; x++)
            {
                glm::vec3 color    = glm::vec3(0.0f);
                glm::vec3 normal   = sample_direction(x, y);
                glm::vec3 position = sample_position(x, y);

                // Check if this is a valid lightmap texel
                if (valid_texel(normal))
                {
                    for (int sample = 0; sample < m_num_samples; sample++)
                        color += path_trace(normal, position) * w;
                }

                m_framebuffer[LIGHTMAP_TEXTURE_SIZE * y + x] = glm::vec4(color, 1.0f);
            }
        }

        m_lightmap_texture->set_data(0, 0, m_framebuffer.data());

        glFinish();

        write_lightmap();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_global_uniforms(const GlobalUniforms& global)
    {
        void* ptr = m_global_ubo->map(GL_WRITE_ONLY);
        memcpy(ptr, &global, sizeof(GlobalUniforms));
        m_global_ubo->unmap();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_transforms(dw::Camera* camera)
    {
        // Update camera matrices.
        m_global_uniforms.view_proj = camera->m_projection * camera->m_view;
        m_global_uniforms.cam_pos   = glm::vec4(camera->m_position, 0.0f);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_camera()
    {
        dw::Camera* current = m_main_camera.get();

        float forward_delta = m_heading_speed * m_delta;
        float right_delta   = m_sideways_speed * m_delta;

        current->set_translation_delta(current->m_forward, forward_delta);
        current->set_translation_delta(current->m_right, right_delta);

        m_camera_x = m_mouse_delta_x * m_camera_sensitivity;
        m_camera_y = m_mouse_delta_y * m_camera_sensitivity;

        if (m_mouse_look)
        {
            // Activate Mouse Look
            current->set_rotatation_delta(glm::vec3((float)(m_camera_y),
                                                    (float)(m_camera_x),
                                                    (float)(0.0f)));
        }
        else
        {
            current->set_rotatation_delta(glm::vec3((float)(0),
                                                    (float)(0),
                                                    (float)(0)));
        }

        current->update();
        update_transforms(current);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    // General GPU resources.
    std::unique_ptr<dw::Shader> m_lightmap_fs;
    std::unique_ptr<dw::Shader> m_dialate_fs;
    std::unique_ptr<dw::Shader> m_mesh_fs;
    std::unique_ptr<dw::Shader> m_visualize_lightmap_fs;

    std::unique_ptr<dw::Shader> m_lightmap_vs;
    std::unique_ptr<dw::Shader> m_triangle_vs;
    std::unique_ptr<dw::Shader> m_mesh_vs;

    std::unique_ptr<dw::Program> m_lightmap_program;
    std::unique_ptr<dw::Program> m_dialate_program;
    std::unique_ptr<dw::Program> m_visualize_lightmap_program;
    std::unique_ptr<dw::Program> m_mesh_program;

    std::unique_ptr<dw::Texture2D> m_lightmap_texture;
    std::unique_ptr<dw::Texture2D> m_lightmap_pos_texture[2];
    std::unique_ptr<dw::Texture2D> m_lightmap_normal_texture[2];

    std::unique_ptr<dw::Framebuffer> m_lightmap_fbo[2];

    std::unique_ptr<dw::UniformBuffer> m_global_ubo;

    std::vector<glm::vec4> m_ray_positions;
    std::vector<glm::vec4> m_ray_directions;
    std::vector<glm::vec4> m_framebuffer;

    // Camera.
    LightmapMesh                m_unwrapped_mesh;
    std::unique_ptr<dw::Camera> m_main_camera;

    GlobalUniforms m_global_uniforms;

    // Scene
    glm::mat4 m_transform;

    // Camera controls.
    bool  m_mouse_look         = false;
    float m_heading_speed      = 0.0f;
    float m_sideways_speed     = 0.0f;
    float m_camera_sensitivity = 0.05f;
    float m_camera_speed       = 0.02f;
    bool  m_debug_gui          = true;
    int   m_num_samples        = LIGHTMAP_SPP;

    // Embree structure
    RTCDevice   m_embree_device        = nullptr;
    RTCScene    m_embree_scene         = nullptr;
    RTCGeometry m_embree_triangle_mesh = nullptr;

    glm::vec3 m_hit_pos;
    bool      is_hit                       = false;
    bool      m_enable_conservative_raster = true;
    bool      m_bilinear_filtering         = true;

    std::default_random_engine            m_generator;
    std::uniform_real_distribution<float> m_distribution;

    glm::vec3 m_light_direction;
    Skybox    m_skybox;

    // Camera orientation.
    float m_camera_x;
    float m_camera_y;
};

DW_DECLARE_MAIN(Lightmaps)