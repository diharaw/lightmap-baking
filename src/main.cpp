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

#define CAMERA_FAR_PLANE 1000.0f
#define LIGHTMAP_TEXTURE_SIZE 4096

struct GlobalUniforms
{
    DW_ALIGNED(16)
    glm::mat4 view_proj;
    DW_ALIGNED(16)
    glm::mat4 light_view_proj;
    DW_ALIGNED(16)
    glm::vec4 cam_pos;
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


class Lightmaps : public dw::Application
{
protected:
    // -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {
        // Load scene.
        if (!load_scene())
            return false;

		if (!lightmap_uv_unwrap())
            return false;

        // Create GPU resources.
        if (!create_shaders())
            return false;

        if (!create_uniform_buffer())
            return false;

        if (!initialize_embree())
            return false;

        create_framebuffers();

        // Create camera.
        create_camera();

        m_transform = glm::mat4(1.0f);

        initialize_lightmap();

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
        // Update camera.
        update_camera();

        update_global_uniforms(m_global_uniforms);

        //render_lit_scene();
        visualize_lightmap();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {
        rtcReleaseGeometry(m_embree_triangle_mesh);
        rtcReleaseScene(m_embree_scene);
        rtcReleaseDevice(m_embree_device);

        dw::Mesh::unload(m_mesh);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void window_resized(int width, int height) override
    {
        // Override window resized method to update camera projection.
        m_main_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));

        create_framebuffers();
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
        if (code == GLFW_MOUSE_BUTTON_LEFT)
        {
            //double xpos, ypos;
            //glfwGetCursorPos(m_window, &xpos, &ypos);

            //glm::vec4 ndc_pos      = glm::vec4((2.0f * float(xpos)) / float(m_width) - 1.0f, 1.0 - (2.0f * float(ypos)) / float(m_height), -1.0f, 1.0f);
            //glm::vec4 view_coords  = glm::inverse(m_main_camera->m_projection) * ndc_pos;
            //glm::vec4 world_coords = glm::inverse(m_main_camera->m_view) * glm::vec4(view_coords.x, view_coords.y, -1.0f, 0.0f);

            //glm::vec3 ray_dir = glm::normalize(glm::vec3(world_coords));

            //RTCRayHit rayhit;

            //rayhit.ray.dir_x = ray_dir.x;
            //rayhit.ray.dir_y = ray_dir.y;
            //rayhit.ray.dir_z = ray_dir.z;

            //rayhit.ray.org_x = m_main_camera->m_position.x;
            //rayhit.ray.org_y = m_main_camera->m_position.y;
            //rayhit.ray.org_z = m_main_camera->m_position.z;

            //rayhit.ray.tnear     = 0;
            //rayhit.ray.tfar      = INFINITY;
            //rayhit.ray.mask      = 0;
            //rayhit.ray.flags     = 0;
            //rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
            //rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

            //rtcIntersect1(m_embree_scene, &m_embree_intersect_context, &rayhit);

            //if (rayhit.ray.tfar != INFINITY)
            //{
            //   
            //}
            //else
            //    rayhit.ray.tfar = INFINITY;
        }

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

        m_lightmap_fbo->bind();

        glViewport(0, 0, LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind shader program.
        m_lightmap_program->use();

        // Bind uniform buffers.
        m_global_ubo->bind_base(0);

        m_lightmap_program->set_uniform("u_Model", m_transform);

        // Bind vertex array.
        m_uv_unwrapped_vao->bind();

        dw::SubMesh* submeshes = m_mesh->sub_meshes();

        for (uint32_t i = 0; i < m_mesh->sub_mesh_count(); i++)
        {
            dw::SubMesh& submesh = submeshes[i];

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
            m_lightmap_fs      = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/lightmap_fs.glsl"));
            m_mesh_vs          = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/mesh_vs.glsl"));
            m_mesh_fs          = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/mesh_fs.glsl"));
            m_triangle_vs      = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/fullscreen_triangle_vs.glsl"));
            m_lightmap_vs      = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_VERTEX_SHADER, "shader/lightmap_vs.glsl"));
            m_visualize_lightmap_fs      = std::unique_ptr<dw::Shader>(dw::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/visualize_lightmap_fs.glsl"));

            {
                if (!m_lightmap_vs || !m_lightmap_fs)
                {
                    DW_LOG_FATAL("Failed to create Shaders");
                    return false;
                }

                // Create general shader program
                dw::Shader* shaders[] = { m_lightmap_vs.get(), m_lightmap_fs.get() };
                m_lightmap_program       = std::make_unique<dw::Program>(2, shaders);

                if (!m_lightmap_program)
                {
                    DW_LOG_FATAL("Failed to create Shader Program");
                    return false;
                }

                m_lightmap_program->uniform_block_binding("GlobalUniforms", 0);
            }

			{
                if (!m_triangle_vs || !m_visualize_lightmap_fs)
                {
                    DW_LOG_FATAL("Failed to create Shaders");
                    return false;
                }

                // Create general shader program
                dw::Shader* shaders[] = { m_triangle_vs.get(), m_visualize_lightmap_fs.get() };
                m_visualize_lightmap_program    = std::make_unique<dw::Program>(2, shaders);

                if (!m_visualize_lightmap_program)
                {
                    DW_LOG_FATAL("Failed to create Shader Program");
                    return false;
                }

                m_visualize_lightmap_program->uniform_block_binding("GlobalUniforms", 0);
            }

            {
                if (!m_mesh_vs || !m_mesh_fs)
                {
                    DW_LOG_FATAL("Failed to create Shaders");
                    return false;
                }

                // Create general shader program
                dw::Shader* shaders[]  = { m_mesh_vs.get(), m_mesh_fs.get() };
                m_mesh_program = std::make_unique<dw::Program>(2, shaders);

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

    void create_framebuffers()
    {
        m_lightmap_texture = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT);
        m_lightmap_pos_texture = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT);
        m_lightmap_normal_texture = std::make_unique<dw::Texture2D>(LIGHTMAP_TEXTURE_SIZE, LIGHTMAP_TEXTURE_SIZE, 1, 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT);

        m_lightmap_texture->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        m_lightmap_texture->set_min_filter(GL_LINEAR_MIPMAP_LINEAR);
        m_lightmap_texture->set_mag_filter(GL_LINEAR);

        m_lightmap_fbo = std::make_unique<dw::Framebuffer>();

		dw::Texture* textures[] = { m_lightmap_pos_texture.get(), m_lightmap_normal_texture.get() };

        m_lightmap_fbo->attach_multiple_render_targets(2, textures);
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
        m_mesh = dw::Mesh::load("mesh/sponza.obj");

        if (!m_mesh)
        {
            DW_LOG_FATAL("Failed to load mesh!");
            return false;
        }

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

	bool create_lightmap_uv_unwrapped_mesh(xatlas::Atlas* atlas)
    {
        dw::Vertex* vertex_ptr = m_mesh->vertices();

		std::vector<LightmapVertex> vertices(m_mesh->vertex_count());

		for (int i = 0; i < atlas->meshes[0].vertexCount; i++)
		{
			vertices[i].position = vertex_ptr[i].position;
			vertices[i].uv = vertex_ptr[i].tex_coord;
            vertices[i].normal = vertex_ptr[i].normal;
            vertices[i].tangent = vertex_ptr[i].tangent;
            vertices[i].bitangent = vertex_ptr[i].bitangent;
		}

		for (int i = 0; i < atlas->meshes[0].vertexCount; i++)
		{
            int idx                 = atlas->meshes[0].vertexArray[i].xref;
            vertices[idx].lightmap_uv = glm::vec2(atlas->meshes[0].vertexArray[i].uv[0] / atlas->width, atlas->meshes[0].vertexArray[i].uv[1] / atlas->height);
		}

		// Create vertex buffer.
        m_uv_unwrapped_vbo = std::make_unique<dw::VertexBuffer>(GL_STATIC_DRAW, sizeof(LightmapVertex) * vertices.size(), vertices.data());

       // Create index buffer.
        m_uv_unwrapped_ibo = std::make_unique<dw::IndexBuffer>(GL_STATIC_DRAW, sizeof(uint32_t) * m_mesh->index_count(), m_mesh->indices());

       // Declare vertex attributes.
       dw::VertexAttrib attribs[] = { { 3, GL_FLOAT, false, 0 },
                                    { 2, GL_FLOAT, false, offsetof(LightmapVertex, uv) },
                                    { 2, GL_FLOAT, false, offsetof(LightmapVertex, lightmap_uv) },
                                    { 3, GL_FLOAT, false, offsetof(LightmapVertex, normal) },
                                    { 3, GL_FLOAT, false, offsetof(LightmapVertex, tangent) },
                                    { 3, GL_FLOAT, false, offsetof(LightmapVertex, bitangent) } };

       // Create vertex array.
       m_uv_unwrapped_vao = std::make_unique<dw::VertexArray>(m_uv_unwrapped_vbo.get(), m_uv_unwrapped_ibo.get(), sizeof(LightmapVertex), 6, attribs);

		return true;
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool lightmap_uv_unwrap()
	{
		std::vector<glm::vec2> uv(m_mesh->vertex_count());
		std::vector<uint32_t>  indices(m_mesh->index_count());
		
		xatlas::Atlas* atlas = xatlas::Create();

		xatlas::UvMeshDecl mesh_decl;
		
		mesh_decl.vertexCount  = m_mesh->vertex_count();
		mesh_decl.vertexStride = sizeof(glm::vec2);
		mesh_decl.indexCount   = m_mesh->index_count();
		mesh_decl.indexFormat  = xatlas::IndexFormat::UInt32;
		
		int                   idx = 0;
		dw::Vertex* vertex_ptr = m_mesh->vertices();
		uint32_t*             index_ptr = m_mesh->indices();
		
		 for (int i = 0; i < m_mesh->vertex_count(); i++)
		    uv[i] = vertex_ptr[i].tex_coord;
		
		for (int mesh_idx = 0; mesh_idx < m_mesh->sub_mesh_count(); mesh_idx++)
		{
		    dw::SubMesh& submesh = m_mesh->sub_meshes()[mesh_idx];
		
			for (int j = submesh.base_index; j < (submesh.base_index + submesh.index_count); j++)
			    indices[idx++] = submesh.base_vertex + index_ptr[j];
		}
		
		mesh_decl.vertexUvData = uv.data();
		mesh_decl.indexData = indices.data();

		xatlas::AddMeshError::Enum error = xatlas::AddUvMesh(atlas, mesh_decl);

		if (error != xatlas::AddMeshError::Success)
		{
            xatlas::Destroy(atlas);
			DW_LOG_ERROR("Failed to add UV mesh to Lightmap Atlas");
			return false;
		}

		xatlas::PackCharts(atlas);

		return create_lightmap_uv_unwrapped_mesh(atlas);
	}

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool initialize_embree()
    {
        m_embree_device = rtcNewDevice(nullptr);

        RTCError embree_error = rtcGetDeviceError(m_embree_device);

        if (embree_error == RTC_ERROR_UNSUPPORTED_CPU)
            throw std::runtime_error("Your CPU does not meet the minimum requirements for embree");
        else if (embree_error != RTC_ERROR_NONE)
            throw std::runtime_error("Failed to initialize embree!");

        m_embree_scene = rtcNewScene(m_embree_device);

        m_embree_triangle_mesh = rtcNewGeometry(m_embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

        std::vector<glm::vec3> vertices(m_mesh->vertex_count());
        std::vector<uint32_t>  indices(m_mesh->index_count());
        uint32_t               idx        = 0;
        dw::Vertex*            vertex_ptr = m_mesh->vertices();
        uint32_t*              index_ptr  = m_mesh->indices();

        for (int i = 0; i < m_mesh->vertex_count(); i++)
            vertices[i] = vertex_ptr[i].position;

        for (int i = 0; i < m_mesh->sub_mesh_count(); i++)
        {
            dw::SubMesh& submesh = m_mesh->sub_meshes()[i];

            for (int j = submesh.base_index; j < (submesh.base_index + submesh.index_count); j++)
                indices[idx++] = submesh.base_vertex + index_ptr[j];
        }

        void* data = rtcSetNewGeometryBuffer(m_embree_triangle_mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(glm::vec3), m_mesh->vertex_count());
        memcpy(data, vertices.data(), vertices.size() * sizeof(glm::vec3));

        data = rtcSetNewGeometryBuffer(m_embree_triangle_mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), m_mesh->index_count() / 3);
        memcpy(data, indices.data(), indices.size() * sizeof(uint32_t));

        rtcCommitGeometry(m_embree_triangle_mesh);
        rtcAttachGeometry(m_embree_scene, m_embree_triangle_mesh);
        rtcCommitScene(m_embree_scene);

        rtcInitIntersectContext(&m_embree_intersect_context);

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

    void render_mesh(dw::Mesh* mesh, glm::mat4 model, std::unique_ptr<dw::Program>& program)
    {
        program->set_uniform("u_Model", model);

        // Bind vertex array.
        m_uv_unwrapped_vao->bind();

        dw::SubMesh* submeshes = mesh->sub_meshes();

        for (uint32_t i = 0; i < mesh->sub_mesh_count(); i++)
        {
            dw::SubMesh& submesh = submeshes[i];

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
        render_mesh(m_mesh, m_transform, program);
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

	void visualize_lightmap()
	{

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
    std::unique_ptr<dw::Shader> m_mesh_fs;
    std::unique_ptr<dw::Shader> m_visualize_lightmap_fs;

	std::unique_ptr<dw::Shader> m_lightmap_vs;
    std::unique_ptr<dw::Shader> m_triangle_vs;
    std::unique_ptr<dw::Shader> m_mesh_vs;

    std::unique_ptr<dw::Program> m_lightmap_program;
    std::unique_ptr<dw::Program> m_visualize_lightmap_program;
    std::unique_ptr<dw::Program> m_mesh_program;

	std::unique_ptr<dw::Texture2D> m_lightmap_texture;
    std::unique_ptr<dw::Texture2D> m_lightmap_pos_texture;
    std::unique_ptr<dw::Texture2D> m_lightmap_normal_texture;

    std::unique_ptr<dw::Framebuffer> m_lightmap_fbo;

    std::unique_ptr<dw::UniformBuffer> m_global_ubo;

	std::unique_ptr<dw::VertexBuffer> m_uv_unwrapped_vbo;
    std::unique_ptr<dw::IndexBuffer> m_uv_unwrapped_ibo;
	std::unique_ptr<dw::VertexArray>  m_uv_unwrapped_vao;

    // Camera.
    std::unique_ptr<dw::Camera> m_main_camera;

    GlobalUniforms m_global_uniforms;

    // Scene
    dw::Mesh* m_mesh;
    glm::mat4 m_transform;

    // Camera controls.
    bool  m_mouse_look         = false;
    float m_heading_speed      = 0.0f;
    float m_sideways_speed     = 0.0f;
    float m_camera_sensitivity = 0.05f;
    float m_camera_speed       = 0.02f;
    bool  m_enable_dither      = true;
    bool  m_debug_gui          = true;

    // Embree structure
    RTCDevice           m_embree_device        = nullptr;
    RTCScene            m_embree_scene         = nullptr;
    RTCGeometry         m_embree_triangle_mesh = nullptr;
    RTCIntersectContext m_embree_intersect_context;

   
    bool    m_enable_conservative_raster   = true;

    // Camera orientation.
    float m_camera_x;
    float m_camera_y;
};

DW_DECLARE_MAIN(Lightmaps)