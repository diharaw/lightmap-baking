#pragma once

#include <glm.hpp>
#include <camera.h>
#include <ogl.h>
#include <memory>

#define MAX_FRUSTUM_SPLITS 8

struct FrustumSplit
{
    float     near_plane;
    float     far_plane;
    float     ratio;
    float     fov;
    glm::vec3 center;
    glm::vec3 corners[8];
};

struct CSM
{
    dw::Texture2D*   m_shadow_maps = nullptr;
    dw::Framebuffer* m_shadow_fbos[MAX_FRUSTUM_SPLITS];
    float            m_lambda;
    float            m_near_offset;
    int              m_split_count;
    int              m_shadow_map_size;
    FrustumSplit     m_splits[MAX_FRUSTUM_SPLITS];
    float            m_far_bounds[MAX_FRUSTUM_SPLITS];
    glm::vec3        m_light_direction;
    glm::mat4        m_bias;
    glm::mat4        m_light_view;
    glm::mat4        m_crop_matrices[MAX_FRUSTUM_SPLITS]; // crop * proj * view
    glm::mat4        m_proj_matrices[MAX_FRUSTUM_SPLITS]; // crop * proj * light_view * inv_view
    glm::mat4        m_texture_matrices[MAX_FRUSTUM_SPLITS];
    bool             m_stable_pssm = false;

    CSM();
    ~CSM();
    void initialize(float lambda, float near_offset, int split_count, int shadow_map_size, dw::Camera* camera, int _width, int _height, glm::vec3 dir);
    void shutdown();
    void update(dw::Camera* camera, glm::vec3 dir);
    void update_splits(dw::Camera* camera);
    void update_frustum_corners(dw::Camera* camera);
    void update_crop_matrices(glm::mat4 t_modelview, dw::Camera* camera);
    void update_texture_matrices(dw::Camera* camera);
    void update_far_bounds(dw::Camera* camera);

    inline FrustumSplit*     frustum_splits() { return &m_splits[0]; }
    inline glm::mat4         split_view_proj(int i) { return m_crop_matrices[i]; }
    inline glm::mat4         texture_matrix(int i) { return m_texture_matrices[i]; }
    inline float             far_bound(int i) { return m_far_bounds[i]; }
    inline dw::Texture2D*    shadow_map() { return m_shadow_maps; }
    inline dw::Framebuffer** framebuffers() { return &m_shadow_fbos[0]; }
    inline uint32_t          frustum_split_count() { return m_split_count; }
    inline uint32_t          near_offset() { return m_near_offset; }
    inline uint32_t          lambda() { return m_lambda; }
    inline uint32_t          shadow_map_size() { return m_shadow_map_size; }
};
