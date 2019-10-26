cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_program(CLANG_FORMAT_EXE NAMES "clang-format" DOC "Path to clang-format executable")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(LIGHTMAPS_SOURCES ${PROJECT_SOURCE_DIR}/src/main.cpp)

set(XATLAS_SOURCES ${PROJECT_SOURCE_DIR}/external/xatlas/xatlas.cpp
                   ${PROJECT_SOURCE_DIR}/external/xatlas/xatlas.h)

file(GLOB_RECURSE SHADER_SOURCES ${PROJECT_SOURCE_DIR}/src/*.glsl)

if(APPLE)
    add_executable(Lightmaps MACOSX_BUNDLE ${LIGHTMAPS_SOURCES} ${XATLAS_SOURCES} ${SHADER_SOURCES} ${ASSET_SOURCES})
    set(MACOSX_BUNDLE_BUNDLE_NAME "Lightmaps") 
    set_source_files_properties(${SHADER_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/shader)
    set_source_files_properties(${ASSET_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
    add_executable(Lightmaps ${LIGHTMAPS_SOURCES} ${XATLAS_SOURCES}) 
endif()

target_link_libraries(Lightmaps dwSampleFramework)
target_link_libraries(Lightmaps embree)

if (NOT APPLE)
    add_custom_command(TARGET Lightmaps POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:Lightmaps>/shader)
    add_custom_command(TARGET Lightmaps POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/mesh $<TARGET_FILE_DIR:Lightmaps>/mesh)
    add_custom_command(TARGET Lightmaps POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/texture $<TARGET_FILE_DIR:Lightmaps>/texture)
endif()

if(CLANG_FORMAT_EXE)
    add_custom_target(Lightmaps-clang-format COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${LIGHTMAPS_SOURCES} ${SHADER_SOURCES})
endif()

set_property(TARGET Lightmaps PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$(Configuration)")