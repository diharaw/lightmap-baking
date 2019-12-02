cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_program(CLANG_FORMAT_EXE NAMES "clang-format" DOC "Path to clang-format executable")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(PRECOMPUTEDGI_SOURCES ${PROJECT_SOURCE_DIR}/src/main.cpp 
                          ${PROJECT_SOURCE_DIR}/src/skybox.h 
                          ${PROJECT_SOURCE_DIR}/src/skybox.cpp)

set(XATLAS_SOURCES ${PROJECT_SOURCE_DIR}/external/xatlas/xatlas.cpp
                   ${PROJECT_SOURCE_DIR}/external/xatlas/xatlas.h)

set(HOSEKSKY_SOURCES ${PROJECT_SOURCE_DIR}/external/HosekSky/ArHosekSkyModel.cpp)

file(GLOB_RECURSE SHADER_SOURCES ${PROJECT_SOURCE_DIR}/src/*.glsl)

if(APPLE)
    add_executable(PrecomputedGI MACOSX_BUNDLE ${PRECOMPUTEDGI_SOURCES} ${XATLAS_SOURCES} ${HOSEKSKY_SOURCES} ${SHADER_SOURCES} ${ASSET_SOURCES})
    set(MACOSX_BUNDLE_BUNDLE_NAME "PrecomputedGI") 
    set_source_files_properties(${SHADER_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/shader)
    set_source_files_properties(${ASSET_SOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
    add_executable(PrecomputedGI ${PRECOMPUTEDGI_SOURCES} ${XATLAS_SOURCES} ${HOSEKSKY_SOURCES}) 
endif()

target_link_libraries(PrecomputedGI dwSampleFramework)
target_link_libraries(PrecomputedGI embree)

if (NOT APPLE)
    add_custom_command(TARGET PrecomputedGI POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:PrecomputedGI>/shader)
endif()

if(CLANG_FORMAT_EXE)
    add_custom_target(PrecomputedGI-clang-format COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${PRECOMPUTEDGI_SOURCES} ${SHADER_SOURCES})
endif()

set_property(TARGET PrecomputedGI PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$(Configuration)")