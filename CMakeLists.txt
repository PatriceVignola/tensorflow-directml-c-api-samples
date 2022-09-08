cmake_minimum_required(VERSION 3.20)

project(tensorflow-directml-c-api-samples VERSION 0.0.1 LANGUAGES CXX)

# Statically link runtime library to avoid runtime dependency on Visual C++ redistributable.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

# The C++ standard needs to be set globally to ensure Abseil targets are compiled
# with the same standard as the dependent targets.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PIC needs to be enabled globally for Abseil targets.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Import dependencies and their targets.
include(cmake/dependencies.cmake)

add_executable(
    infer_squeezenet_model
    src/infer_squeezenet_model.cc
)
target_link_libraries(
    infer_squeezenet_model
    PRIVATE
    tensorflow_libs
    ${CMAKE_DL_LIBS}
)
target_include_directories(
    infer_squeezenet_model
    PRIVATE
    ${tensorflow_SOURCE_DIR}/include
    ${abseil_SOURCE_DIR}
)
set_target_properties(
    infer_squeezenet_model
    PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$\{ORIGIN\}"
)

set(pkg_version "${CMAKE_PROJECT_VERSION}")
if(WIN32)
    set(pkg_platform "win_amd64")
else()
    set(pkg_platform "linux_x86_64")
endif()

string(REPLACE "-" "_" pkg_name "${CMAKE_PROJECT_NAME}")
set(DIRECTML_SHA "0de2b4431c6572ee74152a7ee0cd3fb1534e4a95")
set(c_api_samples_full_name "${pkg_name}-${pkg_version}-${pkg_platform}-c-api-samples.zip")
add_custom_command(
    OUTPUT 
        ${c_api_samples_full_name}
    COMMAND
        ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:infer_squeezenet_model>
        $<$<BOOL:${UNIX}>:${tensorflow_SOURCE_DIR}/lib/libtensorflow.so.1>
        $<$<BOOL:${UNIX}>:${tensorflow_SOURCE_DIR}/lib/libtensorflow_framework.so.1>
        $<$<BOOL:${UNIX}>:${tensorflow_SOURCE_DIR}/lib/libdirectml.${DIRECTML_SHA}.so>
        $<$<BOOL:${WIN32}>:${tensorflow_SOURCE_DIR}/lib/tensorflow.dll>
        $<$<BOOL:${WIN32}>:${tensorflow_SOURCE_DIR}/lib/tensorflow_framework.dll>
        $<$<BOOL:${WIN32}>:${tensorflow_SOURCE_DIR}/lib/DirectML.${DIRECTML_SHA}.dll>
        .
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${squeezenet_model_SOURCE_DIR} ./squeezenet_model
    COMMAND
        ${CMAKE_COMMAND} -E tar "cfv" "${c_api_samples_full_name}" --format=zip --
        $<TARGET_FILE_NAME:infer_squeezenet_model>
        $<$<BOOL:${UNIX}>:libtensorflow.so.1>
        $<$<BOOL:${UNIX}>:libtensorflow_framework.so.1>
        $<$<BOOL:${UNIX}>:libdirectml.${DIRECTML_SHA}.so>
        $<$<BOOL:${WIN32}>:tensorflow.dll>
        $<$<BOOL:${WIN32}>:tensorflow_framework.dll>
        $<$<BOOL:${WIN32}>:DirectML.${DIRECTML_SHA}.dll>
        squeezenet_model/squeezenet.pb
    DEPENDS
        infer_squeezenet_model
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT
        "Creating ${c_api_samples_full_name}"
)
add_custom_target(c_api_samples ALL DEPENDS ${c_api_samples_full_name})
