cmake_minimum_required(VERSION 3.17)

project(XXSLibrary LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the executable target first
add_executable(${CMAKE_PROJECT_NAME} "")

# Include directories
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/inc
)

# --- Source Files ---
file(GLOB_RECURSE src_files
    "src/*.cpp"
    "src/gpu/*.cpp"
    "src/gpu/*.mm"
)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${src_files})


# --- Platform-Specific Settings ---

# Apple (macOS)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS version")
    find_library(METAL_LIBRARY Metal)
    find_library(FOUNDATION_LIBRARY Foundation)
    if(METAL_LIBRARY AND FOUNDATION_LIBRARY)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${METAL_LIBRARY}" "${FOUNDATION_LIBRARY}")
        message(STATUS "Metal framework found and linked.")
    else()
        message(WARNING "Metal framework not found.")
    endif()
endif()

# CUDA (for NVIDIA GPUs, primarily on Linux/Windows)
find_package(CUDA)
if(CUDA_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cudart)
    message(STATUS "CUDA toolkit found and linked.")
else()
    message(STATUS "CUDA toolkit not found. CUDA backend will be unavailable.")
endif()

# OpenCL (Cross-platform)
find_package(OpenCL)
if(OpenCL_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenCL::OpenCL)
    message(STATUS "OpenCL found and linked.")
else()
    message(WARNING "OpenCL not found. OpenCL backend will be unavailable.")
endif()

# Windows-specific settings
if(WIN32)
    file(GLOB_RECURSE rc_files ${PROJECT_SOURCE_DIR}/rc/*.rc)
    set(CMAKE_RC_FLAGS "${CMAKE_RC_FLAGS} -O coff --codepage=65001 -D UNICODE -D _UNICODE")
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${rc_files})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ole32 winmm msimg32 gdi32 user32 psapi)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE -DUNICODE -D_UNICODE)
endif()

# Threads (required by the original project)
find_package(Threads REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Threads::Threads)

# Copy the OpenCL kernel file to the executable's directory so it can be found at runtime.
add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${PROJECT_SOURCE_DIR}/src/gpu/entropy_kernel.cl"
        "$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>"
    COMMENT "Copying OpenCL kernel to build directory")
