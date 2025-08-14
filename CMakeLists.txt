cmake_minimum_required(VERSION 3.10)

Project(XXSLibrary)

if(APPLE)
    set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
endif()

file(GLOB_RECURSE src_files ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE inc_files ${PROJECT_SOURCE_DIR}/inc/*.h ${PROJECT_SOURCE_DIR}/rc/*.h)
set(CMAKE_CXX_STANDARD 23)

add_executable(${CMAKE_PROJECT_NAME} ${src_files} ${inc_files})

if(WIN32)
    file(GLOB_RECURSE rc_files ${PROJECT_SOURCE_DIR}/rc/*.rc)
    set(CMAKE_RC_FLAGS "${CMAKE_RC_FLAGS} -O coff --codepage=65001 -D UNICODE -D _UNICODE")
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${rc_files})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ole32 winmm msimg32 gdi32 user32 psapi)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE -DUNICODE -D_UNICODE)
endif()

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/inc/
)

find_package(Threads REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Threads::Threads)
