# Portable C++20 project. CMakePresets.json selects explicit MinGW/Ninja paths
# and separate Debug/Release build directories for the tested Windows setup.
cmake_minimum_required(VERSION 3.24)
project(CPUrenderer VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

# Reusable library: geometry, scattering, transport, and isolated file adapters.
# Consumers see our headers, while vendored parser/codec headers remain private.
add_library(renderer STATIC src/geometry.cpp src/bsdf.cpp src/render.cpp src/io/gltf.cpp src/io/image.cpp)
target_include_directories(renderer PUBLIC src)
target_include_directories(renderer SYSTEM PRIVATE third_party)
target_link_libraries(renderer PUBLIC Threads::Threads)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Diagnose project mistakes without making third-party warnings fatal.
    target_compile_options(renderer PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()
if(MINGW)
    # Propagate static runtime linking to executables so they launch without
    # MinGW's DLL directory on PATH. Windows system DLLs remain OS dependencies.
    target_link_options(renderer INTERFACE -static -static-libgcc -static-libstdc++)
endif()

# The CLI owns argument parsing and output orchestration, not scene construction.
add_executable(cpu_renderer src/main.cpp)
target_link_libraries(cpu_renderer PRIVATE renderer)
if(WIN32)
    # Optional peak-working-set reporting is isolated to the Windows front end.
    target_link_libraries(cpu_renderer PRIVATE psapi)
endif()
include(CTest)
if(BUILD_TESTING)
    # Manual diagnostics are built with tests but need explicit input file arguments.
    add_executable(renderer_benchmark tests/benchmark.cpp)
    target_link_libraries(renderer_benchmark PRIVATE renderer)
    add_executable(image_metrics tests/image_metrics.cpp)
    target_link_libraries(image_metrics PRIVATE renderer)
    foreach(suite geometry io integrator transport_edges)
        # Standalone exception-based suites work in both Debug and Release.
        # The source path locates bundled fixtures independently of the current cwd.
        add_executable(${suite}_tests tests/test_${suite}.cpp)
        target_link_libraries(${suite}_tests PRIVATE renderer)
        target_compile_definitions(${suite}_tests PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(NAME ${suite} COMMAND ${suite}_tests)
    endforeach()
    add_test(NAME cli_help COMMAND cpu_renderer --help)
    add_test(NAME cli_invalid COMMAND cpu_renderer --samples 0)
    set_tests_properties(cli_invalid PROPERTIES WILL_FAIL TRUE)
    # Exercise the actual executable, then verify pinned source/asset integrity.
    add_test(NAME cli_checks COMMAND ${CMAKE_COMMAND}
        -DRENDERER=$<TARGET_FILE:cpu_renderer> -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/cli-tests -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_checks.cmake)
    add_test(NAME assets COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_assets.cmake)
endif()
