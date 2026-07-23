cmake_minimum_required(VERSION 3.16)
project(metalsw CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(METALSW_USE_PARASAIL "Cross-check oracle scores against Parasail" OFF)

add_executable(metalsw_baseline
    src/baseline_main.cpp
    src/fasta.cpp
    src/sw_reference.cpp
)
target_include_directories(metalsw_baseline PRIVATE include)

if(METALSW_USE_PARASAIL)
    find_library(PARASAIL_LIB parasail REQUIRED)
    find_path(PARASAIL_INCLUDE_DIR parasail.h REQUIRED)
    target_compile_definitions(metalsw_baseline PRIVATE METALSW_USE_PARASAIL)
    target_include_directories(metalsw_baseline PRIVATE ${PARASAIL_INCLUDE_DIR})
    target_link_libraries(metalsw_baseline PRIVATE ${PARASAIL_LIB})
endif()

if(APPLE)
    include(FetchContent)
    FetchContent_Declare(
        metal-cpp
        GIT_REPOSITORY https://github.com/bkaradzic/metal-cpp.git
        GIT_TAG c9727bc9468a90d7ea8fc89d5ee03b8d8992a570
    )
    FetchContent_MakeAvailable(metal-cpp)

    set(METAL_SHADER_SRC ${CMAKE_SOURCE_DIR}/metal/smith_waterman.metal)
    set(METAL_AIR ${CMAKE_BINARY_DIR}/smith_waterman.air)
    set(METAL_LIB ${CMAKE_BINARY_DIR}/smith_waterman.metallib)

    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SRC} -o ${METAL_AIR}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_SHADER_SRC}
        COMMENT "Compiling Metal shader smith_waterman.metal"
    )
    add_custom_target(metalsw_shaders DEPENDS ${METAL_LIB})

    add_executable(metalsw_gpu
        src/gpu_main.cpp
        src/metal_runtime.cpp
        src/fasta.cpp
        src/sw_reference.cpp
    )
    target_include_directories(metalsw_gpu PRIVATE include ${metal-cpp_SOURCE_DIR})
    target_link_libraries(metalsw_gpu PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
    )
    add_dependencies(metalsw_gpu metalsw_shaders)

    # Copy the compiled metallib next to the binary so the default relative
    # path in gpu_main.cpp resolves without extra flags.
    add_custom_command(TARGET metalsw_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${METAL_LIB} $<TARGET_FILE_DIR:metalsw_gpu>/smith_waterman.metallib
    )
endif()
