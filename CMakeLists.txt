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
    # GPU target (Stage 1, v0.2) — Metal kernel + metal-cpp host, not yet implemented.
    # add_executable(metalsw_gpu src/gpu_main.cpp src/metal_runtime.cpp ...)
endif()
