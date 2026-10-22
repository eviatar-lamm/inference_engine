
cmake_minimum_required(VERSION 3.18)
project(llm_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(inference
    src/main.cpp
    src/safetensor.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inference PRIVATE
        -O3
        -march=native
    )
endif()

# Link-time optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported)

if(lto_supported)
    set_property(TARGET inference PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()


target_include_directories(inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)