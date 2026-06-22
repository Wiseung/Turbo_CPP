cmake_minimum_required(VERSION 3.27)

project(Turbo_CPP LANGUAGES CXX)

option(TURBO_CPP_ENABLE_CUDA "Build CUDA turbo decoder kernels" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(TURBO_CPP_ENABLE_CUDA)
  enable_language(CUDA OPTIONAL)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    set(TURBO_CPP_HAS_CUDA ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  else()
    set(TURBO_CPP_HAS_CUDA OFF)
    message(WARNING "CUDA requested but no CUDA compiler is available; building CPU-only.")
  endif()
else()
  set(TURBO_CPP_HAS_CUDA OFF)
endif()

add_library(turbo_cpp
  src/lte_turbo.cpp
)

target_include_directories(turbo_cpp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(turbo_cpp PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(turbo_cpp PRIVATE /W4 /permissive-)
else()
  target_compile_options(turbo_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(turbo_cpp
  PUBLIC
    TURBO_CPP_HAS_CUDA=$<BOOL:${TURBO_CPP_HAS_CUDA}>
)

if(TURBO_CPP_HAS_CUDA)
  add_library(turbo_cpp_cuda STATIC src/lte_turbo_cuda.cu)
  target_include_directories(turbo_cpp_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(turbo_cpp_cuda PUBLIC TURBO_CPP_HAS_CUDA=1)
  set_target_properties(turbo_cpp_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(turbo_cpp PUBLIC turbo_cpp_cuda CUDA::cudart)
endif()

add_executable(turbo_cpp_tests tests/test_main.cpp)
target_link_libraries(turbo_cpp_tests PRIVATE turbo_cpp)

add_executable(turbo_cpp_bench tests/bench_main.cpp)
target_link_libraries(turbo_cpp_bench PRIVATE turbo_cpp)

if(TURBO_CPP_HAS_CUDA)
  set_target_properties(turbo_cpp_tests PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(turbo_cpp_bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(turbo_cpp_tests PRIVATE CUDA::cudart)
  target_link_libraries(turbo_cpp_bench PRIVATE CUDA::cudart)
endif()

enable_testing()
add_test(NAME turbo_cpp_tests COMMAND turbo_cpp_tests)
