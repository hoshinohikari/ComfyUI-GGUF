cmake_minimum_required(VERSION 3.25)

project(comfyui_gguf_ggml_native LANGUAGES C CXX)

foreach(required_var IN ITEMS COMFYUI_GGML_SOURCE TORCH_SDK_ROOT COMFYUI_GGML_COMMIT COMFYUI_TORCH_BUILD COMFYUI_BUILD_ID COMFYUI_NATIVE_OUTPUT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "${required_var} must be provided by the native build command")
    endif()
endforeach()

if(NOT EXISTS "${COMFYUI_GGML_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "COMFYUI_GGML_SOURCE is not a ggml source checkout")
endif()
if(NOT EXISTS "${TORCH_SDK_ROOT}/share/cmake/Torch/TorchConfig.cmake")
    message(FATAL_ERROR "TORCH_SDK_ROOT must contain the extracted torch SDK")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build pinned ggml as static libraries" FORCE)
set(GGML_STATIC OFF CACHE BOOL "Use CUDA import libraries" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "Link the pinned CUDA backend into the extension" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_CUDA_FORCE_MMQ ON CACHE BOOL "" FORCE)
set(GGML_CUDA_FORCE_CUBLAS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_FA OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)
set(COMFYUI_CUDA_ARCHS "75;86;89;120" CACHE STRING "GeForce RTX target compute capabilities")
set(CMAKE_CUDA_ARCHITECTURES "")
foreach(arch IN LISTS COMFYUI_CUDA_ARCHS)
    if(NOT arch MATCHES "^(75|86|89|120)$")
        message(FATAL_ERROR "Unsupported target ${arch}; choose 75, 86, 89 or 120")
    endif()
    # The pinned ggml uses architecture-specific Blackwell instructions.
    if(arch STREQUAL "120")
        list(APPEND CMAKE_CUDA_ARCHITECTURES "120a-real")
    else()
        list(APPEND CMAKE_CUDA_ARCHITECTURES "${arch}-real")
    endif()
endforeach()
if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "COMFYUI_CUDA_ARCHS must not be empty")
endif()
string(REPLACE ";" "," COMFYUI_CUDA_ARCHS_CSV "${COMFYUI_CUDA_ARCHS}")

add_subdirectory("${COMFYUI_GGML_SOURCE}" ggml)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

Python3_add_library(_comfyui_gguf_ggml MODULE WITH_SOABI ggml_torch.cpp)
target_compile_features(_comfyui_gguf_ggml PRIVATE cxx_std_17)
target_include_directories(_comfyui_gguf_ggml PRIVATE
    "${COMFYUI_GGML_SOURCE}/include"
    "${TORCH_SDK_ROOT}/include"
    "${TORCH_SDK_ROOT}/include/torch/csrc/api/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(_comfyui_gguf_ggml PRIVATE
    COMFYUI_GGML_COMMIT="${COMFYUI_GGML_COMMIT}"
    COMFYUI_TORCH_BUILD="${COMFYUI_TORCH_BUILD}"
    COMFYUI_BUILD_ID="${COMFYUI_BUILD_ID}"
    COMFYUI_CUDA_ARCHS="${COMFYUI_CUDA_ARCHS_CSV}"
)
target_link_directories(_comfyui_gguf_ggml PRIVATE
    "${TORCH_SDK_ROOT}/lib"
    "${CUDAToolkit_LIBRARY_DIR}"
)
target_link_libraries(_comfyui_gguf_ggml PRIVATE
    ggml-cuda
    c10
    torch
    torch_cpu
    torch_python
    c10_cuda
    torch_cuda
    CUDA::cudart
    Python3::Module
)

set_target_properties(_comfyui_gguf_ggml PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${COMFYUI_NATIVE_OUTPUT}"
    RUNTIME_OUTPUT_DIRECTORY "${COMFYUI_NATIVE_OUTPUT}"
)
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(_comfyui_gguf_ggml PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config} "${COMFYUI_NATIVE_OUTPUT}"
        RUNTIME_OUTPUT_DIRECTORY_${config} "${COMFYUI_NATIVE_OUTPUT}"
    )
endforeach()

message(STATUS "Building ComfyUI-GGUF ggml MMQ bridge for ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Pinned ggml: ${COMFYUI_GGML_COMMIT}")
message(STATUS "Torch build: ${COMFYUI_TORCH_BUILD}")
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
