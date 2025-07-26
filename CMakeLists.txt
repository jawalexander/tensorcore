cmake_minimum_required(VERSION 3.10)

project(TensorCoreGEMMV2 LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_ARCHITECTURES "100a;110;120;130;140;150;160;170;80;90")
set(CMAKE_CUDA_ARCHITECTURES "120")


# # # CUDA
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
find_package(CUDAToolkit REQUIRED)  # 替代旧的find_package(CUDA)

# SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
# SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")

include_directories(common)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUBLAS_API_H=1")

function(add_tensorcore_gemm version)
    set(target_name "TensorCoreGEMMV${version}")
    set(source_file "hgemm_v${version}.cu")
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE CUDA::cublas)
endfunction()

add_tensorcore_gemm(1)
add_tensorcore_gemm(2)
add_tensorcore_gemm(3)
add_tensorcore_gemm(4)
add_tensorcore_gemm(5)
add_tensorcore_gemm(6)