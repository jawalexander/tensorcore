cmake_minimum_required(VERSION 3.10)

project(TensorCoreGEMMV2 LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 100a)


# # # CUDA
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
find_package(CUDAToolkit REQUIRED)  # 替代旧的find_package(CUDA)

# # # cuBLAS
# find_package(CUBLAS REQUIRED)
# include_directories(${CUBLAS_INCLUDE_DIRS})
# target_link_libraries(TensorCoreGEMMV2 ${CUBLAS_LIBRARIES})

# # # # cuSPARSE
# find_package(CUSPARSE REQUIRED)
# include_directories(${CUSPARSE_INCLUDE_DIRS})
# target_link_libraries(TensorCoreGEMMV2 ${CUSPARSE_LIBRARIES})
# # OpenMP
# find_package(OpenMP REQUIRED)
# include_directories(${OpenMP_INCLUDE_DIRS})

# # CUB
# set(CUB_DIR "/usr/local/cuda-12.9/targets/x86_64-linux/lib/cmake/cub")  # 或手动下载的路径[1](@ref)
# find_package(CUB REQUIRED)
# include_directories(${CUB_INCLUDE_DIRS})
# link_directories(/usr/local/cuda-12.9/lib64)

include_directories(common)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUBLAS_API_H=1")
add_executable(TensorCoreGEMMV2 hgemm_v2.cu)
add_executable(TensorCoreGEMMV1 hgemm_v1.cu)
target_link_libraries(TensorCoreGEMMV2 PRIVATE CUDA::cublas )
target_link_libraries(TensorCoreGEMMV1 PRIVATE CUDA::cublas )
