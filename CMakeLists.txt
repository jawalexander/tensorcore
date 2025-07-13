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

add_executable(TensorCoreGEMMV1 hgemm_v1.cu)
add_executable(TensorCoreGEMMV2 hgemm_v2.cu)
add_executable(TensorCoreGEMMV3 hgemm_v3.cu)
add_executable(TensorCoreGEMMV4 hgemm_v4.cu)
target_link_libraries(TensorCoreGEMMV1 PRIVATE CUDA::cublas)
target_link_libraries(TensorCoreGEMMV2 PRIVATE CUDA::cublas)
target_link_libraries(TensorCoreGEMMV3 PRIVATE CUDA::cublas)
target_link_libraries(TensorCoreGEMMV4 PRIVATE CUDA::cublas)


# target_compile_options(TensorCoreGEMMV2 PRIVATE --ptxas-options=-v)  