cmake_minimum_required(VERSION 3.5.0)

set(CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -w")
# set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -w --use_fast_math")

if(POLICY CMP0146) 
    cmake_policy(SET CMP0146 OLD) 
endif()
if(NOT CUDA_INCLUDE_DIRS)
    find_package(CUDA REQUIRED)
endif()
message(STATUS "CUDA_LIBRARIES        : ${CUDA_LIBRARIES}")
message(STATUS "CUDA_cublas_LIBRARY   : ${CUDA_cublas_LIBRARY}")
message(STATUS "CUDA_cusolver_LIBRARY : ${CUDA_cusolver_LIBRARY}")
message(STATUS "CUDA_INCLUDE_DIRS     : ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_CUDART_LIBRARY   : ${CUDA_CUDART_LIBRARY}")
include_directories(${CUDA_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
message(STATUS "EIGEN3_INCLUDE_DIR : ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(CUDAPlaneLineOdometry SHARED
    calc_corn_coeff.cu
    calc_corn_coeff.cuh
    calc_surf_coeff.cu
    calc_surf_coeff.cuh
    cloud_hash_map.cu
    cloud_hash_map.cuh
    col_piv_householder_qr_v3.cuh
    compute_jac_and_res.cu
    compute_jac_and_res.cuh
    cuda_ata.cuh
    cuda_atb.cuh
    cuda_matrix.cuh
    point_associate_to_map.cu
    point_associate_to_map.cuh
    scan_to_map_optimization.cu
    scan_to_map_optimization.cuh
)
target_link_libraries(CUDAPlaneLineOdometry ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})
set_target_properties(CUDAPlaneLineOdometry PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
