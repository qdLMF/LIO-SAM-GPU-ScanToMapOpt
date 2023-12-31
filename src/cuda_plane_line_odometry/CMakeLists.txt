cmake_minimum_required(VERSION 2.8.3)

set(CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -w") # "-w" means "ignore all warnings"

find_package(CUDA REQUIRED)
message(STATUS "CUDA_LIBRARIES : ${CUDA_LIBRARIES}")
message(STATUS "CUDA_cublas_LIBRARY : ${CUDA_cublas_LIBRARY}")
message(STATUS "CUDA_cusolver_LIBRARY : ${CUDA_cusolver_LIBRARY}")
message(STATUS "CUDA_INCLUDE_DIRS : ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
message(STATUS "EIGEN3_INCLUDE_DIR : ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(CUDALinePlaneOdometry SHARED
    point_associate_to_map.cu
    point_associate_to_map.cuh
    col_piv_householder_qr_v1.cuh
    col_piv_householder_qr_v2.cuh
    calc_surf_coeff.cu
    calc_surf_coeff.cuh
    cuda_matrix.cuh
    compute_jac_and_res.cu
    compute_jac_and_res.cuh
    cuda_ata.cuh
    cuda_atb.cuh
    scan_to_map_optimization.cu
    scan_to_map_optimization.cuh
    calc_corner_coeff.cu
    calc_corner_coeff.cuh
    cloud_hash_map.cu
    cloud_hash_map.cuh
)
target_link_libraries(CUDALinePlaneOdometry ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})
set_target_properties(CUDALinePlaneOdometry PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
