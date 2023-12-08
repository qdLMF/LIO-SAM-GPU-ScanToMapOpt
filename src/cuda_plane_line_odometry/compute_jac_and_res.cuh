//
// Created by lmf on 23-9-6.
//

#ifndef LIO_SAM_CUDA_COMPUTE_JAC_AND_RHS_CUH
#define LIO_SAM_CUDA_COMPUTE_JAC_AND_RHS_CUH

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda_matrix.cuh"


struct ComputeJacAndResKernel {
public :
    ComputeJacAndResKernel() { cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking); }
    ~ComputeJacAndResKernel() { Sync(); cudaStreamDestroy(cuda_stream); }

    void LaunchKernel(
        unsigned int num_points,
        thrust::device_vector<float4>& point_ori,
        thrust::device_vector<char>& flag,
        thrust::device_vector<float4>& coeff,
        float trans_0,
        float trans_1,
        float trans_2,
        CUDAMatrix<float>& A,
        CUDAMatrix<float>& B
    );

    void Sync() const { cudaStreamSynchronize(cuda_stream); }

public :
    float srx;
    float crx;
    float sry;
    float cry;
    float srz;
    float crz;

public :
    cudaStream_t cuda_stream{};
};

#endif //LIO_SAM_CUDA_COMPUTE_JAC_AND_RHS_CUH
