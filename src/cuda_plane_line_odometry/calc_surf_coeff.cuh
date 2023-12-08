//
// Created by lmf on 23-9-6.
//

#ifndef LIO_SAM_CUDA_CALC_SURF_COEFF_CUH
#define LIO_SAM_CUDA_CALC_SURF_COEFF_CUH

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct CalcSurfCoeffKernel {
public :
    CalcSurfCoeffKernel() { cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking); }
    ~CalcSurfCoeffKernel() { Sync(); cudaStreamDestroy(cuda_stream); }

    void LaunchKernel(
        unsigned int num_points,
        thrust::device_vector<float4>& point_sel,
        thrust::device_vector<char>& flag,
        thrust::device_vector<float4>& nbr_0,
        thrust::device_vector<float4>& nbr_1,
        thrust::device_vector<float4>& nbr_2,
        thrust::device_vector<float4>& nbr_3,
        thrust::device_vector<float4>& nbr_4,
        thrust::device_vector<float4>& coeff
    );

    void Sync() const { cudaStreamSynchronize(cuda_stream); }

public :
    cudaStream_t cuda_stream{};
};

#endif //LIO_SAM_CUDA_CALC_SURF_COEFF_CUH
