//
// Created by lmf on 23-9-5.
//

#ifndef LIO_SAM_CUDA_POINT_ASSOCIATE_TO_MAP_CUH
#define LIO_SAM_CUDA_POINT_ASSOCIATE_TO_MAP_CUH

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Affine3x4 {
    float elem[3][4];
};

struct PointAssociateToMapKernel {
public :
    PointAssociateToMapKernel() { cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking); }
    ~PointAssociateToMapKernel() { Sync(); cudaStreamDestroy(cuda_stream); }

    // PointAssociateToMapKernel() = default;

    void LaunchKernel(
        unsigned int num_points,
        thrust::device_vector<float4>& point_ori,
        thrust::device_vector<float4>& point_sel
    );

    void Sync() const { cudaStreamSynchronize(cuda_stream); }

public : 
    // don't use eigen object, 
    // it might cause sync error when copying eigen object from host to device as a param of a kernel function!
    Affine3x4 trans3x4;

public :
    cudaStream_t cuda_stream{};

};

#endif //LIO_SAM_CUDA_POINT_ASSOCIATE_TO_MAP_CUH
