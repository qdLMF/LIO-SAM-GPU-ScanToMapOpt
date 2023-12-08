//
// Created by lmf on 23-9-6.
//

#include "compute_jac_and_res.cuh"


__global__ void _ComputeJacobianAndResidualKernel(
    unsigned int num_elements,
    const float srx,
    const float crx,
    const float sry,
    const float cry,
    const float srz,
    const float crz,
    const float4* ptr_d_vec_point_ori,
    const char* flag,
    const float4* ptr_d_vec_coeff,
    float* ptr_d_mat_jacobian,
    unsigned int leading_dim_jacobian,
    float* ptr_d_mat_res,
    unsigned int leading_dim_res
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= num_elements) {
        return;
    }

    if(flag[i] == 0) {
        return;
    }

    // lidar -> camera
    float4 point_ori_tmp = ptr_d_vec_point_ori[i];
    float4 point_ori{point_ori_tmp.y, point_ori_tmp.z, point_ori_tmp.x, point_ori_tmp.w};

    // lidar -> camera
    float4 coeff_tmp = ptr_d_vec_coeff[i];
    float4 coeff{coeff_tmp.y, coeff_tmp.z, coeff_tmp.x, coeff_tmp.w};

    // in camera
    float arx = (   crx * sry * srz * point_ori.x + crx * crz * sry * point_ori.y - srx * sry * point_ori.z ) * coeff.x
              + ( - srx * srz * point_ori.x       - crz * srx * point_ori.y       - crx * point_ori.z       ) * coeff.y
              + (   crx * cry * srz * point_ori.x + crx * cry * crz * point_ori.y - cry * srx * point_ori.z ) * coeff.z;

    float ary = ( (   cry * srx * srz - crz * sry ) * point_ori.x + ( sry * srz + cry * crz * srx ) * point_ori.y + crx * cry * point_ori.z ) * coeff.x
              + ( ( - cry * crz - srx * sry * srz ) * point_ori.x + ( cry * srz - crz * srx * sry ) * point_ori.y - crx * sry * point_ori.z ) * coeff.z;

    float arz = ( ( crz * srx * sry - cry * srz ) * point_ori.x + ( - cry * crz - srx * sry * srz ) * point_ori.y ) * coeff.x
              + ( ( crx * crz) * point_ori.x - ( crx * srz ) * point_ori.y ) * coeff.y
              + ( ( sry * srz + cry * crz * srx ) * point_ori.x + (   crz * sry - cry * srx * srz ) * point_ori.y ) * coeff.z;

    *(ptr_d_mat_jacobian + 0 * leading_dim_jacobian + i) = arz;
    *(ptr_d_mat_jacobian + 1 * leading_dim_jacobian + i) = arx;
    *(ptr_d_mat_jacobian + 2 * leading_dim_jacobian + i) = ary;
    *(ptr_d_mat_jacobian + 3 * leading_dim_jacobian + i) = coeff.z;
    *(ptr_d_mat_jacobian + 4 * leading_dim_jacobian + i) = coeff.x;
    *(ptr_d_mat_jacobian + 5 * leading_dim_jacobian + i) = coeff.y;
    *(ptr_d_mat_res + 0 * leading_dim_res + i) = -coeff.w;
}

void ComputeJacAndResKernel::LaunchKernel(
    unsigned int num_points,
    thrust::device_vector<float4>& point_ori,
    thrust::device_vector<char>& flag,
    thrust::device_vector<float4>& coeff,
    float trans_0,
    float trans_1,
    float trans_2,
    CUDAMatrix<float>& A,
    CUDAMatrix<float>& B
) {
    assert(A.cols() == 6);
    assert(A.rows() <= A.ld());
    assert(B.cols() == 1);
    assert(A.rows() <= A.ld());
    assert(B.rows() <= B.ld());
    assert(A.rows() == B.rows());
    assert(coeff.size() == A.rows());

    srx = sin(trans_1);
    crx = cos(trans_1);
    sry = sin(trans_2);
    cry = cos(trans_2);
    srz = sin(trans_0);
    crz = cos(trans_0);

    if( (num_points == 0) || coeff.empty() )
        return;

    unsigned int _num_threads_per_block = 32;
    unsigned int _num_blocks = num_points / _num_threads_per_block;
    if( (num_points % _num_threads_per_block) != 0 )
        _num_blocks += 1;

    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads_per_block);

    char* ptr_d_vec_flag = thrust::raw_pointer_cast(&flag[0]);
    float4* ptr_d_vec_point_ori = thrust::raw_pointer_cast(&point_ori[0]);
    float4* ptr_d_vec_coeff = thrust::raw_pointer_cast(&coeff[0]);
    _ComputeJacobianAndResidualKernel<<< num_blocks , num_threads_per_block , 0 >>>(
        num_points,
        srx,
        crx,
        sry,
        cry,
        srz,
        crz,
        ptr_d_vec_point_ori,
        ptr_d_vec_flag,
        ptr_d_vec_coeff,
        A.GetDevicePtr(),
        A.ld(),
        B.GetDevicePtr(),
        B.ld()
    );
}
