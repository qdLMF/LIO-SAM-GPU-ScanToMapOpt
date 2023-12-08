//
// Created by lmf on 23-9-5.
//

#include "point_associate_to_map.cuh"

__global__ void _PointAssociateToMapKernel(
        unsigned int num_elements,
        const Affine3x4 trans3x4,       // don't use eigen object, it might cause sync error when copying eigen object from host to device as a param of a kernel function!
        const float4* d_vec_point_ori,
        float4* d_vec_point_sel
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= num_elements)
        return;

    float4 point_ori = d_vec_point_ori[i];
    float ori_x = point_ori.x;
    float ori_y = point_ori.y;
    float ori_z = point_ori.z;
    float ori_i = point_ori.w;

    float4 point_sel;
    point_sel.x = trans3x4.elem[0][0] * ori_x + trans3x4.elem[0][1] * ori_y + trans3x4.elem[0][2] * ori_z + trans3x4.elem[0][3];
    point_sel.y = trans3x4.elem[1][0] * ori_x + trans3x4.elem[1][1] * ori_y + trans3x4.elem[1][2] * ori_z + trans3x4.elem[1][3];
    point_sel.z = trans3x4.elem[2][0] * ori_x + trans3x4.elem[2][1] * ori_y + trans3x4.elem[2][2] * ori_z + trans3x4.elem[2][3];
    point_sel.w = ori_i;

    d_vec_point_sel[i] = point_sel;
}

void PointAssociateToMapKernel::LaunchKernel(
    unsigned int num_points,
    thrust::device_vector<float4>& point_ori,
    thrust::device_vector<float4>& point_sel
) {
    if( (num_points == 0) || point_ori.empty() )
        return;

    unsigned int _num_threads_per_block = 32;
    unsigned int _num_blocks = num_points / _num_threads_per_block;
    if( (num_points % _num_threads_per_block) != 0 )
        _num_blocks += 1;

    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads_per_block);

    float4* ptr_d_vec_point_ori = thrust::raw_pointer_cast(&point_ori[0]);
    float4* ptr_d_vec_point_sel = thrust::raw_pointer_cast(&point_sel[0]);
    _PointAssociateToMapKernel<<< num_blocks , num_threads_per_block , 0 , cuda_stream >>>(
        num_points,
        trans3x4,
        ptr_d_vec_point_ori,
        ptr_d_vec_point_sel
    );
}



