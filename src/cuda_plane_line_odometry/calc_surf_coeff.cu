//
// Created by lmf on 23-9-6.
//

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "calc_surf_coeff.cuh"

#include "col_piv_householder_qr_v2.cuh"


__global__ void _CalcSurfCoeffKernel(
        unsigned int num_elements,
        const float4* d_vec_point_sel,
        char* d_vec_flag,
        const float4* d_vec_nbr_0,
        const float4* d_vec_nbr_1,
        const float4* d_vec_nbr_2,
        const float4* d_vec_nbr_3,
        const float4* d_vec_nbr_4,
        float4* d_vec_coeff
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= num_elements) {
        return;
    }

    if(d_vec_flag[i] != 1) {
        return;
    }

    float4 nbr_0 = d_vec_nbr_0[i];
    float4 nbr_1 = d_vec_nbr_1[i];
    float4 nbr_2 = d_vec_nbr_2[i];
    float4 nbr_3 = d_vec_nbr_3[i];
    float4 nbr_4 = d_vec_nbr_4[i];

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;

    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();

    matA0(0, 0) = nbr_0.x;
    matA0(0, 1) = nbr_0.y;
    matA0(0, 2) = nbr_0.z;

    matA0(1, 0) = nbr_1.x;
    matA0(1, 1) = nbr_1.y;
    matA0(1, 2) = nbr_1.z;

    matA0(2, 0) = nbr_2.x;
    matA0(2, 1) = nbr_2.y;
    matA0(2, 2) = nbr_2.z;

    matA0(3, 0) = nbr_3.x;
    matA0(3, 1) = nbr_3.y;
    matA0(3, 2) = nbr_3.z;

    matA0(4, 0) = nbr_4.x;
    matA0(4, 1) = nbr_4.y;
    matA0(4, 2) = nbr_4.z;

    // ColPivHouseholderQR5x3 qr_decomp(matA0);
    ColPivHouseholderQRNx3<5> qr_decomp(matA0);
    matX0 = qr_decomp.Solve(matB0);

    // if(i == 0) {
    //     Eigen::Matrix<float, 5, 1> matB1 = matA0 * matX0;
    //     printf(
    //         "matB1 : %.10f , %.10f , %.10f , %.10f , %.10f \n",
    //         matB1(0), matB1(1), matB1(2), matB1(3), matB1(4)
    //     );
    // }

    Eigen::Matrix<float, 5, 1> matB1 = matA0 * matX0;

    float pa = matX0(0, 0);
    float pb = matX0(1, 0);
    float pc = matX0(2, 0);
    float pd = 1;

    float ps = sqrtf(pa * pa + pb * pb + pc * pc);
    pa /= ps; pb /= ps; pc /= ps; pd /= ps;

    float4 coeff{0.0, 0.0, 0.0, 0.0};

    bool planeValid = !(    fabs(pa * nbr_0.x + pb * nbr_0.y + pc * nbr_0.z + pd) > 0.2
                         || fabs(pa * nbr_1.x + pb * nbr_1.y + pc * nbr_1.z + pd) > 0.2
                         || fabs(pa * nbr_2.x + pb * nbr_2.y + pc * nbr_2.z + pd) > 0.2
                         || fabs(pa * nbr_3.x + pb * nbr_3.y + pc * nbr_3.z + pd) > 0.2
                         || fabs(pa * nbr_4.x + pb * nbr_4.y + pc * nbr_4.z + pd) > 0.2 );
    if( !planeValid ) {
        d_vec_flag[i] = 0;
        d_vec_coeff[i] = coeff;
        return;
    }

    float4 point_sel = d_vec_point_sel[i];
    float pd2 = pa * point_sel.x + pb * point_sel.y + pc * point_sel.z + pd;
    float s = 1.0 - 0.9 * fabs(pd2) / sqrtf(sqrtf(pow(point_sel.x, 2) + pow(point_sel.y, 2) + pow(point_sel.z, 2)));

    if(s > 0.1) {
        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.w = s * pd2;
    }

    d_vec_coeff[i] = coeff;
    d_vec_flag[i] = (s > 0.1) ? 1 : 0;
}

void CalcSurfCoeffKernel::LaunchKernel(
        unsigned int num_points,
        thrust::device_vector<float4>& point_sel,
        thrust::device_vector<char>& flag,
        thrust::device_vector<float4>& nbr_0,
        thrust::device_vector<float4>& nbr_1,
        thrust::device_vector<float4>& nbr_2,
        thrust::device_vector<float4>& nbr_3,
        thrust::device_vector<float4>& nbr_4,
        thrust::device_vector<float4>& coeff
) {
    if( (num_points == 0) || coeff.empty() )
        return;

    unsigned int _num_threads_per_block = 32;
    unsigned int _num_blocks = num_points / _num_threads_per_block;
    if( (num_points % _num_threads_per_block) != 0 )
        _num_blocks += 1;

    dim3 num_blocks(_num_blocks);
    dim3 num_threads_per_block(_num_threads_per_block);

    float4* ptr_d_vec_point_sel = thrust::raw_pointer_cast(&point_sel[0]);
    char* ptr_d_vec_flag = thrust::raw_pointer_cast(&flag[0]);
    float4* ptr_d_vec_nbr_0 = thrust::raw_pointer_cast(&nbr_0[0]);
    float4* ptr_d_vec_nbr_1 = thrust::raw_pointer_cast(&nbr_1[0]);
    float4* ptr_d_vec_nbr_2 = thrust::raw_pointer_cast(&nbr_2[0]);
    float4* ptr_d_vec_nbr_3 = thrust::raw_pointer_cast(&nbr_3[0]);
    float4* ptr_d_vec_nbr_4 = thrust::raw_pointer_cast(&nbr_4[0]);
    float4* ptr_d_vec_coeff = thrust::raw_pointer_cast(&coeff[0]);
    _CalcSurfCoeffKernel<<< num_blocks , num_threads_per_block , 0 , cuda_stream >>>(
        num_points,
        ptr_d_vec_point_sel,
        ptr_d_vec_flag,
        ptr_d_vec_nbr_0,
        ptr_d_vec_nbr_1,
        ptr_d_vec_nbr_2,
        ptr_d_vec_nbr_3,
        ptr_d_vec_nbr_4,
        ptr_d_vec_coeff
    );
}


