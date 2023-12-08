//
// Created by lmf on 23-9-7.
//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "calc_corner_coeff.cuh"

__global__ void _CalcCornerCoeffKernel(
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

    float cx = 0.0, cy = 0.0, cz = 0.0;
    cx += nbr_0.x; cy += nbr_0.y; cz += nbr_0.z;
    cx += nbr_1.x; cy += nbr_1.y; cz += nbr_1.z;
    cx += nbr_2.x; cy += nbr_2.y; cz += nbr_2.z;
    cx += nbr_3.x; cy += nbr_3.y; cz += nbr_3.z;
    cx += nbr_4.x; cy += nbr_4.y; cz += nbr_4.z;
    cx /= 5; cy /= 5; cz /= 5;

    float ax = 0.0, ay = 0.0, az = 0.0;
    float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
    ax = nbr_0.x - cx; ay = nbr_0.y - cy; az = nbr_0.z - cz;
    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
    a22 += ay * ay; a23 += ay * az;
    a33 += az * az;
    ax = nbr_1.x - cx; ay = nbr_1.y - cy; az = nbr_1.z - cz;
    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
    a22 += ay * ay; a23 += ay * az;
    a33 += az * az;
    ax = nbr_2.x - cx; ay = nbr_2.y - cy; az = nbr_2.z - cz;
    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
    a22 += ay * ay; a23 += ay * az;
    a33 += az * az;
    ax = nbr_3.x - cx; ay = nbr_3.y - cy; az = nbr_3.z - cz;
    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
    a22 += ay * ay; a23 += ay * az;
    a33 += az * az;
    ax = nbr_4.x - cx; ay = nbr_4.y - cy; az = nbr_4.z - cz;
    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
    a22 += ay * ay; a23 += ay * az;
    a33 += az * az;
    a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

    // Eigen::Matrix<float, 3, 3, Eigen::RowMajor> matA1;
    // Eigen::Matrix<float, 1, 3, Eigen::RowMajor> matD1;
    // Eigen::Matrix<float, 3, 3, Eigen::RowMajor> matV1;

    Eigen::Matrix3f matA1;
    Eigen::Vector3f matD1;
    Eigen::Matrix3f matV1;

    matA1(0, 0) = a11; matA1(0, 1) = a12; matA1(0, 2) = a13;
    matA1(1, 0) = a12; matA1(1, 1) = a22; matA1(1, 2) = a23;
    matA1(2, 0) = a13; matA1(2, 1) = a23; matA1(2, 2) = a33;

    Eigen::SelfAdjointEigenSolver< Eigen::Matrix3f > saes;
    saes.computeDirect(matA1);
    float eigen_val_0 = saes.eigenvalues()(2); float eigen_val_1 = saes.eigenvalues()(1); float eigen_val_2 = saes.eigenvalues()(0);
    Eigen::Vector3f eigen_vec = saes.eigenvectors().col(2);

    float4 coeff{0.0, 0.0, 0.0, 0.0};

    bool lineValid = (eigen_val_0 > 3.0 * eigen_val_1);
    if( !lineValid ) {
        d_vec_flag[i] = 0;
        d_vec_coeff[i] = coeff;
        return;
    }

    float4 point_sel = d_vec_point_sel[i];
    float x0 = point_sel.x;
    float y0 = point_sel.y;
    float z0 = point_sel.z;
    float x1 = cx + 0.1 * eigen_vec(0);
    float y1 = cy + 0.1 * eigen_vec(1);
    float z1 = cz + 0.1 * eigen_vec(2);
    float x2 = cx - 0.1 * eigen_vec(0);
    float y2 = cy - 0.1 * eigen_vec(1);
    float z2 = cz - 0.1 * eigen_vec(2);

    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                      + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                      + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                 - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                 + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

    float ld2 = a012 / l12;

    float s = 1 - 0.9 * fabs(ld2);

    if(s > 0.1) {
        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.w = s * ld2;
    }
    d_vec_coeff[i] = coeff;
    d_vec_flag[i] = (s > 0.1) ? 1 : 0;
}

void CalcCornerCoeffKernel::LaunchKernel(
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
    _CalcCornerCoeffKernel<<< num_blocks , num_threads_per_block , 0 , cuda_stream >>>(
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


