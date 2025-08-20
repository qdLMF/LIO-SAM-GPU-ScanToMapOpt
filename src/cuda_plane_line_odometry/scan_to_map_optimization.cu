//
// Created by lmf on 23-9-7.
//

#include <chrono>
#include <thread>

#include <eigen3/Eigen/Dense>

#include "./scan_to_map_optimization.cuh"


CUDAScanToMapOpt::CUDAScanToMapOpt(
    float resolution_,
    unsigned int max_num_hashes_, 
    unsigned int max_size_surf_insertion_,
    unsigned int max_size_corn_insertion_,
    unsigned int max_size_surf_query_,
    unsigned int max_size_corn_query_
) : surf_hash_map(
        resolution_,
        max_num_hashes_,
        max_size_surf_insertion_
    ),
    corn_hash_map(
        resolution_,
        max_num_hashes_,
        max_size_corn_insertion_
    ), 
    jac(max_size_surf_query_ + max_size_corn_query_, 6),
    res(max_size_surf_query_ + max_size_corn_query_, 1),
    max_size_surf_insertion(max_size_surf_insertion_),
    max_size_corn_insertion(max_size_corn_insertion_),
    max_size_surf_query(max_size_surf_query_),
    max_size_corn_query(max_size_corn_query_)
{
    surf_flag .reserve(max_size_surf_query);
    surf_sel  .reserve(max_size_surf_query);
    surf_nbr_0.reserve(max_size_surf_query);
    surf_nbr_1.reserve(max_size_surf_query);
    surf_nbr_2.reserve(max_size_surf_query);
    surf_nbr_3.reserve(max_size_surf_query);
    surf_nbr_4.reserve(max_size_surf_query);
    surf_coeff.reserve(max_size_surf_query);

    surf_flag .resize(max_size_surf_query,                       0);
    surf_sel  .resize(max_size_surf_query, make_float4(0, 0, 0, 0));
    surf_nbr_0.resize(max_size_surf_query, make_float4(0, 0, 0, 0));
    surf_nbr_1.resize(max_size_surf_query, make_float4(0, 0, 0, 0));
    surf_nbr_2.resize(max_size_surf_query, make_float4(0, 0, 0, 0));
    surf_nbr_3.resize(max_size_surf_query, make_float4(0, 0, 0, 0));
    surf_nbr_4.resize(max_size_surf_query, make_float4(0, 0, 0, 0));
    surf_coeff.resize(max_size_surf_query, make_float4(0, 0, 0, 0));

    corn_flag .reserve(max_size_corn_query);
    corn_sel  .reserve(max_size_corn_query);
    corn_nbr_0.reserve(max_size_corn_query);
    corn_nbr_1.reserve(max_size_corn_query);
    corn_nbr_2.reserve(max_size_corn_query);
    corn_nbr_3.reserve(max_size_corn_query);
    corn_nbr_4.reserve(max_size_corn_query);
    corn_coeff.reserve(max_size_corn_query);

    corn_flag .resize(max_size_corn_query,                       0);
    corn_sel  .resize(max_size_corn_query, make_float4(0, 0, 0, 0));
    corn_nbr_0.resize(max_size_corn_query, make_float4(0, 0, 0, 0));
    corn_nbr_1.resize(max_size_corn_query, make_float4(0, 0, 0, 0));
    corn_nbr_2.resize(max_size_corn_query, make_float4(0, 0, 0, 0));
    corn_nbr_3.resize(max_size_corn_query, make_float4(0, 0, 0, 0));
    corn_nbr_4.resize(max_size_corn_query, make_float4(0, 0, 0, 0));
    corn_coeff.resize(max_size_corn_query, make_float4(0, 0, 0, 0));

    surf_and_corn_flag .reserve(max_size_surf_query + max_size_corn_query);
    surf_and_corn_ori  .reserve(max_size_surf_query + max_size_corn_query);
    surf_and_corn_coeff.reserve(max_size_surf_query + max_size_corn_query);

    surf_and_corn_flag .resize(max_size_surf_query + max_size_corn_query,                       0);
    surf_and_corn_ori  .resize(max_size_surf_query + max_size_corn_query, make_float4(0, 0, 0, 0));
    surf_and_corn_coeff.resize(max_size_surf_query + max_size_corn_query, make_float4(0, 0, 0, 0));
}

void CUDAScanToMapOpt::Trans6ToTrans3x4() {
    float x = trans6[3], y = trans6[4], z = trans6[5], roll = trans6[0], pitch = trans6[1], yaw = trans6[2];
    float A = std::cos(yaw), B = std::sin(yaw), C = std::cos(pitch), D = std::sin(pitch), E = std::cos(roll), F = std::sin(roll), DE = D*E, DF = D*F;
    trans3x4[0][0] = A*C;  trans3x4[0][1] = A*DF - B*E;  trans3x4[0][2] = B*F + A*DE;  trans3x4[0][3] = x;
    trans3x4[1][0] = B*C;  trans3x4[1][1] = A*E + B*DF;  trans3x4[1][2] = B*DE - A*F;  trans3x4[1][3] = y;
    trans3x4[2][0] = -D;   trans3x4[2][1] = C*F;         trans3x4[2][2] = C*E;         trans3x4[2][3] = z;
}

void CUDAScanToMapOpt::Trans3x4ToTrans6() {
    float x = trans3x4[0][3];
    float y = trans3x4[1][3];
    float z = trans3x4[2][3];
    float roll  = std::atan2(trans3x4[2][1], trans3x4[2][2]);
    float pitch = std::asin(-trans3x4[2][0]);
    float yaw   = std::atan2(trans3x4[1][0], trans3x4[0][0]);
    trans6[0] = roll; trans6[1] = pitch; trans6[2] = yaw; trans6[3] = x; trans6[4] = y; trans6[5] = z;
}

void CUDAScanToMapOpt::SetAffineMatInit(const Eigen::Affine3f& mat_3x4) {
    trans3x4[0][0] = mat_3x4(0, 0); trans3x4[0][1] = mat_3x4(0, 1); trans3x4[0][2] = mat_3x4(0, 2); trans3x4[0][3] = mat_3x4(0, 3);
    trans3x4[1][0] = mat_3x4(1, 0); trans3x4[1][1] = mat_3x4(1, 1); trans3x4[1][2] = mat_3x4(1, 2); trans3x4[1][3] = mat_3x4(1, 3);
    trans3x4[2][0] = mat_3x4(2, 0); trans3x4[2][1] = mat_3x4(2, 1); trans3x4[2][2] = mat_3x4(2, 2); trans3x4[2][3] = mat_3x4(2, 3);

    Trans3x4ToTrans6();

    trans6_init[0] = trans6[0]; trans6_init[1] = trans6[1]; trans6_init[2] = trans6[2];
    trans6_init[3] = trans6[3]; trans6_init[4] = trans6[4]; trans6_init[5] = trans6[5];

    trans3x4_init[0][0] = trans3x4[0][0]; trans3x4_init[0][1] = trans3x4[0][1]; trans3x4_init[0][2] = trans3x4[0][2]; trans3x4_init[0][3] = trans3x4[0][3];
    trans3x4_init[1][0] = trans3x4[1][0]; trans3x4_init[1][1] = trans3x4[1][1]; trans3x4_init[1][2] = trans3x4[1][2]; trans3x4_init[1][3] = trans3x4[1][3];
    trans3x4_init[2][0] = trans3x4[2][0]; trans3x4_init[2][1] = trans3x4[2][1]; trans3x4_init[2][2] = trans3x4[2][2]; trans3x4_init[2][3] = trans3x4[2][3];

    iter_count = 0;
    degenerated = false;
    converged = false;
}

void CUDAScanToMapOpt::SetTrans6DOFInit(
    const Eigen::Matrix<float, 6, 1>& mat_6x1
) {
    trans6[0] = mat_6x1(0); trans6[1] = mat_6x1(1); trans6[2] = mat_6x1(2);
    trans6[3] = mat_6x1(3); trans6[4] = mat_6x1(4); trans6[5] = mat_6x1(5);

    Trans6ToTrans3x4();

    trans6_init[0] = trans6[0]; trans6_init[1] = trans6[1]; trans6_init[2] = trans6[2];
    trans6_init[3] = trans6[3]; trans6_init[4] = trans6[4]; trans6_init[5] = trans6[5];

    trans3x4_init[0][0] = trans3x4[0][0]; trans3x4_init[0][1] = trans3x4[0][1]; trans3x4_init[0][2] = trans3x4[0][2]; trans3x4_init[0][3] = trans3x4[0][3];
    trans3x4_init[1][0] = trans3x4[1][0]; trans3x4_init[1][1] = trans3x4[1][1]; trans3x4_init[1][2] = trans3x4[1][2]; trans3x4_init[1][3] = trans3x4[1][3];
    trans3x4_init[2][0] = trans3x4[2][0]; trans3x4_init[2][1] = trans3x4[2][1]; trans3x4_init[2][2] = trans3x4[2][2]; trans3x4_init[2][3] = trans3x4[2][3];

    iter_count = 0;
    degenerated = false;
    converged = false;
}

inline float rad2deg(float alpha) {
    return (alpha * 57.29578f);
}

void CUDAScanToMapOpt::ResetSurfMap () {
    surf_hash_map.Reset();
}

void CUDAScanToMapOpt::ResetCornMap () {
    corn_hash_map.Reset();
}

void CUDAScanToMapOpt::ResetSurfAndCornMap () {
    cudaStreamSynchronize(0);

    surf_hash_map.Reset();
    corn_hash_map.Reset();

    surf_hash_map.Sync();
    corn_hash_map.Sync();

    opt_count = 0;
}

void CUDAScanToMapOpt::InsertSurfAndCornToHashMap(
    const thrust::host_vector<float4>& host_surf_map_3d,
    const thrust::host_vector<float4>& host_corn_map_3d
) {
    cudaStreamSynchronize(0);

    if (!host_surf_map_3d.empty()) {
        surf_hash_map.InsertV2(host_surf_map_3d);
    }
    if (!host_corn_map_3d.empty()) {
        corn_hash_map.InsertV2(host_corn_map_3d);
    }

    if (!host_surf_map_3d.empty()) {
        surf_hash_map.SyncAfterInsertion();
    }
    if (!host_corn_map_3d.empty()) {
        corn_hash_map.SyncAfterInsertion();
    }
}

void CUDAScanToMapOpt::InsertSurfToHashMap(
    const thrust::host_vector<float4>& host_surf_map_3d
) {
    cudaStreamSynchronize(0);

    if (!host_surf_map_3d.empty()) {
        surf_hash_map.InsertV2(host_surf_map_3d);
    }

    if (!host_surf_map_3d.empty()) {
        surf_hash_map.SyncAfterInsertion();
    }
}

void CUDAScanToMapOpt::InsertCornToHashMap(const thrust::host_vector<float4>& host_corn_map_3d) {
    cudaStreamSynchronize(0);

    if (!host_corn_map_3d.empty()) {
        corn_hash_map.InsertV2(host_corn_map_3d);
    }

    if (!host_corn_map_3d.empty()) {
        corn_hash_map.SyncAfterInsertion();
    }
}

void CUDAScanToMapOpt::SetSurfPoints(
    const thrust::host_vector<float4>& surf_pts_3d
) {
    num_surf_points = surf_pts_3d.size();
    surf_ori = surf_pts_3d;
}

void CUDAScanToMapOpt::SetCornPoints(
    const thrust::host_vector<float4>& corn_pts_3d
) {
    num_corn_points = corn_pts_3d.size();
    corn_ori = corn_pts_3d;
}

void CUDAScanToMapOpt::TransformSurfPoints() {
    // thrust::fill(surf_flag .begin(), surf_flag .end(), 0);
    // thrust::fill(surf_sel  .begin(), surf_sel  .end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_0.begin(), surf_nbr_0.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_1.begin(), surf_nbr_1.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_2.begin(), surf_nbr_2.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_3.begin(), surf_nbr_3.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_4.begin(), surf_nbr_4.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_coeff.begin(), surf_coeff.end(), make_float4(0, 0, 0, 0));

    surf_associate_to_map.trans3x4.elem[0][0] = trans3x4[0][0];
    surf_associate_to_map.trans3x4.elem[0][1] = trans3x4[0][1];
    surf_associate_to_map.trans3x4.elem[0][2] = trans3x4[0][2];
    surf_associate_to_map.trans3x4.elem[0][3] = trans3x4[0][3];
    surf_associate_to_map.trans3x4.elem[1][0] = trans3x4[1][0];
    surf_associate_to_map.trans3x4.elem[1][1] = trans3x4[1][1];
    surf_associate_to_map.trans3x4.elem[1][2] = trans3x4[1][2];
    surf_associate_to_map.trans3x4.elem[1][3] = trans3x4[1][3];
    surf_associate_to_map.trans3x4.elem[2][0] = trans3x4[2][0];
    surf_associate_to_map.trans3x4.elem[2][1] = trans3x4[2][1];
    surf_associate_to_map.trans3x4.elem[2][2] = trans3x4[2][2];
    surf_associate_to_map.trans3x4.elem[2][3] = trans3x4[2][3];

    cudaStreamSynchronize(0);

    surf_associate_to_map.LaunchKernel(num_surf_points, surf_ori, surf_sel);
    surf_associate_to_map.Sync();
}

void CUDAScanToMapOpt::TransformCornPoints() {
    // thrust::fill(corn_flag .begin(), corn_flag .end(), 0);
    // thrust::fill(corn_sel  .begin(), corn_sel  .end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_0.begin(), corn_nbr_0.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_1.begin(), corn_nbr_1.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_2.begin(), corn_nbr_2.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_3.begin(), corn_nbr_3.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_4.begin(), corn_nbr_4.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_coeff.begin(), corn_coeff.end(), make_float4(0, 0, 0, 0));

    corn_associate_to_map.trans3x4.elem[0][0] = trans3x4[0][0];
    corn_associate_to_map.trans3x4.elem[0][1] = trans3x4[0][1];
    corn_associate_to_map.trans3x4.elem[0][2] = trans3x4[0][2];
    corn_associate_to_map.trans3x4.elem[0][3] = trans3x4[0][3];
    corn_associate_to_map.trans3x4.elem[1][0] = trans3x4[1][0];
    corn_associate_to_map.trans3x4.elem[1][1] = trans3x4[1][1];
    corn_associate_to_map.trans3x4.elem[1][2] = trans3x4[1][2];
    corn_associate_to_map.trans3x4.elem[1][3] = trans3x4[1][3];
    corn_associate_to_map.trans3x4.elem[2][0] = trans3x4[2][0];
    corn_associate_to_map.trans3x4.elem[2][1] = trans3x4[2][1];
    corn_associate_to_map.trans3x4.elem[2][2] = trans3x4[2][2];
    corn_associate_to_map.trans3x4.elem[2][3] = trans3x4[2][3];

    cudaStreamSynchronize(0);

    corn_associate_to_map.LaunchKernel(num_corn_points, corn_ori, corn_sel);
    corn_associate_to_map.Sync();
}

void CUDAScanToMapOpt::TransformSurfAndCornPoints() {
    // thrust::fill(surf_flag .begin(), surf_flag .end(), 0);
    // thrust::fill(surf_sel  .begin(), surf_sel  .end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_0.begin(), surf_nbr_0.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_1.begin(), surf_nbr_1.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_2.begin(), surf_nbr_2.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_3.begin(), surf_nbr_3.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_nbr_4.begin(), surf_nbr_4.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(surf_coeff.begin(), surf_coeff.end(), make_float4(0, 0, 0, 0));

    // thrust::fill(corn_flag .begin(), corn_flag .end(), 0);
    // thrust::fill(corn_sel  .begin(), corn_sel  .end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_0.begin(), corn_nbr_0.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_1.begin(), corn_nbr_1.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_2.begin(), corn_nbr_2.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_3.begin(), corn_nbr_3.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_nbr_4.begin(), corn_nbr_4.end(), make_float4(0, 0, 0, 0));
    // thrust::fill(corn_coeff.begin(), corn_coeff.end(), make_float4(0, 0, 0, 0));

    surf_associate_to_map.trans3x4.elem[0][0] = trans3x4[0][0];
    surf_associate_to_map.trans3x4.elem[0][1] = trans3x4[0][1];
    surf_associate_to_map.trans3x4.elem[0][2] = trans3x4[0][2];
    surf_associate_to_map.trans3x4.elem[0][3] = trans3x4[0][3];
    surf_associate_to_map.trans3x4.elem[1][0] = trans3x4[1][0];
    surf_associate_to_map.trans3x4.elem[1][1] = trans3x4[1][1];
    surf_associate_to_map.trans3x4.elem[1][2] = trans3x4[1][2];
    surf_associate_to_map.trans3x4.elem[1][3] = trans3x4[1][3];
    surf_associate_to_map.trans3x4.elem[2][0] = trans3x4[2][0];
    surf_associate_to_map.trans3x4.elem[2][1] = trans3x4[2][1];
    surf_associate_to_map.trans3x4.elem[2][2] = trans3x4[2][2];
    surf_associate_to_map.trans3x4.elem[2][3] = trans3x4[2][3];
    corn_associate_to_map.trans3x4.elem[0][0] = trans3x4[0][0];
    corn_associate_to_map.trans3x4.elem[0][1] = trans3x4[0][1];
    corn_associate_to_map.trans3x4.elem[0][2] = trans3x4[0][2];
    corn_associate_to_map.trans3x4.elem[0][3] = trans3x4[0][3];
    corn_associate_to_map.trans3x4.elem[1][0] = trans3x4[1][0];
    corn_associate_to_map.trans3x4.elem[1][1] = trans3x4[1][1];
    corn_associate_to_map.trans3x4.elem[1][2] = trans3x4[1][2];
    corn_associate_to_map.trans3x4.elem[1][3] = trans3x4[1][3];
    corn_associate_to_map.trans3x4.elem[2][0] = trans3x4[2][0];
    corn_associate_to_map.trans3x4.elem[2][1] = trans3x4[2][1];
    corn_associate_to_map.trans3x4.elem[2][2] = trans3x4[2][2];
    corn_associate_to_map.trans3x4.elem[2][3] = trans3x4[2][3];

    cudaStreamSynchronize(0);

    surf_associate_to_map.LaunchKernel(num_surf_points, surf_ori, surf_sel);
    corn_associate_to_map.LaunchKernel(num_corn_points, corn_ori, corn_sel);

    surf_associate_to_map.Sync();
    corn_associate_to_map.Sync();
}

void CUDAScanToMapOpt::SearchSurfPointsWithHashMap() {
    cudaStreamSynchronize(0);

    surf_hash_map.Query(surf_sel, surf_flag, surf_nbr_0, surf_nbr_1, surf_nbr_2, surf_nbr_3, surf_nbr_4);
    surf_hash_map.Sync();
}

void CUDAScanToMapOpt::SearchCornPointsWithHashMap() {
    cudaStreamSynchronize(0);

    corn_hash_map.Query(corn_sel, corn_flag, corn_nbr_0, corn_nbr_1, corn_nbr_2, corn_nbr_3, corn_nbr_4);
    corn_hash_map.Sync();
}

void CUDAScanToMapOpt::SearchSurfAndCornPointsWithHashMap() {
    cudaStreamSynchronize(0);

    surf_hash_map.Query(surf_sel, surf_flag, surf_nbr_0, surf_nbr_1, surf_nbr_2, surf_nbr_3, surf_nbr_4);
    corn_hash_map.Query(corn_sel, corn_flag, corn_nbr_0, corn_nbr_1, corn_nbr_2, corn_nbr_3, corn_nbr_4);

    surf_hash_map.Sync();
    corn_hash_map.Sync();
}

void CUDAScanToMapOpt::CalcSurfCoeff() {
    cudaStreamSynchronize(0);

    calc_surf_coeff.LaunchKernel(
        num_surf_points,
        surf_sel,
        surf_flag,
        surf_nbr_0,
        surf_nbr_1,
        surf_nbr_2,
        surf_nbr_3,
        surf_nbr_4,
        surf_coeff
    );
    calc_surf_coeff.Sync();
}

void CUDAScanToMapOpt::CalcCornCoeff() {
    cudaStreamSynchronize(0);

    calc_corn_coeff.LaunchKernel(
        num_corn_points,
        corn_sel,
        corn_flag,
        corn_nbr_0,
        corn_nbr_1,
        corn_nbr_2,
        corn_nbr_3,
        corn_nbr_4,
        corn_coeff
    );
    calc_corn_coeff.Sync();
}

void CUDAScanToMapOpt::CalcSurfAndCornCoeff() {
    cudaStreamSynchronize(0);

    calc_surf_coeff.LaunchKernel(
        num_surf_points,
        surf_sel,
        surf_flag,
        surf_nbr_0,
        surf_nbr_1,
        surf_nbr_2,
        surf_nbr_3,
        surf_nbr_4,
        surf_coeff
    );
    calc_corn_coeff.LaunchKernel(
        num_corn_points,
        corn_sel,
        corn_flag,
        corn_nbr_0,
        corn_nbr_1,
        corn_nbr_2,
        corn_nbr_3,
        corn_nbr_4,
        corn_coeff
    );

    calc_surf_coeff.Sync();
    calc_corn_coeff.Sync();
}

void CUDAScanToMapOpt::MallocForJacAndRes() {
    jac.resize(num_surf_points + num_corn_points, 6);
    res.resize(num_surf_points + num_corn_points, 1);

    thrust::copy(corn_ori.begin(), corn_ori.begin() + num_corn_points, surf_and_corn_ori.begin());
    thrust::copy(surf_ori.begin(), surf_ori.begin() + num_surf_points, surf_and_corn_ori.begin() + num_corn_points);
    
    thrust::copy(corn_flag.begin(), corn_flag.begin() + num_corn_points, surf_and_corn_flag.begin());
    thrust::copy(surf_flag.begin(), surf_flag.begin() + num_surf_points, surf_and_corn_flag.begin() + num_corn_points);
    
    thrust::copy(corn_coeff.begin(), corn_coeff.begin() + num_corn_points, surf_and_corn_coeff.begin());
    thrust::copy(surf_coeff.begin(), surf_coeff.begin() + num_surf_points, surf_and_corn_coeff.begin() + num_corn_points);
}

void CUDAScanToMapOpt::ComputeJacAndRes() {
    MallocForJacAndRes();

    cudaStreamSynchronize(0);

    compute_jac_and_res.LaunchKernel(
        num_surf_points + num_corn_points,
        surf_and_corn_ori,
        surf_and_corn_flag,
        surf_and_corn_coeff,
        trans6[0],
        trans6[1],
        trans6[2],
        jac,
        res
    );
    compute_jac_and_res.Sync();
}

void CUDAScanToMapOpt::UpdateTranform() {
    static Eigen::MatrixXf P = Eigen::Matrix<float, 6, 6>::Identity();

    Eigen::MatrixXf hes = cuda_AtA.Compute(jac); assert( (hes.rows() == 6) && (hes.cols() == 6) );
    Eigen::MatrixXf rhs = cuda_AtB.Compute(jac, res); assert( (rhs.rows() == 6) && (rhs.cols() == 1) );
    Eigen::MatrixXf sol = hes.fullPivHouseholderQr().solve(rhs); assert( (sol.rows() == 6) && (sol.cols() == 1) );

    if(iter_count == 0) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> saes(hes);
        Eigen::MatrixXf eigen_vec = saes.eigenvectors();
        degenerated = false;
        for(int i = 0; i < 6; i++) {
            if(saes.eigenvalues()(i) < 100.0) {
                eigen_vec.col(i).setZero();
                degenerated = true;
            } else {
                break;
            }
        }

        if(degenerated) {
            P = saes.eigenvectors() * eigen_vec.transpose();
        } else {
            P = Eigen::Matrix<float, 6, 6>::Identity();
        }
    }

    if(degenerated) {
        sol = P * sol;
    }

    trans6[0] += sol(0); trans6[1] += sol(1); trans6[2] += sol(2);
    trans6[3] += sol(3); trans6[4] += sol(4); trans6[5] += sol(5);

    Trans6ToTrans3x4();

    float deltaR = sqrtf(
        powf(rad2deg(sol(0)), 2) +
        powf(rad2deg(sol(1)), 2) +
        powf(rad2deg(sol(2)), 2)
    );
    float deltaT = sqrtf(
        powf(sol(3) * 100, 2) +
        powf(sol(4) * 100, 2) +
        powf(sol(5) * 100, 2)
    );

    if(deltaR < 0.05 && deltaT < 0.05) {
        converged = true;
    }

    iter_count++;
}

void CUDAScanToMapOpt::PrintStates() {
    printf(
        "CUDAScanToMapOpt::trans6 : %f , %f , %f , %f , %f , %f \n", 
        trans6[0], trans6[1], trans6[2], trans6[3], trans6[4], trans6[5]
    );

    if (surf_hash_map.key_overflow_warning != 0) {
        printf("surf_hash_map.num_hash             : %d \n", surf_hash_map.num_hash);
        printf("surf_hash_map.num_keys             : %d \n", surf_hash_map.num_keys);
        printf("surf_hash_map.key_overflow_warning : %d / %d \n", surf_hash_map.key_overflow_warning, surf_hash_map.max_num_keys_per_hash);
    } else {
        printf("surf_hash_map.num_hash : %d \n", surf_hash_map.num_hash);
        printf("surf_hash_map.num_keys : %d \n", surf_hash_map.num_keys);
    }

    if (corn_hash_map.key_overflow_warning != 0) {
        printf("corn_hash_map.num_hash             : %d \n", corn_hash_map.num_hash);
        printf("corn_hash_map.num_keys             : %d \n", corn_hash_map.num_keys);
        printf("corn_hash_map.key_overflow_warning : %d / %d \n", corn_hash_map.key_overflow_warning, corn_hash_map.max_num_keys_per_hash);
    } else {
        printf("corn_hash_map.num_hash : %d \n", corn_hash_map.num_hash);
        printf("corn_hash_map.num_keys : %d \n", corn_hash_map.num_keys);
    }
}





