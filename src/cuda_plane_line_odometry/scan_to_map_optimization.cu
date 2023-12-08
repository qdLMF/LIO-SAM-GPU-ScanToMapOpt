//
// Created by lmf on 23-9-7.
//

#include <chrono>
#include <thread>

#include <eigen3/Eigen/Dense>

#include "scan_to_map_optimization.cuh"

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

void CUDAScanToMapOpt::SetTrans6DOFInit(const Eigen::Matrix<float, 6, 1>& mat_6x1) {
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

void CUDAScanToMapOpt::BuildSurfHashMap(const thrust::host_vector<float4>& host_surf_map_3d) {
    cudaStreamSynchronize(0);

    surf_hash_map.BuildMap(host_surf_map_3d);
    surf_hash_map.Sync();
    // surf_hash_map.ClearAfterBuildMap();
}

void CUDAScanToMapOpt::BuildCornerHashMap(const thrust::host_vector<float4>& host_corner_map_3d) {
    cudaStreamSynchronize(0);

    corner_hash_map.BuildMap(host_corner_map_3d);
    corner_hash_map.Sync();
    // corner_hash_map.ClearAfterBuildMap();
}

void CUDAScanToMapOpt::BuildSurfAndCornerHashMap(
    const thrust::host_vector<float4>& host_surf_map_3d,
    const thrust::host_vector<float4>& host_corner_map_3d
) {
    cudaStreamSynchronize(0);

    surf_hash_map.BuildMap(host_surf_map_3d);
    corner_hash_map.BuildMap(host_corner_map_3d);

    surf_hash_map.Sync();
    corner_hash_map.Sync();

    // surf_hash_map.ClearAfterBuildMap();
    // corner_hash_map.ClearAfterBuildMap();
}

void CUDAScanToMapOpt::SetSurfPoints(const thrust::host_vector<float4>& surf_pts_3d) {
    num_surf_points = surf_pts_3d.size();
    surf_ori = surf_pts_3d;
}

void CUDAScanToMapOpt::SetCornerPoints(const thrust::host_vector<float4>& corner_pts_3d) {
    num_corner_points = corner_pts_3d.size();
    corner_ori = corner_pts_3d;
}

void CUDAScanToMapOpt::TransformSurfPoints() {
    surf_sel.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_flag.resize(max_size_surf_query, 0);
    surf_nbr_0.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_1.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_2.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_3.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_4.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_coeff.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});

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

void CUDAScanToMapOpt::TransformCornerPoints() {
    corner_sel.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_flag.resize(max_size_corner_query, 0);
    corner_nbr_0.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_1.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_2.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_3.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_4.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_coeff.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});

    corner_associate_to_map.trans3x4.elem[0][0] = trans3x4[0][0];
    corner_associate_to_map.trans3x4.elem[0][1] = trans3x4[0][1];
    corner_associate_to_map.trans3x4.elem[0][2] = trans3x4[0][2];
    corner_associate_to_map.trans3x4.elem[0][3] = trans3x4[0][3];
    corner_associate_to_map.trans3x4.elem[1][0] = trans3x4[1][0];
    corner_associate_to_map.trans3x4.elem[1][1] = trans3x4[1][1];
    corner_associate_to_map.trans3x4.elem[1][2] = trans3x4[1][2];
    corner_associate_to_map.trans3x4.elem[1][3] = trans3x4[1][3];
    corner_associate_to_map.trans3x4.elem[2][0] = trans3x4[2][0];
    corner_associate_to_map.trans3x4.elem[2][1] = trans3x4[2][1];
    corner_associate_to_map.trans3x4.elem[2][2] = trans3x4[2][2];
    corner_associate_to_map.trans3x4.elem[2][3] = trans3x4[2][3];

    cudaStreamSynchronize(0);

    corner_associate_to_map.LaunchKernel(num_corner_points, corner_ori, corner_sel);
    corner_associate_to_map.Sync();
}

void CUDAScanToMapOpt::TransformSurfAndCornerPoints() {
    surf_sel.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_flag.resize(max_size_surf_query, 0);
    surf_nbr_0.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_1.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_2.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_3.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_nbr_4.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});
    surf_coeff.resize(max_size_surf_query, {0.0, 0.0, 0.0, 0.0});

    corner_sel.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_flag.resize(max_size_corner_query, 0);
    corner_nbr_0.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_1.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_2.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_3.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_nbr_4.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    corner_coeff.resize(max_size_corner_query, {0.0, 0.0, 0.0, 0.0});

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
    corner_associate_to_map.trans3x4.elem[0][0] = trans3x4[0][0];
    corner_associate_to_map.trans3x4.elem[0][1] = trans3x4[0][1];
    corner_associate_to_map.trans3x4.elem[0][2] = trans3x4[0][2];
    corner_associate_to_map.trans3x4.elem[0][3] = trans3x4[0][3];
    corner_associate_to_map.trans3x4.elem[1][0] = trans3x4[1][0];
    corner_associate_to_map.trans3x4.elem[1][1] = trans3x4[1][1];
    corner_associate_to_map.trans3x4.elem[1][2] = trans3x4[1][2];
    corner_associate_to_map.trans3x4.elem[1][3] = trans3x4[1][3];
    corner_associate_to_map.trans3x4.elem[2][0] = trans3x4[2][0];
    corner_associate_to_map.trans3x4.elem[2][1] = trans3x4[2][1];
    corner_associate_to_map.trans3x4.elem[2][2] = trans3x4[2][2];
    corner_associate_to_map.trans3x4.elem[2][3] = trans3x4[2][3];

    cudaStreamSynchronize(0);

    surf_associate_to_map.LaunchKernel(num_surf_points, surf_ori, surf_sel);
    corner_associate_to_map.LaunchKernel(num_corner_points, corner_ori, corner_sel);

    surf_associate_to_map.Sync();
    corner_associate_to_map.Sync();
}

void CUDAScanToMapOpt::SearchSurfPointsWithHashMap() {
    cudaStreamSynchronize(0);

    surf_hash_map.Query(surf_sel, surf_flag, surf_nbr_0, surf_nbr_1, surf_nbr_2, surf_nbr_3, surf_nbr_4);
    surf_hash_map.Sync();
}

void CUDAScanToMapOpt::SearchCornerPointsWithHashMap() {
    cudaStreamSynchronize(0);

    corner_hash_map.Query(corner_sel, corner_flag, corner_nbr_0, corner_nbr_1, corner_nbr_2, corner_nbr_3, corner_nbr_4);
    corner_hash_map.Sync();
}

void CUDAScanToMapOpt::SearchSurfAndCornerPointsWithHashMap() {
    cudaStreamSynchronize(0);

    surf_hash_map.Query(surf_sel, surf_flag, surf_nbr_0, surf_nbr_1, surf_nbr_2, surf_nbr_3, surf_nbr_4);
    corner_hash_map.Query(corner_sel, corner_flag, corner_nbr_0, corner_nbr_1, corner_nbr_2, corner_nbr_3, corner_nbr_4);

    surf_hash_map.Sync();
    corner_hash_map.Sync();
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

void CUDAScanToMapOpt::CalcCornerCoeff() {
    cudaStreamSynchronize(0);

    calc_corner_coeff.LaunchKernel(
        num_corner_points,
        corner_sel,
        corner_flag,
        corner_nbr_0,
        corner_nbr_1,
        corner_nbr_2,
        corner_nbr_3,
        corner_nbr_4,
        corner_coeff
    );
    calc_corner_coeff.Sync();
}

void CUDAScanToMapOpt::CalcSurfAndCornerCoeff() {
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
    calc_corner_coeff.LaunchKernel(
        num_corner_points,
        corner_sel,
        corner_flag,
        corner_nbr_0,
        corner_nbr_1,
        corner_nbr_2,
        corner_nbr_3,
        corner_nbr_4,
        corner_coeff
    );

    calc_surf_coeff.Sync();
    calc_corner_coeff.Sync();
}

void CUDAScanToMapOpt::MallocForJacAndRes() {
    jac.resize(num_surf_points + num_corner_points, 6);
    res.resize(num_surf_points + num_corner_points, 1);

    surf_and_corner_ori.resize(max_size_surf_query + max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    thrust::copy(corner_ori.begin(), corner_ori.begin() + num_corner_points, surf_and_corner_ori.begin());
    thrust::copy(surf_ori.begin(), surf_ori.begin() + num_surf_points, surf_and_corner_ori.begin() + num_corner_points);

    surf_and_corner_flag.resize(max_size_surf_query + max_size_corner_query, 0);
    thrust::copy(corner_flag.begin(), corner_flag.begin() + num_corner_points, surf_and_corner_flag.begin());
    thrust::copy(surf_flag.begin(), surf_flag.begin() + num_surf_points, surf_and_corner_flag.begin() + num_corner_points);

    surf_and_corner_coeff.resize(max_size_surf_query + max_size_corner_query, {0.0, 0.0, 0.0, 0.0});
    thrust::copy(corner_coeff.begin(), corner_coeff.begin() + num_corner_points, surf_and_corner_coeff.begin());
    thrust::copy(surf_coeff.begin(), surf_coeff.begin() + num_surf_points, surf_and_corner_coeff.begin() + num_corner_points);
}

void CUDAScanToMapOpt::ComputeJacAndRes() {
    MallocForJacAndRes();

    cudaStreamSynchronize(0);

    compute_jac_and_res.LaunchKernel(
        num_surf_points + num_corner_points,
        surf_and_corner_ori,
        surf_and_corner_flag,
        surf_and_corner_coeff,
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
}




