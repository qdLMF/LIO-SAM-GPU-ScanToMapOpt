//
// Created by lmf on 23-9-7.
//

#ifndef LIO_SAM_CUDA_SCAN_TO_MAP_OPTIMIZATION_CUH
#define LIO_SAM_CUDA_SCAN_TO_MAP_OPTIMIZATION_CUH

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "./cuda_matrix.cuh"
#include "./cloud_hash_map.cuh"
#include "./point_associate_to_map.cuh"
#include "./calc_surf_coeff.cuh"
#include "./calc_corn_coeff.cuh"
#include "./compute_jac_and_res.cuh"
#include "./cuda_ata.cuh"
#include "./cuda_atb.cuh"

struct CUDAScanToMapOpt {
public :
    CUDAScanToMapOpt() = delete;
    explicit CUDAScanToMapOpt(
        float resolution,
        unsigned int max_num_hashes_, 
        unsigned int max_size_surf_insertion_,
        unsigned int max_size_corn_insertion_,
        unsigned int max_size_surf_query_,
        unsigned int max_size_corn_query_
    );

public :
    CUDACloudHashMap<16, 32> surf_hash_map;
    CUDACloudHashMap< 8, 16> corn_hash_map;
    PointAssociateToMapKernel surf_associate_to_map;
    PointAssociateToMapKernel corn_associate_to_map;
    CalcSurfCoeffKernel calc_surf_coeff;
    CalcCornCoeffKernel calc_corn_coeff;
    ComputeJacAndResKernel compute_jac_and_res;

public :
    // cudaError_t Test();
    void SetAffineMatInit(const Eigen::Affine3f& mat_3x4);
    void SetTrans6DOFInit(const Eigen::Matrix<float, 6, 1>& mat_6x1);
    void Trans3x4ToTrans6();
    void Trans6ToTrans3x4();
    void InsertSurfToHashMap(const thrust::host_vector<float4>& host_surf_map_3d);
    void InsertCornToHashMap(const thrust::host_vector<float4>& host_corn_map_3d);
    void InsertSurfAndCornToHashMap(
        const thrust::host_vector<float4>& host_surf_map_3d,
        const thrust::host_vector<float4>& host_corn_map_3d
    );
    void SetSurfPoints(const thrust::host_vector<float4>& surf_pts_3d);
    void SetCornPoints(const thrust::host_vector<float4>& corn_pts_3d);
    void TransformSurfPoints();
    void TransformCornPoints();
    void TransformSurfAndCornPoints();
    void SearchSurfPointsWithHashMap();
    void SearchCornPointsWithHashMap();
    void SearchSurfAndCornPointsWithHashMap();
    void CalcSurfCoeff();
    void CalcCornCoeff();
    void CalcSurfAndCornCoeff();
    void MallocForJacAndRes();
    void ComputeJacAndRes();
    void UpdateTranform();
    void PrintStates();
    void ResetSurfMap();
    void ResetCornMap();
    void ResetSurfAndCornMap();

public :
    thrust::device_vector<float4> surf_ori;
    thrust::device_vector<float4> surf_sel;

    thrust::device_vector<char> surf_flag;

    thrust::device_vector<float4> surf_nbr_0;
    thrust::device_vector<float4> surf_nbr_1;
    thrust::device_vector<float4> surf_nbr_2;
    thrust::device_vector<float4> surf_nbr_3;
    thrust::device_vector<float4> surf_nbr_4;

    thrust::device_vector<float4> surf_coeff;

public :
    thrust::device_vector<float4> corn_ori;
    thrust::device_vector<float4> corn_sel;

    thrust::device_vector<char> corn_flag;

    thrust::device_vector<float4> corn_nbr_0;
    thrust::device_vector<float4> corn_nbr_1;
    thrust::device_vector<float4> corn_nbr_2;
    thrust::device_vector<float4> corn_nbr_3;
    thrust::device_vector<float4> corn_nbr_4;

    thrust::device_vector<float4> corn_coeff;

public :
    thrust::device_vector<float4> surf_and_corn_ori;
    thrust::device_vector<char>   surf_and_corn_flag;
    thrust::device_vector<float4> surf_and_corn_coeff;

public :
    CUDAMatrix<float> jac;
    CUDAMatrix<float> res;

public :
    CUDAATA cuda_AtA;
    CUDAATB cuda_AtB;

public :
    unsigned int num_surf_points = 0;
    unsigned int num_corn_points = 0;
    unsigned int max_size_surf_insertion;
    unsigned int max_size_corn_insertion;
    unsigned int max_size_surf_query;
    unsigned int max_size_corn_query;
    unsigned int iter_count = 0;
    bool degenerated = false;
    bool converged = false;
    float trans3x4_init[3][4];
    float trans6_init[6];
    float trans3x4[3][4];
    float trans6[6]; // roll, pitch, yaw, x, y, z

    int opt_count = 0;
};

#endif //LIO_SAM_CUDA_SCAN_TO_MAP_OPTIMIZATION_CUH
