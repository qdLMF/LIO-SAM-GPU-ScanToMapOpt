//
// Created by lmf on 23-9-7.
//

#ifndef LIO_SAM_CUDA_CLOUD_HASH_MAP_CUH
#define LIO_SAM_CUDA_CLOUD_HASH_MAP_CUH

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


struct GridKey {
    int x;
    int y;
    int z;
    int w;

    __device__ __host__
    GridKey operator+(const GridKey& right) const {
        return {x + right.x, y + right.y, z + right.z, w + right.w};
    }

    __device__ __host__
    bool operator<(const GridKey& right) const {
        return x < right.x || (x == right.x && y < right.y) || (x == right.x && y == right.y && z < right.z);
    }

    __device__ __host__
    bool operator==(const GridKey& right) const {
        return x == right.x && y == right.y && z == right.z && w == right.w;
    }

    __device__ __host__
    GridKey& operator=(const GridKey& right) = default;
};

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
struct CUDACloudHashMap {
public :
    CUDACloudHashMap() = delete;
    explicit CUDACloudHashMap(
        float resolution_, 
        unsigned int mod_, 
        unsigned int reduction_factor_, 
        unsigned int max_insertion_size_
    );
    ~CUDACloudHashMap();

    void InsertV2(const thrust::host_vector<float4>& host_point);
    void Query(
        const thrust::device_vector<float4>& cloud_query,
        thrust::device_vector<char>& flag,
        thrust::device_vector<float4>& nbr_0,
        thrust::device_vector<float4>& nbr_1,
        thrust::device_vector<float4>& nbr_2,
        thrust::device_vector<float4>& nbr_3,
        thrust::device_vector<float4>& nbr_4
    );
    void Sync();
    void SyncAfterInsertion();
    void ClearBuckets();
    void ClearTempVectors();
    void Reset();
    

public :
    thrust::device_vector<float4> dev_point;
    
    thrust::device_vector<GridKey> dev_key;
    thrust::device_vector<int> dev_hash;

    thrust::device_vector<GridKey> dev_unique_by_key_key;
    thrust::device_vector<int> dev_unique_by_key_hash;
    thrust::device_vector<int> dev_unique_by_key_point_start;
    thrust::device_vector<int2> dev_unique_by_key_point_start_end;
    thrust::device_vector<int> dev_unique_by_key_key_idx;

    thrust::device_vector<int> dev_unique_by_hash_hash;
    thrust::device_vector<int> dev_unique_by_hash_key_start;
    thrust::device_vector<int2> dev_unique_by_hash_key_start_end;
    thrust::device_vector<int> dev_unique_by_hash_hash_idx;

    thrust::device_vector<int> dev_src_idx;

    thrust::device_vector<float4> dev_point_backup;
    thrust::device_vector<GridKey> dev_key_backup;
    thrust::device_vector<int> dev_hash_backup;
    thrust::device_vector<int2> dev_unique_by_key_point_start_end_backup;
    thrust::device_vector<GridKey> dev_unique_by_key_key_backup;

    // ----------

    thrust::device_vector<int> dev_from_hash_to_hash_idx;

    thrust::device_vector<GridKey> dev_key_bucket_key; // indexed with hash_idx
    thrust::device_vector<int> dev_key_bucket_key_idx; // indexed with hash_idx
    thrust::device_vector<int> dev_key_bucket_key_num; // indexed with hash_idx

    thrust::device_vector<float4> dev_point_bucket_point;  // indexed with key_idx
    thrust::device_vector<int> dev_point_bucket_point_num; // indexed with key_idx

    thrust::device_vector<int> dev_num_hash;
    thrust::device_vector<int> dev_num_keys;

    thrust::device_vector<int> dev_key_overflow_warning;


public :
    const float resolution;
    const unsigned int mod;
    const unsigned int reduction_factor;
    const unsigned int max_num_hash;
    const unsigned int max_num_keys;
    const unsigned int max_num_keys_per_hash;
    const unsigned int max_num_points_per_key;
    const unsigned int key_bucket_max_size;
    const unsigned int point_bucket_max_size;
    const unsigned int max_insertion_size;
    cudaStream_t stream;
    int num_keys;
    int num_hash;
    int key_overflow_warning;
    bool empty;
    int num_inserted_points = 0;
    int num_unique_keys_in_inserted_points = 0;
    int num_unique_hash_in_inserted_points = 0;
    float percent_used_hash = 0.0;
};


#endif //LIO_SAM_CUDA_CLOUD_HASH_MAP_CUH
