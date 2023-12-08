//
// Created by lmf on 23-9-7.
//

#ifndef LIO_SAM_CUDA_CLOUD_HASH_MAP_CUH
#define LIO_SAM_CUDA_CLOUD_HASH_MAP_CUH

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MAX_NUM_KEYS_PER_HASH 8
#define MAX_NUM_POINTS_PER_KEY 8

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

struct CUDACloudHashMap {
public :
    explicit CUDACloudHashMap(unsigned int max_map_size_);
    ~CUDACloudHashMap();

    void BuildMapPre();
    void BuildMap(const thrust::host_vector<float4>& cloud_map_3d);
    void Query(
        const thrust::device_vector<float4>& cloud_query_3d,
        thrust::device_vector<char>& flag,
        thrust::device_vector<float4>& nbr_0,
        thrust::device_vector<float4>& nbr_1,
        thrust::device_vector<float4>& nbr_2,
        thrust::device_vector<float4>& nbr_3,
        thrust::device_vector<float4>& nbr_4
    );

    void Sync();

    void ClearAfterBuildMap();

public :
    thrust::device_vector<float4> dev_point_3d;

    thrust::device_vector<GridKey> dev_key;
    thrust::device_vector<int> dev_hash;

    thrust::device_vector<GridKey> dev_key_backup;
    thrust::device_vector<int> dev_hash_backup;
    // thrust::device_vector<int> dev_key_idx_backup;
    // thrust::device_vector<int> dev_hash_idx_backup;

    thrust::device_vector<GridKey> dev_unique_by_key_key;
    thrust::device_vector<int> dev_unique_by_key_hash;
    thrust::device_vector<int> dev_unique_by_key_point_start;  // temp
    thrust::device_vector<int2> dev_unique_by_key_point_start_end;
    thrust::device_vector<int> dev_unique_by_key_key_idx;
    // thrust::device_vector<float4> dev_point_buckets;    // indexed with key_idx // useless

    thrust::device_vector<int> dev_unique_by_hash_hash;
    thrust::device_vector<int> dev_unique_by_hash_key_start;    // temp
    thrust::device_vector<int2> dev_unique_by_hash_key_start_end;
    thrust::device_vector<int> dev_unique_by_hash_hash_idx;

    thrust::device_vector<GridKey> dev_key_bucket_key;              // indexed with hash_idx
    thrust::device_vector<int> dev_key_bucket_key_idx;              // indexed with hash_idx

    thrust::device_vector<float4> dev_point_bucket_point;           // indexed with key_idx

    thrust::device_vector<int> dev_from_hash_to_hash_idx;

    // thrust::device_vector<int> dev_unique_by_key_hash_idx;
    // thrust::device_vector<int> dev_num_unique_keys_in_map;

public :
    unsigned int max_map_size;
    unsigned int max_query_size;
    cudaStream_t stream;
};


#endif //LIO_SAM_CUDA_CLOUD_HASH_MAP_CUH
