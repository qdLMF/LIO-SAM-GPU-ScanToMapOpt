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

struct CUDACloudHashMapBase {
public : 
    CUDACloudHashMapBase() = default;
    virtual ~CUDACloudHashMapBase() = default;
    virtual void Allocate(
        float resolution_, 
        unsigned int mod_, 
        unsigned int reduction_factor_nrtr_, // numerator
        unsigned int reduction_factor_dntr_, // denominator
        unsigned int max_insertion_size_
    ) = 0;
    virtual void InsertV2(
        const thrust::host_vector<float4>& host_point
    ) = 0;
    virtual void Query(
        const thrust::device_vector<float4>& cloud_query,
        thrust::device_vector<char>& flag,
        thrust::device_vector<float4>& nbr_0,
        thrust::device_vector<float4>& nbr_1,
        thrust::device_vector<float4>& nbr_2,
        thrust::device_vector<float4>& nbr_3,
        thrust::device_vector<float4>& nbr_4
    ) = 0;
    virtual void Sync() = 0;
    virtual void SyncAfterInsertion() = 0;
    virtual void ClearBuckets() = 0;
    virtual void ClearTempVectors() = 0;
    virtual void Reset() = 0;

protected :
    cudaStream_t stream;

protected : 
    float        resolution;
    unsigned int mod;
    unsigned int reduction_factor_nrtr;
    unsigned int reduction_factor_dntr;
    unsigned int max_num_hash;
    unsigned int max_num_keys;
    unsigned int max_num_keys_per_hash;
    unsigned int max_num_points_per_key;
    unsigned int key_bucket_max_size;
    unsigned int point_bucket_max_size;
    unsigned int max_insertion_size;
    int          num_keys;
    int          num_hash;
    int          key_overflow_warning; // this is for overflow risk in any key bucket, key idx overflow risk is indicated by num_keys, if num_keys is >= max_num_keys, it means key indices are ran out, and we should expand capacity or reset local map.
    bool         empty;
    int          num_inserted_points = 0;
    int          num_unique_keys_in_inserted_points = 0;
    int          num_unique_hash_in_inserted_points = 0;
    float        percent_used_hash = 0.0;

public : 
    float        get_resolution()                         { return resolution;                         }
    unsigned int get_mod()                                { return mod;                                }
    unsigned int get_reduction_factor_nrtr()              { return reduction_factor_nrtr;              }
    unsigned int get_reduction_factor_dntr()              { return reduction_factor_dntr;              }
    unsigned int get_max_num_hash()                       { return max_num_hash;                       }
    unsigned int get_max_num_keys()                       { return max_num_keys;                       }
    unsigned int get_max_num_keys_per_hash()              { return max_num_keys_per_hash;              }
    unsigned int get_max_num_points_per_key()             { return max_num_points_per_key;             }
    unsigned int get_key_bucket_max_size()                { return key_bucket_max_size;                }
    unsigned int get_point_bucket_max_size()              { return point_bucket_max_size;              }
    unsigned int get_max_insertion_size()                 { return max_insertion_size;                 }
    int          get_num_keys()                           { return num_keys;                           }
    int          get_num_hash()                           { return num_hash;                           }
    int          get_key_overflow_warning()               { return key_overflow_warning;               }
    bool         get_empty()                              { return empty;                              }
    int          get_num_inserted_points()                { return num_inserted_points;                }
    int          get_num_unique_keys_in_inserted_points() { return num_unique_keys_in_inserted_points; }
    int          get_num_unique_hash_in_inserted_points() { return num_unique_hash_in_inserted_points; }
    float        get_percent_used_hash()                  { return percent_used_hash;                  }
};

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
struct CUDACloudHashMap : public CUDACloudHashMapBase {
public :
    CUDACloudHashMap() = default;
    ~CUDACloudHashMap();

    void Allocate(
        float resolution_, 
        unsigned int mod_, 
        unsigned int reduction_factor_nrtr_, 
        unsigned int reduction_factor_dntr_, 
        unsigned int max_insertion_size_
    ) override;
    void InsertV2(
        const thrust::host_vector<float4>& host_point
    ) override;
    void Query(
        const thrust::device_vector<float4>& cloud_query,
        thrust::device_vector<char>& flag,
        thrust::device_vector<float4>& nbr_0,
        thrust::device_vector<float4>& nbr_1,
        thrust::device_vector<float4>& nbr_2,
        thrust::device_vector<float4>& nbr_3,
        thrust::device_vector<float4>& nbr_4
    ) override;
    void Sync() override;
    void SyncAfterInsertion() override;
    void ClearBuckets() override;
    void ClearTempVectors() override;
    void Reset() override;

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
};

std::unique_ptr<CUDACloudHashMapBase> GetCUDACloudHashMapInstance(
    int key_bucket_size,
    int point_bucket_size
);

// TODO : 
// Three situations and corresponding strategies about adjusting capacity of hash map :
// max_num_hash is not enough, max_num_keys is enough : 
//     DoubleTheHashAndDoubleTheReductionFactor(), max_num_keys stays the same
// max_num_hash is not enough, max_num_keys is also not enough : 
//     DoubleTheHash(), max_num_keys doubles
//     DoubleTheHashAndDecreaseTheReductionFactor(numerator_increment), gradually increase max_num_keys by increasing reduction_factor_numerator by a small step
// max_num_hash is enough, max_num_keys is not enough : 
//     DecreaseTheReductionFactor(numerator_increment), gradually increase max_num_keys by increasing reduction_factor_numerator by a small step

#endif //LIO_SAM_CUDA_CLOUD_HASH_MAP_CUH
