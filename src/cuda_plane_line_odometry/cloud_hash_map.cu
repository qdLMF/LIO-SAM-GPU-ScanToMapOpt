//
// Created by lmf on 23-9-7.
//

#include <chrono>
#include <unordered_map>
#include <map>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "./cloud_hash_map.cuh"


// #define RESOLUTION float(0.5)
// #define MOD 100000

#define MAX_NUM_KEYS_PER_HASH  8
#define MAX_NUM_POINTS_PER_KEY 32

#define KEY_BUCKET_MAX_SIZE   (MAX_NUM_KEYS_PER_HASH * MOD)
#define POINT_BUCKET_MAX_SIZE (MAX_NUM_KEYS_PER_HASH * MOD * MAX_NUM_POINTS_PER_KEY)

#define NTHREADS_BUILD_MAP  256 // 256
#define NTHREADS_KNN_SEARCH 256 // 256

// ----------

__device__ __host__ GridKey key_func(
    const float4& point, 
    float resolution
) {
    return {
        int(round(point.x / resolution)),
        int(round(point.y / resolution)),
        int(round(point.z / resolution)),
        0
    };
}

__device__ __host__ int hash_func(
    const GridKey& key, 
    unsigned int mod
) {
    return int(size_t(((key.x) * 73856093) ^ ((key.y) * 471943) ^ ((key.z) * 83492791)) % mod);
}

// ----------

__global__ void kernel_compute_key_and_hash(
    int num_points, 
    float resolution, 
    unsigned int mod, 
    const float4* point_,
    GridKey* key_,
    int* hash_
);

__global__ void kernel_compute_point_start_end(
    int num_unique_keys,
    int num_map_points,
    const int* point_start_,
    int2* point_start_end_
);

__global__ void kernel_compute_key_start_end(
    int num_unique_hash,
    int num_unique_keys,
    const int* key_start_,
    int2* key_start_end_
);

template<int KEY_BUCKET_SIZE>
__global__ void kernel_insert_key_to_key_bucket_when_map_is_empty(
    int num_unique_hash,
    const int* unique_by_hash_hash_,
    const int* unique_by_hash_hash_idx_,
    const int2* unique_by_hash_key_start_end_,
    const GridKey* unique_by_key_key_,
    int* unique_by_key_key_idx_,
    GridKey* key_bucket_key_,
    int* key_bucket_key_idx_,
    int* key_bucket_key_num_,
    int* from_hash_to_hash_idx_,
    int* key_overflow_warning
);

template<int KEY_BUCKET_SIZE>
__global__ void kernel_insert_key_to_key_bucket_when_map_is_not_empty(
    int num_unique_hash,
    int* num_hash_,
    int* num_keys_,
    const int* unique_by_hash_hash_,
    const int2* unique_by_hash_key_start_end_,
    const GridKey* unique_by_key_key_,
    int* unique_by_key_key_idx_,
    int* from_hash_to_hash_idx_,
    GridKey* key_bucket_key_,
    int* key_bucket_key_idx_,
    int* key_bucket_key_num_,
    int* key_overflow_warning
);

template<int POINT_BUCKET_SIZE>
__global__ void kernel_insert_point_to_point_bucket_when_map_is_empty(
    int num_unique_keys,
    const int* unique_by_key_key_idx_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_,
    float4* point_bucket_point_,
    int* point_bucket_point_num_
);

template<int POINT_BUCKET_SIZE>
__global__ void kernel_insert_point_to_point_bucket_when_map_is_not_empty(
    int num_unique_keys,
    const int* unique_by_key_key_idx_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_,
    float4* point_bucket_point_,
    int* point_bucket_point_num_
);

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
__global__ void kernel_5nn_search(
    int num_queries,
    float resolution,
    unsigned int mod,
    const float4* query_,
    const int* from_hash_to_hash_idx_,
    const GridKey* key_bucket_key_,
    const int* key_bucket_key_idx_,
    const int* key_bucket_key_num_,
    const float4* point_bucket_point_,
    const int* point_bucket_point_num_,
    char* flag,
    float4* nbr_0,
    float4* nbr_1,
    float4* nbr_2,
    float4* nbr_3,
    float4* nbr_4
);

__global__ void kernel_rearrange_point_and_hash(
    int num_map_points,
    const int* src_idx_,
    const float4* point_src_,
    const int* hash_src_,
    float4* point_,
    int* hash_
);

__global__ void kernel_rearrange_unique_by_key(
    int num_unique_keys,
    const int* src_idx_,
    const int2* unique_by_key_point_start_end_src_,
    const GridKey* unique_by_key_key_src_,
    int2* unique_by_key_point_start_end_,
    GridKey* unique_by_key_key_
);

// ----------

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::CUDACloudHashMap(
    float resolution_, 
    unsigned int max_num_hashes_, 
    unsigned int max_insertion_size_
) : resolution(resolution_), 
    max_num_hashes(max_num_hashes_), 
    mod(max_num_hashes_),
    max_num_keys_per_hash(KEY_BUCKET_SIZE), 
    max_num_points_per_key(POINT_BUCKET_SIZE), 
    key_bucket_max_size(max_num_hashes_ * KEY_BUCKET_SIZE),
    point_bucket_max_size(max_num_hashes_ * KEY_BUCKET_SIZE * POINT_BUCKET_SIZE),
    max_insertion_size(max_insertion_size_)
{
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // ----------------------------------------------------------------------------------------------------
    dev_point       .reserve(max_insertion_size);
    dev_point_backup.reserve(max_insertion_size);

    dev_from_hash_to_hash_idx.reserve(mod);

    dev_key_bucket_key    .reserve(key_bucket_max_size);
    dev_key_bucket_key_idx.reserve(key_bucket_max_size);
    dev_key_bucket_key_num.reserve(                mod);

    dev_point_bucket_point    .reserve(point_bucket_max_size);
    dev_point_bucket_point_num.reserve(  key_bucket_max_size);

    dev_key .reserve(max_insertion_size);
    dev_hash.reserve(max_insertion_size);

    dev_unique_by_key_key            .reserve(max_insertion_size);
    dev_unique_by_key_hash           .reserve(max_insertion_size);
    dev_unique_by_key_point_start    .reserve(max_insertion_size);
    dev_unique_by_key_point_start_end.reserve(max_insertion_size);
    dev_unique_by_key_key_idx        .reserve(max_insertion_size);

    dev_unique_by_hash_hash         .reserve(max_insertion_size);
    dev_unique_by_hash_key_start    .reserve(max_insertion_size);
    dev_unique_by_hash_key_start_end.reserve(max_insertion_size);
    dev_unique_by_hash_hash_idx     .reserve(max_insertion_size);

    dev_key_backup .reserve(max_insertion_size);
    dev_hash_backup.reserve(max_insertion_size);

    dev_src_idx.reserve(max_insertion_size);

    dev_unique_by_key_point_start_end_backup.reserve(max_insertion_size);
    dev_unique_by_key_key_backup            .reserve(max_insertion_size);

    dev_num_hash.reserve(1);
    dev_num_keys.reserve(1);

    dev_key_overflow_warning.reserve(1);
    // ----------------------------------------------------------------------------------------------------

    // ----------------------------------------------------------------------------------------------------
    // dev_point       .resize(max_insertion_size); // no need
    // dev_point_backup.resize(max_insertion_size); // no need

    dev_from_hash_to_hash_idx.resize(mod, -1);

    dev_key_bucket_key    .resize(key_bucket_max_size, GridKey{0, 0, 0, 1});
    dev_key_bucket_key_idx.resize(key_bucket_max_size,                  -1);
    dev_key_bucket_key_num.resize(                mod,                   0);

    dev_point_bucket_point    .resize(point_bucket_max_size, make_float4(0.0, 0.0, 0.0, 0.0));
    dev_point_bucket_point_num.resize(  key_bucket_max_size,                               0);

    dev_key .resize(max_insertion_size, GridKey{0, 0, 0, 1});
    dev_hash.resize(max_insertion_size,                  -1);

    dev_unique_by_key_key            .resize(max_insertion_size, GridKey{0, 0, 0, 1});
    dev_unique_by_key_hash           .resize(max_insertion_size,                  -1);
    dev_unique_by_key_point_start    .resize(max_insertion_size,                  -1);
    dev_unique_by_key_point_start_end.resize(max_insertion_size,   make_int2(-1, -1));
    dev_unique_by_key_key_idx        .resize(max_insertion_size,                  -1);

    dev_unique_by_hash_hash         .resize(max_insertion_size,                -1);
    dev_unique_by_hash_key_start    .resize(max_insertion_size,                -1);
    dev_unique_by_hash_key_start_end.resize(max_insertion_size, make_int2(-1, -1));
    dev_unique_by_hash_hash_idx     .resize(max_insertion_size,                -1);

    dev_key_backup .resize(max_insertion_size, GridKey{0, 0, 0, 1});
    dev_hash_backup.resize(max_insertion_size,                  -1);

    dev_src_idx.resize(max_insertion_size, -1);

    dev_unique_by_key_point_start_end_backup.resize(max_insertion_size,   make_int2(-1, -1));
    dev_unique_by_key_key_backup            .resize(max_insertion_size, GridKey{0, 0, 0, 1});

    dev_num_hash.resize(1, 0);
    dev_num_keys.resize(1, 0);

    dev_key_overflow_warning.resize(1, 0);
    // ----------------------------------------------------------------------------------------------------

    Reset();
}

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::~CUDACloudHashMap() {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
void CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::SyncAfterInsertion() {
    cudaStreamSynchronize(stream);

    num_hash = dev_num_hash[0];
    num_keys = dev_num_keys[0];
    key_overflow_warning = dev_key_overflow_warning[0];
    percent_used_hash = float(num_hash) / float(mod);
    // ClearTempVectors();
}

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
void CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::Sync() {
    cudaStreamSynchronize(stream);
}

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
void CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::ClearBuckets() {
    thrust::fill(dev_from_hash_to_hash_idx .begin(), dev_from_hash_to_hash_idx .end(), -1);
    thrust::fill(dev_key_bucket_key_num    .begin(), dev_key_bucket_key_num    .end(),  0);
    thrust::fill(dev_point_bucket_point_num.begin(), dev_point_bucket_point_num.end(),  0);
}

// 开销大，但没必要，以下所有向量在读取之前都会被写入正确值，所以不需要调用该函数
template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
void CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::ClearTempVectors() {
    thrust::fill(dev_key .begin(), dev_key .end(), GridKey{0, 0, 0, 1});
    thrust::fill(dev_hash.begin(), dev_hash.end(),                  -1);

    thrust::fill(dev_unique_by_key_key            .begin(), dev_unique_by_key_key            .end(), GridKey{0, 0, 0, 1});
    thrust::fill(dev_unique_by_key_hash           .begin(), dev_unique_by_key_hash           .end(),                  -1);
    thrust::fill(dev_unique_by_key_point_start    .begin(), dev_unique_by_key_point_start    .end(),                  -1);
    thrust::fill(dev_unique_by_key_point_start_end.begin(), dev_unique_by_key_point_start_end.end(),   make_int2(-1, -1));
    thrust::fill(dev_unique_by_key_key_idx        .begin(), dev_unique_by_key_key_idx        .end(),                  -1);

    thrust::fill(dev_unique_by_hash_hash         .begin(), dev_unique_by_hash_hash         .end(),                -1);
    thrust::fill(dev_unique_by_hash_key_start    .begin(), dev_unique_by_hash_key_start    .end(),                -1);
    thrust::fill(dev_unique_by_hash_key_start_end.begin(), dev_unique_by_hash_key_start_end.end(), make_int2(-1, -1));
    thrust::fill(dev_unique_by_hash_hash_idx     .begin(), dev_unique_by_hash_hash_idx     .end(),                -1);

    thrust::fill(dev_src_idx.begin(), dev_src_idx.end(), -1);

    thrust::fill(dev_key_backup .begin(), dev_key_backup .end(), GridKey{0, 0, 0, 1});
    thrust::fill(dev_hash_backup.begin(), dev_hash_backup.end(),                  -1);

    thrust::fill(dev_unique_by_key_point_start_end_backup.begin(), dev_unique_by_key_point_start_end_backup.end(),   make_int2(-1, -1));
    thrust::fill(dev_unique_by_key_key_backup            .begin(), dev_unique_by_key_key_backup            .end(), GridKey{0, 0, 0, 1});
}

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
void CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::Reset() {
    cudaStreamSynchronize(stream);

    dev_num_hash[0] = 0;
    dev_num_keys[0] = 0;
    dev_key_overflow_warning[0] = 0;
    num_inserted_points = 0;
    num_unique_keys_in_inserted_points = 0;
    num_unique_hash_in_inserted_points = 0;
    num_keys = 0;
    num_hash = 0;
    empty = true;
    ClearBuckets();
    // ClearTempVectors();
}

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
void CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::InsertV2(const thrust::host_vector<float4> &host_point) {
    if (host_point.empty()) {
        num_inserted_points = 0;
        num_unique_keys_in_inserted_points = 0;
        num_unique_hash_in_inserted_points = 0;
        return;
    }

    dev_point = host_point;
    dev_point_backup.resize(dev_point.size(), make_float4(0, 0, 0, 0));

    thrust::counting_iterator<int> c_iter_1(0);
    thrust::counting_iterator<int> c_iter_2(0);
    thrust::counting_iterator<int> c_iter_3(0);
    thrust::counting_iterator<int> c_iter_4(0);


    // ------------------------------------------------------------------------------------------------------------------------
    num_inserted_points = dev_point.size();
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_inserted_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_compute_key_and_hash<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_inserted_points, 
            resolution, 
            mod, 
            thrust::raw_pointer_cast(&(dev_point[0])),
            thrust::raw_pointer_cast(&(dev_key[0])),
            thrust::raw_pointer_cast(&(dev_hash[0]))
        );
    }
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // sort by key
    thrust::copy(thrust::cuda::par.on(stream), dev_point.begin(), dev_point.begin() + num_inserted_points, dev_point_backup.begin());
    thrust::copy(thrust::cuda::par.on(stream), dev_hash.begin(), dev_hash.begin() + num_inserted_points, dev_hash_backup.begin());
    thrust::copy(thrust::cuda::par.on(stream), c_iter_4, c_iter_4 + num_inserted_points, dev_src_idx.begin());
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        dev_key.begin(),
        dev_key.begin() + num_inserted_points,
        dev_src_idx.begin()
    );
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_inserted_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_rearrange_point_and_hash<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_inserted_points,
            thrust::raw_pointer_cast(&(dev_src_idx[0])),
            thrust::raw_pointer_cast(&(dev_point_backup[0])),
            thrust::raw_pointer_cast(&(dev_hash_backup[0])),
            thrust::raw_pointer_cast(&(dev_point[0])),
            thrust::raw_pointer_cast(&(dev_hash[0]))
        );
    }
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // unique by key
    num_unique_keys_in_inserted_points = \
    thrust::unique_by_key_copy(
        thrust::cuda::par.on(stream),
        dev_key.begin(),
        dev_key.begin() + num_inserted_points,
        thrust::counting_iterator<int>(0),
        dev_unique_by_key_key.begin(),
        dev_unique_by_key_point_start.begin()
    ).first - dev_unique_by_key_key.begin();
    num_unique_keys_in_inserted_points = \
    thrust::unique_by_key_copy(
        thrust::cuda::par.on(stream),
        dev_key.begin(),
        dev_key.begin() + num_inserted_points,
        dev_hash.begin(),
        dev_unique_by_key_key.begin(),
        dev_unique_by_key_hash.begin()
    ).first - dev_unique_by_key_key.begin();

    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_keys_in_inserted_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_compute_point_start_end<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_unique_keys_in_inserted_points,
            num_inserted_points,
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end[0]))
        );
    }

    // and then sort by hash
    thrust::copy(thrust::cuda::par.on(stream), dev_unique_by_key_point_start_end.begin(), dev_unique_by_key_point_start_end.begin() + num_unique_keys_in_inserted_points, dev_unique_by_key_point_start_end_backup.begin());
    thrust::copy(thrust::cuda::par.on(stream), dev_unique_by_key_key.begin(), dev_unique_by_key_key.begin() + num_unique_keys_in_inserted_points, dev_unique_by_key_key_backup.begin());
    thrust::copy(thrust::cuda::par.on(stream), c_iter_3, c_iter_3 + num_unique_keys_in_inserted_points, dev_src_idx.begin());
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        dev_unique_by_key_hash.begin(),
        dev_unique_by_key_hash.begin() + num_unique_keys_in_inserted_points,
        dev_src_idx.begin()
    );
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_keys_in_inserted_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_rearrange_unique_by_key<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_unique_keys_in_inserted_points,
            thrust::raw_pointer_cast(&(dev_src_idx[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end_backup[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_key_backup[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_key[0]))
        );
    }

    thrust::copy(
        thrust::cuda::par.on(stream), 
        c_iter_1, 
        c_iter_1 + num_unique_keys_in_inserted_points, 
        dev_unique_by_key_key_idx.begin()
    );

    if (empty) {
        cudaStreamSynchronize(stream);
        dev_num_keys[0] = num_unique_keys_in_inserted_points;
    }
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // unique by hash
    num_unique_hash_in_inserted_points = \
    thrust::unique_by_key_copy(
        thrust::cuda::par.on(stream),
        dev_unique_by_key_hash.begin(),
        dev_unique_by_key_hash.begin() + num_unique_keys_in_inserted_points,
        dev_unique_by_key_key_idx.begin(),
        dev_unique_by_hash_hash.begin(),
        dev_unique_by_hash_key_start.begin()
    ).first - dev_unique_by_hash_hash.begin();

    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_hash_in_inserted_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_compute_key_start_end<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_unique_hash_in_inserted_points,
            num_unique_keys_in_inserted_points,
            thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start_end[0]))    // key_idx in dev_unique_by_key_key
        );
    }

    thrust::copy(
        thrust::cuda::par.on(stream), 
        c_iter_2, 
        c_iter_2 + num_unique_hash_in_inserted_points, 
        dev_unique_by_hash_hash_idx.begin()
    );

    if (empty) {
        thrust::scatter(
            thrust::cuda::par.on(stream),
            dev_unique_by_hash_hash_idx.begin(),
            dev_unique_by_hash_hash_idx.begin() + num_unique_hash_in_inserted_points,
            dev_unique_by_hash_hash.begin(),
            dev_from_hash_to_hash_idx.begin()
        );

        cudaStreamSynchronize(stream);
        dev_num_hash[0] = num_unique_hash_in_inserted_points;
    }
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // update key bucket
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_hash_in_inserted_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        if (empty) {
            kernel_insert_key_to_key_bucket_when_map_is_empty<KEY_BUCKET_SIZE><<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
                num_unique_hash_in_inserted_points,
                thrust::raw_pointer_cast(&(dev_unique_by_hash_hash[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_hash_hash_idx[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start_end[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_key_key[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_key_key_idx[0])),
                thrust::raw_pointer_cast(&(dev_key_bucket_key[0])),
                thrust::raw_pointer_cast(&(dev_key_bucket_key_idx[0])),
                thrust::raw_pointer_cast(&(dev_key_bucket_key_num[0])),
                thrust::raw_pointer_cast(&(dev_from_hash_to_hash_idx[0])),
                thrust::raw_pointer_cast(&(dev_key_overflow_warning[0]))
            );
        } else {
            kernel_insert_key_to_key_bucket_when_map_is_not_empty<KEY_BUCKET_SIZE><<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
                num_unique_hash_in_inserted_points,
                thrust::raw_pointer_cast(&(dev_num_hash[0])),
                thrust::raw_pointer_cast(&(dev_num_keys[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_hash_hash[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start_end[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_key_key[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_key_key_idx[0])),
                thrust::raw_pointer_cast(&(dev_from_hash_to_hash_idx[0])),
                thrust::raw_pointer_cast(&(dev_key_bucket_key[0])),
                thrust::raw_pointer_cast(&(dev_key_bucket_key_idx[0])),
                thrust::raw_pointer_cast(&(dev_key_bucket_key_num[0])),
                thrust::raw_pointer_cast(&(dev_key_overflow_warning[0]))
            );
        }
    }
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // update point bucket
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_keys_in_inserted_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        if (empty) {
            kernel_insert_point_to_point_bucket_when_map_is_empty<POINT_BUCKET_SIZE><<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
                num_unique_keys_in_inserted_points,
                thrust::raw_pointer_cast(&(dev_unique_by_key_key_idx[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end[0])),
                thrust::raw_pointer_cast(&(dev_point[0])),
                thrust::raw_pointer_cast(&(dev_point_bucket_point[0])),
                thrust::raw_pointer_cast(&(dev_point_bucket_point_num[0]))
            );
        } else {
            kernel_insert_point_to_point_bucket_when_map_is_not_empty<POINT_BUCKET_SIZE><<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
                num_unique_keys_in_inserted_points,
                thrust::raw_pointer_cast(&(dev_unique_by_key_key_idx[0])),
                thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end[0])),
                thrust::raw_pointer_cast(&(dev_point[0])),
                thrust::raw_pointer_cast(&(dev_point_bucket_point[0])),
                thrust::raw_pointer_cast(&(dev_point_bucket_point_num[0]))
            );
        }
    }
    // ------------------------------------------------------------------------------------------------------------------------


    if (empty) {
        empty = false;
    }
}

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
void CUDACloudHashMap<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE>::Query(
    const thrust::device_vector<float4>& cloud_query_3d,
    thrust::device_vector<char>& flag,
    thrust::device_vector<float4>& nbr_0,
    thrust::device_vector<float4>& nbr_1,
    thrust::device_vector<float4>& nbr_2,
    thrust::device_vector<float4>& nbr_3,
    thrust::device_vector<float4>& nbr_4
) {
    int num_queries = cloud_query_3d.size();
    {
        dim3 num_threads_per_block(NTHREADS_KNN_SEARCH);
        dim3 num_blocks_per_grid((num_queries + NTHREADS_KNN_SEARCH - 1) / NTHREADS_KNN_SEARCH);
        kernel_5nn_search<KEY_BUCKET_SIZE, POINT_BUCKET_SIZE><<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_queries,
            resolution,
            mod,
            thrust::raw_pointer_cast(&(cloud_query_3d[0])),
            thrust::raw_pointer_cast(&(dev_from_hash_to_hash_idx[0])),
            thrust::raw_pointer_cast(&(dev_key_bucket_key[0])),
            thrust::raw_pointer_cast(&(dev_key_bucket_key_idx[0])),
            thrust::raw_pointer_cast(&(dev_key_bucket_key_num[0])),
            thrust::raw_pointer_cast(&(dev_point_bucket_point[0])),
            thrust::raw_pointer_cast(&(dev_point_bucket_point_num[0])),
            thrust::raw_pointer_cast(&(flag[0])),
            thrust::raw_pointer_cast(&(nbr_0[0])),
            thrust::raw_pointer_cast(&(nbr_1[0])),
            thrust::raw_pointer_cast(&(nbr_2[0])),
            thrust::raw_pointer_cast(&(nbr_3[0])),
            thrust::raw_pointer_cast(&(nbr_4[0]))
        );
    }
}
template struct CUDACloudHashMap< 8, 16>;
template struct CUDACloudHashMap< 8, 32>;
template struct CUDACloudHashMap<16, 16>;
template struct CUDACloudHashMap<16, 32>;

// ----------

__global__ void kernel_compute_key_and_hash(
    int num_points, 
    float resolution, 
    unsigned int mod, 
    const float4* point_,
    GridKey* key_,
    int* hash_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_points) return;

    float4 point = point_[tid];
    GridKey key = {
        int(round(point.x / resolution)),
        int(round(point.y / resolution)),
        int(round(point.z / resolution)),
        0
    };
    int hash = hash_func(key, mod);

    key_[tid] = key;
    hash_[tid] = hash;
}

__global__ void kernel_rearrange_point_and_hash(
    int num_map_points,
    const int* src_idx_,
    const float4* point_src_,
    const int* hash_src_,
    float4* point_,
    int* hash_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_map_points) return;

    point_[tid] = point_src_[src_idx_[tid]];
    hash_[tid]  = hash_src_[src_idx_[tid]];
}

__global__ void kernel_rearrange_unique_by_key(
    int num_unique_keys,
    const int* src_idx_,
    const int2* unique_by_key_point_start_end_src_,
    const GridKey* unique_by_key_key_src_,
    int2* unique_by_key_point_start_end_,
    GridKey* unique_by_key_key_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_unique_keys) return;

    unique_by_key_point_start_end_[tid] = unique_by_key_point_start_end_src_[src_idx_[tid]];
    unique_by_key_key_[tid] = unique_by_key_key_src_[src_idx_[tid]];
}

__global__ void kernel_compute_point_start_end(
    int num_unique_keys,
    int num_map_points,
    const int* point_start_,
    int2* point_start_end_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_unique_keys) return;

    int point_start = point_start_[tid];
    int point_end = (tid < num_unique_keys - 1) ? (point_start_[tid + 1]) : num_map_points;
    point_start_end_[tid] = {point_start, point_end};
}

__global__ void kernel_compute_key_start_end(
    int num_unique_hash,
    int num_unique_keys,
    const int* key_start_,
    int2* key_start_end_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_unique_hash) return;

    int key_start = key_start_[tid];
    int key_end = (tid < num_unique_hash - 1) ? (key_start_[tid + 1]) : num_unique_keys;
    key_start_end_[tid] = {key_start, key_end};
}

template<int KEY_BUCKET_SIZE>
__global__ void kernel_insert_key_to_key_bucket_when_map_is_empty(
    int num_unique_hash,
    const int* unique_by_hash_hash_,
    const int* unique_by_hash_hash_idx_,
    const int2* unique_by_hash_key_start_end_,
    const GridKey* unique_by_key_key_,
    int* unique_by_key_key_idx_,
    GridKey* key_bucket_key_,
    int* key_bucket_key_idx_,
    int* key_bucket_key_num_,
    int* from_hash_to_hash_idx_,
    int* key_overflow_warning
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_unique_hash) return;

    int hash_idx = unique_by_hash_hash_idx_[tid];

    int2 key_start_end = unique_by_hash_key_start_end_[tid];
    int& key_start     = key_start_end.x;
    int& key_end       = key_start_end.y;
    int  num_keys      = min(key_end - key_start, KEY_BUCKET_SIZE);

    key_bucket_key_num_[hash_idx] = num_keys;

    for (int i = 0; i < KEY_BUCKET_SIZE; i++) {
        int bucket_idx = hash_idx * KEY_BUCKET_SIZE + i;
        key_bucket_key_[bucket_idx] = i < num_keys ? unique_by_key_key_[key_start + i] : GridKey{0, 0, 0, 1};
    }
    for (int i = 0; i < KEY_BUCKET_SIZE / 4; i++) {
        int bucket_idx = hash_idx * KEY_BUCKET_SIZE + i * 4;
        int4 temp = {
            i * 4 + 0 < num_keys ? key_start + i * 4 + 0 : -1, 
            i * 4 + 1 < num_keys ? key_start + i * 4 + 1 : -1, 
            i * 4 + 2 < num_keys ? key_start + i * 4 + 2 : -1, 
            i * 4 + 3 < num_keys ? key_start + i * 4 + 3 : -1
        };
        *((int4*)(&(key_bucket_key_idx_[bucket_idx]))) = temp;
    }
    for (int i = KEY_BUCKET_SIZE; i < (key_end - key_start); i++) {
        unique_by_key_key_idx_[key_start + i] = -1;
    }

    if (num_keys >= KEY_BUCKET_SIZE * 3 / 4) {
        atomicMax(key_overflow_warning, num_keys);
    }
}
// template
// __global__ void kernel_insert_key_to_key_bucket_when_map_is_empty<MAX_NUM_KEYS_PER_HASH>(
//     int num_unique_hash,
//     const int* unique_by_hash_hash_,
//     const int* unique_by_hash_hash_idx_,
//     const int2* unique_by_hash_key_start_end_,
//     const GridKey* unique_by_key_key_,
//     int* unique_by_key_key_idx_,
//     GridKey* key_bucket_key_,
//     int* key_bucket_key_idx_,
//     int* key_bucket_key_num_,
//     int* from_hash_to_hash_idx_,
//     int* key_overflow_warning
// );

template<int KEY_BUCKET_SIZE>
__global__ void kernel_insert_key_to_key_bucket_when_map_is_not_empty(
    int num_unique_hash,
    int* num_hash_,
    int* num_keys_,
    const int* unique_by_hash_hash_,
    const int2* unique_by_hash_key_start_end_,
    const GridKey* unique_by_key_key_,
    int* unique_by_key_key_idx_,
    int* from_hash_to_hash_idx_,
    GridKey* key_bucket_key_,
    int* key_bucket_key_idx_,
    int* key_bucket_key_num_,
    int* key_overflow_warning
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_unique_hash) {
        return;
    }

    __shared__ int num_new_hash;
    __shared__ int new_hash_idx_start;
    __shared__ int num_new_keys;
    __shared__ int new_key_idx_start;
    if (threadIdx.x == 0) {
        num_new_hash = 0;
        new_hash_idx_start = 0;
        num_new_keys = 0;
        new_key_idx_start = 0;
    }
    __syncthreads();

    int hash = unique_by_hash_hash_[tid];
    int hash_idx = from_hash_to_hash_idx_[hash];

    int hash_idx_local = -1;
    if (hash_idx == -1) {
        hash_idx_local = atomicAdd(&num_new_hash, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0 && num_new_hash != 0) {
        new_hash_idx_start = atomicAdd(num_hash_, num_new_hash);
    }
    __syncthreads();

    if (hash_idx_local != -1) { // this is a new hash
        hash_idx = new_hash_idx_start + hash_idx_local;
        from_hash_to_hash_idx_[hash] = hash_idx;
    }
    __syncthreads();

    GridKey key_in_this_hash[KEY_BUCKET_SIZE];          // content in key bucket
    int key_idx_in_this_hash[KEY_BUCKET_SIZE];          // content in key bucket
    int key_num_in_this_hash = key_bucket_key_num_[hash_idx]; // content in key bucket
    for (int i = 0; i < KEY_BUCKET_SIZE; i++) {         // content in key bucket
        key_in_this_hash    [i] = i < key_num_in_this_hash ? key_bucket_key_    [hash_idx * KEY_BUCKET_SIZE + i] : GridKey{0, 0, 0, 1};
        key_idx_in_this_hash[i] = i < key_num_in_this_hash ? key_bucket_key_idx_[hash_idx * KEY_BUCKET_SIZE + i] :                  -1;
    };

    int2 key_start_end = unique_by_hash_key_start_end_[tid];
    int& key_start     = key_start_end.x;
    int& key_end       = key_start_end.y;
    int num_keys       = min(key_end - key_start, KEY_BUCKET_SIZE);
    GridKey key_to_insert[KEY_BUCKET_SIZE];     // content in unique_by_key
    int flag_new_key[KEY_BUCKET_SIZE];          // content in unique_by_key
    int key_idx_local[KEY_BUCKET_SIZE];         // content in unique_by_key
    for (int i = 0; i < KEY_BUCKET_SIZE; i++) { // content in unique_by_key
        key_idx_local[i] = -1;  // new key's local idx
        key_to_insert[i] = i < num_keys ? unique_by_key_key_[key_start + i] : GridKey{0, 0, 0, 0};
        flag_new_key[i]  = i < num_keys ? 0 : -1;
        //  0 means old key
        // -1 means out of boundry
    }
    for (int i = key_start + KEY_BUCKET_SIZE; i < key_end; i++) {
        unique_by_key_key_idx_[i] = -1;
    }

    // for each key in unique_by_key
    // if it's a old key, write the old key's idx to unique_by_key_key_idx_
    // if it's a new key, flag_new_key[i] set to 1
    for (int i = 0; i < num_keys; i++) {
        GridKey key = key_to_insert[i];
        int key_idx_global = -1;
        for (int j = 0; j < key_num_in_this_hash; j++) {
            if (key == key_in_this_hash[j]) {
                key_idx_global = key_idx_in_this_hash[j];
                break;
            }
        }
        if (key_idx_global == -1) { // it's a new key
            flag_new_key[i] = 1;
        } else { // it's a old key
            unique_by_key_key_idx_[key_start + i] = key_idx_global;
        }
    }

    int count = 0;
    for (int i = 0; i < num_keys; i++) {
        if (flag_new_key[i] == 1) {
            count++;
            if (count > (KEY_BUCKET_SIZE - key_num_in_this_hash)) {
                flag_new_key[i] == 2; // new key, but there is no room for it in this hash
            }
        }
    }

    // for each key in unique_by_key
    // if it's a new key, get this key's local idx
    for (int i = 0; i < num_keys; i++) {
        if (flag_new_key[i] == 1) {
            key_idx_local[i] = atomicAdd(&num_new_keys, 1);
        }
    }
    __syncthreads();

    // starting point of new key's idx
    if (threadIdx.x == 0 && num_new_keys != 0) {
        new_key_idx_start = atomicAdd(num_keys_, num_new_keys);
    }
    __syncthreads();

    // for each key in unique_by_key,
    // deal with new keys
    for (int i = 0; i < num_keys; i++) {
        if (flag_new_key[i] == 0) {
            continue;
        } else if (flag_new_key[i] == 1) {
            int key_idx_global = new_key_idx_start + key_idx_local[i];
            GridKey key = key_to_insert[i];
            key_in_this_hash    [key_num_in_this_hash] = key;
            key_idx_in_this_hash[key_num_in_this_hash] = key_idx_global;
            key_num_in_this_hash = key_num_in_this_hash + 1;
            unique_by_key_key_idx_[key_start + i] = key_idx_global;
        } else { // flag_new_key[i] == 2
            unique_by_key_key_idx_[key_start + i] = -1;
        }
    }

    key_bucket_key_num_[hash_idx] = key_num_in_this_hash;
    for (int i = 0; i < KEY_BUCKET_SIZE; i++) {
        key_bucket_key_[hash_idx * KEY_BUCKET_SIZE + i] = key_in_this_hash[i];
    };
    for (int i = 0; i < KEY_BUCKET_SIZE / 4; i++) {
        int4 temp = {
            key_idx_in_this_hash[i * 4 + 0],
            key_idx_in_this_hash[i * 4 + 1],
            key_idx_in_this_hash[i * 4 + 2],
            key_idx_in_this_hash[i * 4 + 3]
        };
        *((int4*)(&(key_bucket_key_idx_[hash_idx * KEY_BUCKET_SIZE + i * 4]))) = temp;
    };

    if (key_num_in_this_hash >= KEY_BUCKET_SIZE * 3 / 4) {
        atomicMax(key_overflow_warning, num_keys);
    }
}
// template
// __global__ void kernel_insert_key_to_key_bucket_when_map_is_not_empty<MAX_NUM_KEYS_PER_HASH>(
//     int num_unique_hash,
//     int* num_hash_,
//     int* num_keys_,
//     const int* unique_by_hash_hash_,
//     const int2* unique_by_hash_key_start_end_,
//     const GridKey* unique_by_key_key_,
//     int* unique_by_key_key_idx_,
//     int* from_hash_to_hash_idx_,
//     GridKey* key_bucket_key_,
//     int* key_bucket_key_idx_,
//     int* key_bucket_key_num_,
//     int* key_overflow_warning
// );

template<int POINT_BUCKET_SIZE>
__global__ void kernel_insert_point_to_point_bucket_when_map_is_empty(
    int num_unique_keys,
    const int* unique_by_key_key_idx_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_,
    float4* point_bucket_point_,
    int* point_bucket_point_num_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_unique_keys) return;

    int key_idx = unique_by_key_key_idx_[tid];
    if (key_idx == -1) return;

    int2 point_start_end = unique_by_key_point_start_end_[tid];
    int& point_start     = point_start_end.x;
    int& point_end       = point_start_end.y;
    int  num_points      = min(point_end - point_start, POINT_BUCKET_SIZE);

    point_bucket_point_num_[key_idx] = num_points;
    for (int i = 0; i < POINT_BUCKET_SIZE; i++) {
        int bucket_idx = key_idx * POINT_BUCKET_SIZE + i;
        point_bucket_point_[bucket_idx] = i < num_points ? point_[point_start + i] : float4{0, 0, 0, 0};
    }
}
// template
// __global__ void kernel_insert_point_to_point_bucket_when_map_is_empty<MAX_NUM_POINTS_PER_KEY>(
//     int num_unique_keys,
//     const int* unique_by_key_key_idx_,
//     const int2* unique_by_key_point_start_end_,
//     const float4* point_,
//     float4* point_bucket_point_,
//     int* point_bucket_point_num_
// );

template<int POINT_BUCKET_SIZE>
__global__ void kernel_insert_point_to_point_bucket_when_map_is_not_empty(
    int num_unique_keys,
    const int* unique_by_key_key_idx_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_,
    float4* point_bucket_point_,
    int* point_bucket_point_num_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_unique_keys) return;

    int key_idx = unique_by_key_key_idx_[tid];
    if (key_idx == -1) return;

    int num_points_in_bucket_old = point_bucket_point_num_[key_idx];

    int2 point_start_end = unique_by_key_point_start_end_[tid];
    int& point_start     = point_start_end.x;
    int& point_end       = point_start_end.y;
    int  num_points      = min(point_end - point_start, POINT_BUCKET_SIZE - num_points_in_bucket_old);

    int num_points_in_bucket_new = num_points_in_bucket_old + num_points;

    point_bucket_point_num_[key_idx] = num_points_in_bucket_new;
    for (int i = num_points_in_bucket_old; i < POINT_BUCKET_SIZE; i++) {
        point_bucket_point_[key_idx * POINT_BUCKET_SIZE + i] = \
        i < num_points_in_bucket_new ? point_[point_start + (i - num_points_in_bucket_old)] : float4{0, 0, 0, 0};
    }
}
// template
// __global__ void kernel_insert_point_to_point_bucket_when_map_is_not_empty<MAX_NUM_POINTS_PER_KEY>(
//     int num_unique_keys,
//     const int* unique_by_key_key_idx_,
//     const int2* unique_by_key_point_start_end_,
//     const float4* point_,
//     float4* point_bucket_point_,
//     int* point_bucket_point_num_
// );

template<int KEY_BUCKET_SIZE, int POINT_BUCKET_SIZE>
__global__ void kernel_5nn_search(
    int num_queries,
    float resolution,
    unsigned int mod,
    const float4* query_,
    const int* from_hash_to_hash_idx_,
    const GridKey* key_bucket_key_,
    const int* key_bucket_key_idx_,
    const int* key_bucket_key_num_,
    const float4* point_bucket_point_,
    const int* point_bucket_point_num_,
    char* flag_,
    float4* nbr_0_,
    float4* nbr_1_,
    float4* nbr_2_,
    float4* nbr_3_,
    float4* nbr_4_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_queries) return;

    float4 query = query_[tid];

    GridKey grid_key_[27];
    grid_key_[0]  = {
        int(round(query.x / resolution)), 
        int(round(query.y / resolution)), 
        int(round(query.z / resolution)), 
        0
    };
    grid_key_[1]  = grid_key_[0] + GridKey{-1,  0,  0, 0};
    grid_key_[2]  = grid_key_[0] + GridKey{ 1,  0,  0, 0};
    grid_key_[3]  = grid_key_[0] + GridKey{ 0,  1,  0, 0};
    grid_key_[4]  = grid_key_[0] + GridKey{ 0, -1,  0, 0};
    grid_key_[5]  = grid_key_[0] + GridKey{ 0,  0, -1, 0};
    grid_key_[6]  = grid_key_[0] + GridKey{ 0,  0,  1, 0};
    grid_key_[7]  = grid_key_[0] + GridKey{ 1,  1,  0, 0};
    grid_key_[8]  = grid_key_[0] + GridKey{-1,  1,  0, 0};
    grid_key_[9]  = grid_key_[0] + GridKey{ 1, -1,  0, 0};
    grid_key_[10] = grid_key_[0] + GridKey{-1, -1,  0, 0};
    grid_key_[11] = grid_key_[0] + GridKey{ 1,  0,  1, 0};
    grid_key_[12] = grid_key_[0] + GridKey{-1,  0,  1, 0};
    grid_key_[13] = grid_key_[0] + GridKey{ 1,  0, -1, 0};
    grid_key_[14] = grid_key_[0] + GridKey{-1,  0, -1, 0};
    grid_key_[15] = grid_key_[0] + GridKey{ 0,  1,  1, 0};
    grid_key_[16] = grid_key_[0] + GridKey{ 0, -1,  1, 0};
    grid_key_[17] = grid_key_[0] + GridKey{ 0,  1, -1, 0};
    grid_key_[18] = grid_key_[0] + GridKey{ 0, -1, -1, 0};
    grid_key_[19] = grid_key_[0] + GridKey{ 1,  1,  1, 0};
    grid_key_[20] = grid_key_[0] + GridKey{-1,  1,  1, 0};
    grid_key_[21] = grid_key_[0] + GridKey{ 1, -1,  1, 0};
    grid_key_[22] = grid_key_[0] + GridKey{ 1,  1, -1, 0};
    grid_key_[23] = grid_key_[0] + GridKey{-1, -1,  1, 0};
    grid_key_[24] = grid_key_[0] + GridKey{-1,  1, -1, 0};
    grid_key_[25] = grid_key_[0] + GridKey{ 1, -1, -1, 0};
    grid_key_[26] = grid_key_[0] + GridKey{-1, -1, -1, 0};

    float worst_distance = __FLT_MAX__;
    float distances[5] = {worst_distance, worst_distance, worst_distance, worst_distance, worst_distance};
    int neighbors[5] = {-1, -1, -1, -1, -1};
    int num_neighbors = 0;

    for (const auto& grid_key : grid_key_) {
        int hash = hash_func(grid_key, mod);
        int hash_idx = from_hash_to_hash_idx_[hash];
        int num_keys = min(key_bucket_key_num_[hash_idx], KEY_BUCKET_SIZE);
        int key_idx = -1;
        int k;
        for (k = 0; k < num_keys; k++) {
            GridKey temp_grid_key = key_bucket_key_[hash_idx * KEY_BUCKET_SIZE + k];
            key_idx = key_bucket_key_idx_[hash_idx * KEY_BUCKET_SIZE + k];
            if (temp_grid_key == grid_key) {
                break;
            }
        }
        if (k >= num_keys) {
            continue;
        }

        int num_points = point_bucket_point_num_[key_idx];
        for (int i = 0; i < num_points; i++) {
            float4 neighbor = point_bucket_point_[key_idx * POINT_BUCKET_SIZE + i];
            float distance = powf((query.x - neighbor.x), 2) + powf((query.y - neighbor.y), 2) + powf((query.z - neighbor.z), 2);
            if (distance < worst_distance) {
                if (num_neighbors < 5) num_neighbors++;
                int j = 0;
                for (j = 0; j < 5; j++) {
                    if (distance < distances[j]) {
                        break;
                    }
                }
                if (j < 5) {
                    for (int m = 5 - 1; m > j; m--) {
                        distances[m] = distances[m - 1];
                        neighbors[m] = neighbors[m - 1];
                    }
                    distances[j] = distance;
                    neighbors[j] = key_idx * POINT_BUCKET_SIZE + i;
                }
            }
            worst_distance = distances[5 - 1];
        }
    }

    float4 nbr;

    if (neighbors[0] != -1) {
        nbr = point_bucket_point_[ neighbors[0] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_0_[tid] = nbr;

    if (neighbors[1] != -1) {
        nbr = point_bucket_point_[ neighbors[1] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_1_[tid] = nbr;

    if (neighbors[2] != -1) {
        nbr = point_bucket_point_[ neighbors[2] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_2_[tid] = nbr;

    if (neighbors[3] != -1) {
        nbr = point_bucket_point_[ neighbors[3] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_3_[tid] = nbr;

    if (neighbors[4] != -1) {
        nbr = point_bucket_point_[ neighbors[4] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_4_[tid] = nbr;

    flag_[tid] = distances[4] < 1.0 ? char(1) : char(0);
}
// template
// __global__ void kernel_5nn_search<MAX_NUM_KEYS_PER_HASH, MAX_NUM_POINTS_PER_KEY>(
//     int num_queries,
//     float resolution,
//     unsigned int mod,
//     const float4* query_,
//     const int* from_hash_to_hash_idx_,
//     const GridKey* key_bucket_key_,
//     const int* key_bucket_key_idx_,
//     const int* key_bucket_key_num_,
//     const float4* point_bucket_point_,
//     const int* point_bucket_point_num_,
//     char* flag_,
//     float4* nbr_0_,
//     float4* nbr_1_,
//     float4* nbr_2_,
//     float4* nbr_3_,
//     float4* nbr_4_
// );
