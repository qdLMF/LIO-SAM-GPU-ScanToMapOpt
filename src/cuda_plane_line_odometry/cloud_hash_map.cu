//
// Created by lmf on 23-9-7.
//

#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "cloud_hash_map.cuh"


#define RESOLUTION 0.5
#define MOD 100000

#define KEY_BUCKET_SIZE (MAX_NUM_KEYS_PER_HASH * (MOD / 10) * 16)
#define POINT_BUCKET_SIZE (MAX_NUM_POINTS_PER_KEY * (MOD / 10) * 16)

#define NTHREADS_BUILD_MAP 256
#define NTHREADS_KNN_SEARCH 256

// ----------

__device__ GridKey key_func(const float4& point) {
    static const float inv_resolution = float(1.0) / RESOLUTION;
    return {
        int(round(point.x * inv_resolution)),
        int(round(point.y * inv_resolution)),
        int(round(point.z * inv_resolution)),
        0
    };
}

__device__ int hash_func(const GridKey& key) {
    return int(size_t(((key.x) * 73856093) ^ ((key.y) * 471943) ^ ((key.z) * 83492791)) % MOD);
}

// ----------

__global__ void kernel_1(
    int num_points,
    const float4* point_,
    GridKey* key_,
    int* hash_
);

__global__ void kernel_4(
    int num_unique_keys,
    int num_map_points,
    const int* point_start_,
    int2* point_start_end_
);

__global__ void kernel_4_1(
    int num_unique_hash,
    int num_unique_keys,
    const int* key_start_,
    int2* key_start_end_
);

__global__ void kernel_5(
    int num_unique_hash,
    const int* unique_by_hash_hash_idx_,
    const int2* unique_by_hash_key_start_end_,
    const GridKey* unique_by_key_key_,
    GridKey* key_bucket_key_,
    int* key_bucket_key_idx_
);

__global__ void kernel_6(
    int num_unique_keys,
    const int* unique_by_key_key_idx_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_,
    float4* point_bucket_point_
);

__global__ void knn_search(
    int num_queries,
    const float4* query_,
    const int* from_hash_to_hash_idx_,
    const GridKey* key_bucket_key_,
    const int* key_bucket_key_idx_,
    const int2* unique_by_hash_key_start_end_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_bucket_point_,
    char* flag,
    float4* nbr_0,
    float4* nbr_1,
    float4* nbr_2,
    float4* nbr_3,
    float4* nbr_4
);

// ----------

CUDACloudHashMap::CUDACloudHashMap(unsigned int max_map_size_) : max_map_size(max_map_size_) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    BuildMapPre();
}

CUDACloudHashMap::~CUDACloudHashMap() {
    Sync();
    cudaStreamDestroy(stream);
}

void CUDACloudHashMap::Sync() {
    cudaStreamSynchronize(stream);
}

void CUDACloudHashMap::BuildMapPre() {
    dev_key_bucket_key.resize(KEY_BUCKET_SIZE, {0, 0, 0, 1});     // 100000
    dev_key_bucket_key_idx.resize(KEY_BUCKET_SIZE, -1);
    dev_point_bucket_point.resize(POINT_BUCKET_SIZE, make_float4(0.0, 0.0, 0.0, 0.0));

    dev_from_hash_to_hash_idx.resize(MOD, -1);
}

void CUDACloudHashMap::BuildMap(const thrust::host_vector<float4> &cloud_map_3d) {
    dev_point_3d = cloud_map_3d;

    dev_key.resize(max_map_size, {0, 0, 0, 1});
    dev_hash.resize(max_map_size, -1);

    dev_unique_by_key_key.resize(max_map_size, {0, 0, 0, 1});
    dev_unique_by_key_hash.resize(max_map_size, -1);
    dev_unique_by_key_point_start.resize(max_map_size, -1);
    dev_unique_by_key_point_start_end.resize(max_map_size, make_int2(-1, -1));
    dev_unique_by_key_key_idx.resize(max_map_size, -1);

    dev_unique_by_hash_hash.resize(max_map_size, -1);
    dev_unique_by_hash_key_start.resize(max_map_size, -1);
    dev_unique_by_hash_key_start_end.resize(max_map_size, make_int2(-1, -1));
    dev_unique_by_hash_hash_idx.resize(max_map_size, -1);

    dev_key_backup.resize(max_map_size, {0, 0, 0, 1});
    dev_hash_backup.resize(max_map_size, -1);

    thrust::counting_iterator<int> c_iter_1(0);
    thrust::counting_iterator<int> c_iter_2(0);

    BuildMapPre();

    int num_map_points = 0;
    int num_unique_keys = 0;
    int num_unique_hash = 0;
    int num_queries = 0;

    int num_full_hash_slots = 0;
    int num_empty_hash_slots = 0;

    // ------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------
    // start : build map v1

    // auto now = std::chrono::system_clock::now();

    // ------------------------------------------------------------------------------------------------------------------------
    num_map_points = dev_point_3d.size();
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_map_points + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_1<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_map_points,
            thrust::raw_pointer_cast(&(dev_point_3d[0])),
            thrust::raw_pointer_cast(&(dev_key[0])),
            thrust::raw_pointer_cast(&(dev_hash[0]))
        );
    }
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // sort by key
    // dev_key_backup.resize(dev_key.size(), {0, 0, 0, 1});
    thrust::copy(thrust::cuda::par.on(stream), dev_key.begin(), dev_key.end(), dev_key_backup.begin());
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        dev_key.begin(),
        dev_key.begin() + num_map_points,
        dev_point_3d.begin()
    );
    thrust::copy(thrust::cuda::par.on(stream), dev_key_backup.begin(), dev_key_backup.end(), dev_key.begin());
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        dev_key.begin(),
        dev_key.begin() + num_map_points,
        dev_hash.begin()
    );
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // unique by key
    num_unique_keys = \
    thrust::unique_by_key_copy(
        thrust::cuda::par.on(stream),
        dev_key.begin(),
        dev_key.begin() + num_map_points,
        thrust::counting_iterator<int>(0),
        dev_unique_by_key_key.begin(),
        dev_unique_by_key_point_start.begin()
    ).first - dev_unique_by_key_key.begin();
    num_unique_keys = \
    thrust::unique_by_key_copy(
        thrust::cuda::par.on(stream),
        dev_key.begin(),
        dev_key.begin() + num_map_points,
        dev_hash.begin(),
        dev_unique_by_key_key.begin(),
        dev_unique_by_key_hash.begin()
    ).first - dev_unique_by_key_key.begin();

    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_keys + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_4<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_unique_keys,
            num_map_points,
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end[0]))
        );
    }
    // and then sort by hash
    // dev_hash_backup.resize(num_unique_keys, -1);
    thrust::copy(thrust::cuda::par.on(stream), dev_unique_by_key_hash.begin(), dev_unique_by_key_hash.begin() + num_unique_keys, dev_hash_backup.begin());
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        dev_unique_by_key_hash.begin(),
        dev_unique_by_key_hash.begin() + num_unique_keys,
        dev_unique_by_key_point_start_end.begin()
    );
    thrust::copy(thrust::cuda::par.on(stream), dev_hash_backup.begin(), dev_hash_backup.end(), dev_unique_by_key_hash.begin());
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        dev_unique_by_key_hash.begin(),
        dev_unique_by_key_hash.begin() + num_unique_keys,
        dev_unique_by_key_key.begin()
    );

    thrust::copy(thrust::cuda::par.on(stream), c_iter_1, c_iter_1 + num_unique_keys, dev_unique_by_key_key_idx.begin());
    // output :
    //     thrust::device_vector<GridKey> dev_unique_by_key_key;
    //     thrust::device_vector<int> dev_unique_by_key_hash;
    //     thrust::device_vector<int> dev_unique_by_key_point_start;  // temp
    //     thrust::device_vector<int2> dev_unique_by_key_point_start_end;
    //     thrust::device_vector<int> dev_unique_by_key_key_idx;
    // all above are unique by key and then sorted by hash, hash values are not unique
    // ------------------------------------------------------------------------------------------------------------------------


    // ------------------------------------------------------------------------------------------------------------------------
    // unique by hash
    num_unique_hash = \
    thrust::unique_by_key_copy(
        thrust::cuda::par.on(stream),
        dev_unique_by_key_hash.begin(),
        dev_unique_by_key_hash.begin() + num_unique_keys,
        dev_unique_by_key_key_idx.begin(),
        dev_unique_by_hash_hash.begin(),
        dev_unique_by_hash_key_start.begin()
    ).first - dev_unique_by_hash_hash.begin();

    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_hash + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_4_1<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_unique_hash,
            num_unique_keys,
            thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start_end[0]))    // key_idx in dev_unique_by_key_key
        );
    }

    thrust::copy(thrust::cuda::par.on(stream), c_iter_2, c_iter_2 + num_unique_hash, dev_unique_by_hash_hash_idx.begin());
    // ------------------------------------------------------------------------------------------------------------------------

    // update key bucket
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_hash + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_5<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_unique_hash,
            thrust::raw_pointer_cast(&(dev_unique_by_hash_hash_idx[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start_end[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_key[0])),
            thrust::raw_pointer_cast(&(dev_key_bucket_key[0])),
            thrust::raw_pointer_cast(&(dev_key_bucket_key_idx[0]))
        );
    }

#if 0
    num_full_hash_slots = 0;
    num_empty_hash_slots = 0;
    for(int i = 0; i < num_unique_hash; i++) {
        int hash_idx = dev_unique_by_hash_hash_idx[i];
        int hash = dev_unique_by_hash_hash[i];
//        printf("%d , %d : \n", hash_idx, hash);
        int count = 0;
        for(int j = 0; j < MAX_NUM_KEYS_PER_HASH; j++) {
            GridKey key = dev_key_bucket_key[i * MAX_NUM_KEYS_PER_HASH + j];
            int key_idx = dev_key_bucket_key_idx[i * MAX_NUM_KEYS_PER_HASH + j];
//            printf("\t( %d , %d , %d , %d ) , %d \n", key.x, key.y, key.z , key.w, key_idx);
            if(key_idx != -1) {
                count++;
            }
        }
        if(count == MAX_NUM_KEYS_PER_HASH) {
            num_full_hash_slots++;
        }
        if(count == 0) {
            num_empty_hash_slots++;
        }
    }
    std::cout << "dev_unique_by_hash_hash.size() : " << dev_unique_by_hash_hash.size() << std::endl;
    std::cout << "num_unique_hash : " << num_unique_hash << std::endl;
    std::cout << "num_unique_keys : " << num_unique_keys << std::endl;
    std::cout << "num_full_hash_slots  : " << num_full_hash_slots << std::endl;
    std::cout << "num_empty_hash_slots : " << num_empty_hash_slots << std::endl;
#endif

    // dev_point_bucket_point.resize(MAX_NUM_POINTS_PER_KEY * num_unique_keys * 4, make_float4(0.0, 0.0, 0.0, 0.0));
    {
        dim3 num_threads_per_block(NTHREADS_BUILD_MAP);
        dim3 num_blocks_per_grid((num_unique_keys + NTHREADS_BUILD_MAP - 1) / NTHREADS_BUILD_MAP);
        kernel_6<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_unique_keys,
            thrust::raw_pointer_cast(&(dev_unique_by_key_key_idx[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end[0])),
            thrust::raw_pointer_cast(&(dev_point_3d[0])),
            thrust::raw_pointer_cast(&(dev_point_bucket_point[0]))
        );
    }

    thrust::scatter(
        thrust::cuda::par.on(stream),
        dev_unique_by_hash_hash_idx.begin(),
        dev_unique_by_hash_hash_idx.begin() + num_unique_hash,
        dev_unique_by_hash_hash.begin(),
        dev_from_hash_to_hash_idx.begin()
    );

    // auto dur = std::chrono::system_clock::now() - now;
    // std::cout << "duration ms : " << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() / 1000.0 << std::endl;

    // end : build map v1
    // ------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------
}

void CUDACloudHashMap::ClearAfterBuildMap() {
    // ------------------------------------------------------------------------------------------------------------------------
    // start : clean up temp vectors

    dev_key.clear();
    dev_hash.clear();
    dev_key_backup.clear();
    dev_hash_backup.clear();
    dev_unique_by_key_key.clear();
    dev_unique_by_key_hash.clear();
    dev_unique_by_key_point_start.clear();
    dev_unique_by_key_key_idx.clear();
    // dev_point_buckets.clear();
    dev_unique_by_hash_hash.clear();
    dev_unique_by_hash_key_start.clear();
    dev_unique_by_hash_hash_idx.clear();

    // end : clean up temp vectors
    // ------------------------------------------------------------------------------------------------------------------------
}

void CUDACloudHashMap::Query(
    const thrust::device_vector<float4>& cloud_query_3d,
    thrust::device_vector<char>& flag,
    thrust::device_vector<float4>& nbr_0,
    thrust::device_vector<float4>& nbr_1,
    thrust::device_vector<float4>& nbr_2,
    thrust::device_vector<float4>& nbr_3,
    thrust::device_vector<float4>& nbr_4
) {
    // auto now = std::chrono::system_clock::now();

    int num_queries = cloud_query_3d.size();
    {
        dim3 num_threads_per_block(NTHREADS_KNN_SEARCH);
        dim3 num_blocks_per_grid((num_queries + NTHREADS_KNN_SEARCH - 1) / NTHREADS_KNN_SEARCH);
        knn_search<<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(
            num_queries,
            thrust::raw_pointer_cast(&(cloud_query_3d[0])),
            thrust::raw_pointer_cast(&(dev_from_hash_to_hash_idx[0])),
            thrust::raw_pointer_cast(&(dev_key_bucket_key[0])),
            thrust::raw_pointer_cast(&(dev_key_bucket_key_idx[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_hash_key_start_end[0])),
            thrust::raw_pointer_cast(&(dev_unique_by_key_point_start_end[0])),
            thrust::raw_pointer_cast(&(dev_point_bucket_point[0])),
            thrust::raw_pointer_cast(&(flag[0])),
            thrust::raw_pointer_cast(&(nbr_0[0])),
            thrust::raw_pointer_cast(&(nbr_1[0])),
            thrust::raw_pointer_cast(&(nbr_2[0])),
            thrust::raw_pointer_cast(&(nbr_3[0])),
            thrust::raw_pointer_cast(&(nbr_4[0]))
        );
    }

    // auto dur = std::chrono::system_clock::now() - now;
    // std::cout << "duration ms : " << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() / 1000.0 << std::endl;
}

// ----------

__global__ void kernel_1(
    int num_points,
    const float4* point_,
    GridKey* key_,
    int* hash_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= num_points) 
        return;

    float resolution = RESOLUTION;
    float inv_resolution = float(1.0) / resolution;

    float4 point = point_[tid];
    GridKey key = {
        int(round(point.x * inv_resolution)),
        int(round(point.y * inv_resolution)),
        int(round(point.z * inv_resolution)),
        0
    };
    int hash = hash_func(key);

    key_[tid] = key;
    hash_[tid] = hash;
}

__global__ void kernel_4(
    int num_unique_keys,
    int num_map_points,
    const int* point_start_,
    int2* point_start_end_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= num_unique_keys) 
        return;

    int point_start = point_start_[tid];
    int point_end = (tid < num_unique_keys - 1) ? (point_start_[tid + 1]) : num_map_points;
    point_start_end_[tid] = {point_start, point_end};
}

__global__ void kernel_4_1(
    int num_unique_hash,
    int num_unique_keys,
    const int* key_start_,
    int2* key_start_end_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= num_unique_hash) 
        return;

    int key_start = key_start_[tid];
    int key_end = (tid < num_unique_hash - 1) ? (key_start_[tid + 1]) : num_unique_keys;
    key_start_end_[tid] = {key_start, key_end};
}

__global__ void kernel_5(
    int num_unique_hash,
    const int* unique_by_hash_hash_idx_,
    const int2* unique_by_hash_key_start_end_,
    const GridKey* unique_by_key_key_,
    GridKey* key_bucket_key_,
    int* key_bucket_key_idx_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= num_unique_hash) return;

    int hash_idx = unique_by_hash_hash_idx_[tid];

    int2 key_start_end = unique_by_hash_key_start_end_[tid];
    int key_start = key_start_end.x;
    int key_end = key_start_end.y;
    int num_keys = min(key_start_end.y - key_start_end.x, MAX_NUM_KEYS_PER_HASH);
    // for(int key_idx = key_start; key_idx < key_end; key_idx++) {
    for(int i = 0; i < num_keys; i++) {
        int key_idx = key_start + i;
        GridKey key = unique_by_key_key_[key_idx];
        int bucket_idx = hash_idx * MAX_NUM_KEYS_PER_HASH + (key_idx - key_start);
        key_bucket_key_[bucket_idx] = key;
        key_bucket_key_idx_[bucket_idx] = key_idx;
    }
}

__global__ void kernel_6(
    int num_unique_keys,
    const int* unique_by_key_key_idx_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_,
    float4* point_bucket_point_
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= num_unique_keys) return;

    int key_idx = unique_by_key_key_idx_[tid];

    int2 point_start_end = unique_by_key_point_start_end_[tid];
    int point_start = point_start_end.x;
    int point_end = point_start_end.y;
    int num_points = min(point_start_end.y - point_start_end.x, MAX_NUM_POINTS_PER_KEY);
    // for(int point_idx = point_start; point_idx < point_end; point_idx++) {
    for(int i = 0; i < num_points; i++) {
        int point_idx = point_start + i;
        float4 point = point_[point_idx];
        int bucket_idx = key_idx * MAX_NUM_POINTS_PER_KEY + (point_idx - point_start);
        point_bucket_point_[bucket_idx] = point;
    }
}

__global__ void knn_search(
    int num_queries,
    const float4* query_,
    const int* from_hash_to_hash_idx_,
    const GridKey* key_bucket_key_,
    const int* key_bucket_key_idx_,
    const int2* unique_by_hash_key_start_end_,
    const int2* unique_by_key_point_start_end_,
    const float4* point_bucket_point_,
    char* flag,
    float4* nbr_0,
    float4* nbr_1,
    float4* nbr_2,
    float4* nbr_3,
    float4* nbr_4
) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= num_queries) 
        return;

    float4 query = query_[tid];

    float resolution = RESOLUTION;
    float inv_resolution = float(1.0) / resolution;

    GridKey grid_key_[27];
    grid_key_[0]  = {int(round(query.x * inv_resolution)), int(round(query.y * inv_resolution)), int(round(query.z * inv_resolution)), 0};
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

    float worst_distance = 3.4e+38;
    float distances[5] = {worst_distance, worst_distance, worst_distance, worst_distance, worst_distance};
    int neighbors[5] = {-1, -1, -1, -1, -1};
    int num_neighbors = 0;

    for(const auto& grid_key : grid_key_) {
        int hash = hash_func(grid_key);
        int hash_idx = from_hash_to_hash_idx_[hash];
        int2 key_start_end = unique_by_hash_key_start_end_[hash_idx];
        int num_keys = min(key_start_end.y - key_start_end.x, MAX_NUM_KEYS_PER_HASH);
        int key_idx = -1;
        int k;
        for(k = 0; k < num_keys; k++) {
            GridKey temp_grid_key = key_bucket_key_[hash_idx * MAX_NUM_KEYS_PER_HASH + k];
            key_idx = key_bucket_key_idx_[hash_idx * MAX_NUM_KEYS_PER_HASH + k];
            if(temp_grid_key == grid_key) {
                break;
            }
        }
        if(k >= num_keys) {
            continue;
        }

        int2 point_start_end = unique_by_key_point_start_end_[key_idx];
        int num_points = point_start_end.y - point_start_end.x;
        for(int i = 0; i < num_points; i++) {
            float4 neighbor = point_bucket_point_[key_idx * MAX_NUM_POINTS_PER_KEY + i];
            float distance = powf((query.x - neighbor.x), 2) + powf((query.y - neighbor.y), 2) + powf((query.z - neighbor.z), 2);
            if(distance < worst_distance) {
                if(num_neighbors < 5) num_neighbors++;
                int j = 0;
                for(j = 0; j < 5; j++) {
                    if(distance < distances[j]) {
                        break;
                    }
                }
                if(j < 5) {
                    for(int m = 5 - 1; m > j; m--) {
                        distances[m] = distances[m - 1];
                        neighbors[m] = neighbors[m - 1];
                    }
                    distances[j] = distance;
                    neighbors[j] = key_idx * MAX_NUM_POINTS_PER_KEY + i;
                }
            }
            worst_distance = distances[5 - 1];
        }
    }

    float4 nbr;

    if(neighbors[0] != -1) {
        nbr = point_bucket_point_[ neighbors[0] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_0[tid] = nbr;

    if(neighbors[1] != -1) {
        nbr = point_bucket_point_[ neighbors[1] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_1[tid] = nbr;

    if(neighbors[2] != -1) {
        nbr = point_bucket_point_[ neighbors[2] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_2[tid] = nbr;

    if(neighbors[3] != -1) {
        nbr = point_bucket_point_[ neighbors[3] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_3[tid] = nbr;

    if(neighbors[4] != -1) {
        nbr = point_bucket_point_[ neighbors[4] ];
    } else {
        nbr = {0.0, 0.0, 0.0, 0.0};
    }
    nbr_4[tid] = nbr;

    flag[tid] = distances[4] < 1.0 ? char(1) : char(0);
}

