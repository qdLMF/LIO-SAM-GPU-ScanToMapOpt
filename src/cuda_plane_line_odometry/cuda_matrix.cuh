//
// Created by lmf on 23-9-6.
//

#ifndef LIO_SAM_CUDA_CUDA_MATRIX_CUH
#define LIO_SAM_CUDA_CUDA_MATRIX_CUH

#include <cstdio>
#include <cassert>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>


template<typename T>
struct GPUMemBlock {
private :
    T* device_ptr;
    int max_num_rows;
    int max_num_cols;
    int num_rows;
    int num_cols;
    int leading_dim;
    cudaError_t cuda_status;

public :
    GPUMemBlock() = delete;

    // GPUMemBlock(int nrows, int ncols, int ld) {
    //    assert(nrows <= ld);

    //    num_rows = nrows;
    //    num_cols = ncols;
    //    leading_dim = ld;

    //    device_ptr = nullptr;
    //    cuda_status = cudaMalloc((void**)&device_ptr, sizeof(T) * leading_dim * num_cols);
    //    assert(cudaSuccess == cuda_status);
    //    cuda_status = cudaMemset(device_ptr, 0, sizeof(T) * leading_dim * num_cols);
    //    assert(cudaSuccess == cuda_status);
    // }

    GPUMemBlock(int max_nrows, int max_ncols) {
        max_num_rows = max_nrows;
        max_num_cols = max_ncols;
        leading_dim = max_nrows;

        num_rows = 0;
        num_cols = 0;

        device_ptr = nullptr;
        cuda_status = cudaMalloc((void**)&device_ptr, sizeof(T) * max_num_rows * max_num_cols);
        assert(cudaSuccess == cuda_status);
        cuda_status = cudaMemset(device_ptr, 0, sizeof(T) * max_num_rows * max_num_cols);
        assert(cudaSuccess == cuda_status);
    }

    void resize(int nrows, int ncols) {
        assert(nrows <= max_num_rows);
        assert(ncols <= max_num_cols);

        num_rows = nrows;
        num_cols = ncols;

        cuda_status = cudaMemset(device_ptr, 0, sizeof(T) * max_num_rows * max_num_cols);
        assert(cudaSuccess == cuda_status);
    }

    ~GPUMemBlock() {
        if(device_ptr) {
            cudaFree(device_ptr);
            device_ptr = nullptr;
        }
    }

    int rows() const {
        return num_rows;
    }

    int cols() const {
        return num_cols;
    }

    int ld() const {
        return leading_dim;
    }

    T* GetDevicePtr() const {
        return device_ptr;
    };

    void SetZero() {
        if(device_ptr) {
            cuda_status = cudaMemset(device_ptr, 0, sizeof(T) * max_num_rows * max_num_cols);
            assert(cudaSuccess == cuda_status);
        }
    }
};
template struct GPUMemBlock<float>;


template<typename T>
class CUDAMatrix{
    typedef GPUMemBlock<T> GPUMemBlockType;

public :
    CUDAMatrix() = default;

    // CUDAMatrix(int nrows, int ncols, int ld) {
    //    assert(nrows > 0);
    //    assert(ncols > 0);
    //    assert(ld >= nrows);

    //    matrix_ptr = std::make_shared<GPUMemBlockType>(nrows, ncols, 8192);
    // }

    CUDAMatrix(int max_nrows, int max_ncols) {
        assert(max_nrows > 0);
        assert(max_ncols > 0);

        matrix_ptr = std::make_shared<GPUMemBlockType>(max_nrows, max_ncols);
    }

    void resize(int nrows, int ncols) {
        matrix_ptr->resize(nrows, ncols);
    }

    CUDAMatrix<T>& operator = (const CUDAMatrix<T>& other) {
        if(this == &other) {
            return *this;
        }

        matrix_ptr = other.matrix_ptr;

        return *this;
    }

    int rows() const {
        return matrix_ptr->rows();
    }

    int cols() const {
        return matrix_ptr->cols();
    }

    int ld() const {
        return matrix_ptr->ld();
    }

    T* GetDevicePtr() const {
        return matrix_ptr->GetDevicePtr();
    }

    void SetZero() {
        matrix_ptr->SetZero();
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ToEigen() {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix;
        eigen_matrix.resize(rows(), cols()); eigen_matrix.setZero();

        for(int i = 0; i < cols(); i++) {
            cudaMemcpy(eigen_matrix.col(i).data(), GetDevicePtr() + i * ld(), sizeof(T) * rows(), cudaMemcpyDeviceToHost);
        }

        return eigen_matrix;
    }

public :
    std::shared_ptr<GPUMemBlockType> matrix_ptr;
};
template class CUDAMatrix<float>;

#endif //LIO_SAM_CUDA_CUDA_MATRIX_CUH
