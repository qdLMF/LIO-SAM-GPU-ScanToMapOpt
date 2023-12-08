//
// Created by lmf on 23-9-6.
//

#ifndef LIO_SAM_CUDA_CUDA_ATA_CUH
#define LIO_SAM_CUDA_CUDA_ATA_CUH

#include <eigen3/Eigen/Core>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_matrix.cuh"


struct CUDAATA {
    cudaError_t m_cuda_status;
    cublasHandle_t m_cublas_handle;
    cublasStatus_t m_cublas_status;

    CUDAATA() {
        cublasCreate_v2(&m_cublas_handle);
        m_cublas_status = CUBLAS_STATUS_SUCCESS;
    }

    const float m_alpha = 1.0;
    const float m_beta = 0.0;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_host_ATA;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Compute(CUDAMatrix<float>& A) {
        assert(A.rows() > 0);
        assert(A.cols() > 0);
        assert(A.ld() >= A.rows());

        CUDAMatrix<float> ATA(A.cols(), A.cols()); ATA.resize(A.cols(), A.cols());

        m_cublas_status = \
        cublasSsyrk_v2(
            m_cublas_handle,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T,
            A.cols(),   // number of rows of A.transpose()
            A.rows(),   // number of cols of A.transpose()
            &m_alpha,
            A.GetDevicePtr(),
            A.ld(),
            &m_beta,
            ATA.GetDevicePtr(),
            ATA.ld()
        );
        // m_cuda_status = cudaStreamSynchronize(0);
        assert(cudaSuccess == m_cuda_status);
        assert(CUBLAS_STATUS_SUCCESS == m_cublas_status);

        m_host_ATA.resize(ATA.rows(), ATA.cols());
        m_host_ATA.setZero();
        for(int i = 0; i < ATA.cols(); i++) {
            m_cuda_status = cudaMemcpy((void*)(m_host_ATA.col(i).data()), (void*)(ATA.GetDevicePtr() + i * ATA.ld()), sizeof(float) * (i + 1), cudaMemcpyDeviceToHost);
            assert(cudaSuccess == m_cuda_status);
        }

        m_host_ATA.template triangularView<Eigen::StrictlyLower>() = m_host_ATA.template triangularView<Eigen::StrictlyUpper>().transpose();

        return m_host_ATA;
    }

    ~CUDAATA() {
        if(m_cublas_handle) {
            cublasDestroy(m_cublas_handle);
        }
    }
};


#endif //LIO_SAM_CUDA_CUDA_ATA_CUH
