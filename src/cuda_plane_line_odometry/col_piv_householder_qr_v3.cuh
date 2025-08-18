//
// Created by lmf on 23-11-21.
//

#ifndef LIO_SAM_CUDA_COL_PIV_HOUSEHOLDER_QR_V3_CUH
#define LIO_SAM_CUDA_COL_PIV_HOUSEHOLDER_QR_V3_CUH

#include <eigen3/Eigen/Core>


#define STD_NUMERIC_LIMITS_FLOAT_MIN 0.0000000000000000000000000000000000000117549435082228750796873653722224567781866555677208752150875170627841725945472717285156250000

template<int R, int C>
__device__ inline float mat_col_norm(float input[R][C], int col) {
    float squred_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < R; row++) {
        squred_norm += powf(input[row][col], 2);
    }
    return sqrt(squred_norm);
}

template<int R, int C>
__device__ inline float mat_col_squared_norm(float input[R][C], int col) {
    float squred_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < R; row++) {
        squred_norm += powf(input[row][col], 2);
    }
    return squred_norm;
}

template<int R, int C>
__device__ inline float mat_col_max(float input[R][C], int col, int& idx) {
    float max_val = input[R - 1][col];
    int   max_idx = R - 1;
    #pragma unroll 8
    for (int row = 1; row < R; row++) {
        if (input[R - 1 - row][col] > max_val) {
            max_val = input[R - 1 - row][col];
            max_idx = R - 1 - row;
        }
    }
    idx = max_idx;
    return max_val;
}

template<int R, int C>
__device__ inline float mat_col_max(float input[R][C], int col) {
    float max_val = input[R - 1][col];
    #pragma unroll 8
    for (int row = 1; row < R; row++) {
        if (input[R - 1 - row][col] > max_val) {
            max_val = input[R - 1 - row][col];
        }
    }
    return max_val;
}

template<int R, int C>
__device__ inline float mat_col_tail_norm(float input[R][C], int col, int tail) {
    float squred_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < tail; row++) {
        squred_norm += powf(input[R - 1 - row][col], 2);
    }
    return sqrt(squred_norm);
}

template<int R, int C>
__device__ inline float mat_col_tail_squared_norm(float input[R][C], int col, int tail) {
    float squred_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < tail; row++) {
        squred_norm += powf(input[R - 1 - row][col], 2);
    }
    return squred_norm;
}

template<int R, int C>
__device__ inline float mat_col_tail_max(float input[R][C], int col, int tail, int& idx) {
    float max_val = input[R - 1][col];
    int   max_idx = R - 1;
    #pragma unroll 8
    for (int row = 1; row < tail; row++) {
        if (input[R - 1 - row][col] > max_val) {
            max_val = input[R - 1 - row][col];
            max_idx = R - 1 - row;
        }
    }
    idx = max_idx - (R - tail);
    return max_val;
}

template<int R, int C>
__device__ inline float mat_col_tail_max(float input[R][C], int col, int tail) {
    float max_val = input[R - 1][col];
    #pragma unroll 8
    for (int row = 1; row < tail; row++) {
        if (input[R - 1 - row][col] > max_val) {
            max_val = input[R - 1 - row][col];
        }
    }
    return max_val;
}

template<int R, int C>
__device__ inline void mat_col_tail_set_scalar(float input[R][C], int col, int tail, float scalar) {
    #pragma unroll 8
    for (int row = 0; row < tail; row++) {
        input[R - 1 - row][col] = scalar;
    }
}

template<int R, int C>
__device__ inline void mat_col_tail_mul_scalar(float input[R][C], int col, int tail, float scalar) {
    #pragma unroll 8
    for (int row = 0; row < tail; row++) {
        input[R - 1 - row][col] *= scalar;
    }
}

template<int R, int C>
__device__ inline void mat_col_tail_div_scalar(float input[R][C], int col, int tail, float scalar) {
    #pragma unroll 8
    for (int row = 0; row < tail; row++) {
        input[R - 1 - row][col] /= scalar;
    }
}

template<int R, int C>
__device__ inline void mat_col_swap(float input[R][C], int col_0, int col_1) {
    float temp;
    #pragma unroll 8
    for (int row = 0; row < R; row++) {
        temp              = input[row][col_0];
        input[row][col_0] = input[row][col_1];
        input[row][col_1] = temp;
    }
}

// ----------

template<int N>
__device__ inline float vec_norm(float input[N]) {
    float squared_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < N; row++) {
        squared_norm += powf(input[row], 2);;
    }
    return sqrt(squared_norm);
}

template<int N>
__device__ inline float vec_squared_norm(float input[N]) {
    float squared_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < N; row++) {
        squared_norm += powf(input[row], 2);;
    }
    return squared_norm;
}

template<int N>
__device__ inline float vec_max(float input[N], int& idx) {
    float max_val = input[0];
    int   max_idx = 0;
    #pragma unroll 8
    for (int row = 1; row < N; row++) {
        if (input[row] > max_val) {
            max_val = input[row];
            max_idx = row;
        }
    }
    idx = max_idx;
    return max_val;
}

template<int N>
__device__ inline float vec_max(float input[N]) {
    float max_val = input[0];
    #pragma unroll 8
    for (int row = 1; row < N; row++) {
        if (input[row] > max_val) {
            max_val = input[row];
        }
    }
    return max_val;
}

template<int N>
__device__ inline float vec_tail_norm(float input[N], int tail) {
    float squared_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < tail; row++) {
        squared_norm += powf(input[N - 1 - row], 2);
    }
    return sqrt(squared_norm);
}

template<int N>
__device__ inline float vec_tail_squared_norm(float input[N], int tail) {
    float squared_norm = 0.0f;
    #pragma unroll 8
    for (int row = 0; row < tail; row++) {
        squared_norm += powf(input[N - 1 - row], 2);
    }
    return squared_norm;
}

template<int N>
__device__ inline float vec_tail_max(float input[N], int tail, int& idx) {
    float max_val = input[N - 1];
    int   max_idx = N - 1;
    #pragma unroll 8
    for (int row = 1; row < tail; row++) {
        if (input[N - 1 - row] > max_val) {
            max_val = input[N - 1 - row];
            max_idx = N - 1 - row;
        }
    }
    idx = max_idx - (N - tail);;
    return max_val;
}

template<int N>
__device__ inline float vec_tail_max(float input[N], int tail) {
    float max_val = input[N - 1];
    #pragma unroll 8
    for (int row = 1; row < tail; row++) {
        if (input[N - 1 - row] > max_val) {
            max_val = input[N - 1 - row];
        }
    }
    return max_val;
}

template<int N>
__device__ inline void vec_swap(float input[N], int idx_0, int idx_1) {
    float temp   = input[idx_0];
    input[idx_0] = input[idx_1];
    input[idx_1] = temp;
}

template<unsigned int NROW>     // NROW must be >= 3!
struct HouseholderSequenceNx3 {
    float m_vectors[NROW][3];
    float m_coeffs[3] = {0.0f, 0.0f, 0.0f};

    bool m_reverse = false;

    int m_length = 0;
};

template<unsigned int NROW>     // NROW must be >= 3!
struct ColPivHouseholderQRNx3 {
    int m_tid;

    float m_qr[NROW][3];
    float m_c[NROW];
    float m_hCoeffs[3]         = {0.0f, 0.0f, 0.0f};
    float m_solution[3]        = {0.0f, 0.0f, 0.0f};
    float m_temp[3]            = {0.0f, 0.0f, 0.0f};
    float m_colNormsUpdated[3] = {0.0f, 0.0f, 0.0f};
    float m_colNormsDirect[3]  = {0.0f, 0.0f, 0.0f};
    float m_rhs_workspace      =  0.0f;

    unsigned char m_colsTranspositions[3] = {0, 0, 0};
    unsigned char m_colsPermutation[3]    = {0, 0, 0};

    unsigned char m_nonzero_pivots = 0;

    HouseholderSequenceNx3<NROW> m_hhs;

    
    __device__ ColPivHouseholderQRNx3();
    __device__ explicit ColPivHouseholderQRNx3(const Eigen::Matrix<float, NROW, 3>& A, int tid);

    __device__ void Compute(const Eigen::Matrix<float, NROW, 3>& A);
    __device__ void ApplyHouseholderOnTheLeft(int col_idx);
    __device__ void MakeHouseHolderInPlace(int col_idx, float& beta);

    __device__ Eigen::Vector3f Solve(const Eigen::Matrix<float, NROW, 1>& rhs);
    __device__ void SetUpHouseholderSequence();
    __device__ void ApplyHouseholderSequenceOnTheLeftToRHS(const Eigen::Matrix<float, NROW, 1>& rhs);
    __device__ void ApplyHouseholderOnTheLeftToRHS(int actual_k);
};

template<unsigned int NROW>
__device__ ColPivHouseholderQRNx3<NROW>::ColPivHouseholderQRNx3() = default;

template<unsigned int NROW>
__device__ ColPivHouseholderQRNx3<NROW>::ColPivHouseholderQRNx3(const Eigen::Matrix<float, NROW, 3>& A, int tid) {
    m_tid = tid;

    #pragma unroll 8
    for (int i = 0; i < NROW; i++) {
        m_qr[i][0] = 0.0f; m_qr[i][1] = 0.0f; m_qr[i][2] = 0.0f;
    }

    #pragma unroll 8
    for (int i = 0; i < NROW; i++) {
        m_hhs.m_vectors[i][0] = 0.0f; m_hhs.m_vectors[i][1] = 0.0f; m_hhs.m_vectors[i][2] = 0.0f;
    }

    #pragma unroll 8
    for (int i = 0; i < NROW; i++) {
        m_c[i] = 0.0f;
    }

    Compute(A);
}

template<unsigned int NROW>
__device__ inline void ColPivHouseholderQRNx3<NROW>::Compute(const Eigen::Matrix<float, NROW, 3>& A) {
    #pragma unroll 8
    for (int r = 0; r < NROW; r++) {
        m_qr[r][0] = A(r, 0); m_qr[r][1] = A(r, 1); m_qr[r][2] = A(r, 2);
    }

    #pragma unroll 8
    for (int k = 0; k < 3; k++) {
        m_colNormsDirect[k]  = mat_col_norm<NROW, 3>(m_qr, k);
        m_colNormsUpdated[k] = m_colNormsDirect[k];
    }

    float threshold_helper = Eigen::numext::abs2<float>( vec_max<3>(m_colNormsUpdated) * Eigen::NumTraits<float>::epsilon() ) / float(NROW);
    float norm_downdate_threshold = Eigen::numext::sqrt( Eigen::NumTraits<float>::epsilon() );

    m_nonzero_pivots = 3;

    #pragma unroll 8
    for (int k = 0; k < 3; k++) {
        // first, we look up in our table m_colNormsUpdated which column has the biggest norm
        int biggest_col_index;
        float biggest_col_sq_norm = Eigen::numext::abs2( vec_tail_max<3>(m_colNormsUpdated, 3 - k, biggest_col_index) );
        biggest_col_index += k;

        // Track the number of meaningful pivots but do not stop the decomposition to make
        // sure that the initial matrix is properly reproduced. See bug 941.
        if(    m_nonzero_pivots == 3
            && biggest_col_sq_norm < threshold_helper * float(NROW - k) )
        {
            m_nonzero_pivots = k;
        }

        // apply the transposition to the columns
        m_colsTranspositions[k] = biggest_col_index;
        if ( k != biggest_col_index ) {
            mat_col_swap<NROW, 3>(m_qr, k, biggest_col_index);
            vec_swap<3>(m_colNormsUpdated, k, biggest_col_index);
            vec_swap<3>(m_colNormsDirect, k, biggest_col_index);
        }

        // generate the householder vector, store it below the diagonal
        float beta;
        MakeHouseHolderInPlace(k, beta);

        // apply the householder transformation to the diagonal coefficient
        m_qr[k][k] = beta;

        // apply the householder transformation
        ApplyHouseholderOnTheLeft(k);

        // update our table of norms of the columns
        #pragma unroll 8
        for (int j = k + 1; j < 3; j++) {
            // The following implements the stable norm downgrade step discussed in
            // http://www.netlib.org/lapack/lawnspdf/lawn176.pdf
            // and used in LAPACK routines xGEQPF and xGEQP3.
            // See lines 278-297 in http://www.netlib.org/lapack/explore-html/dc/df4/sgeqpf_8f_source.html
            if ( m_colNormsUpdated[j] != float(0) ) {
                float temp = Eigen::numext::abs2<float>( m_qr[k][j] ) / m_colNormsUpdated[j];
                temp = ( float(1) + temp ) * ( float(1) - temp );
                temp = temp < float(0) ? float(0) : temp;
                float temp2 = temp * Eigen::numext::abs2<float>( m_colNormsUpdated[j] / m_colNormsDirect[j] );

                if ( temp2 <= norm_downdate_threshold ) {
                    m_colNormsDirect [j] = mat_col_tail_norm<NROW, 3>(m_qr, j, NROW - k - 1);
                    m_colNormsUpdated[j] = m_colNormsDirect[j];
                } else {
                    m_colNormsUpdated[j] *= Eigen::numext::sqrt(temp);
                }
            }
        }
    }

    m_colsPermutation[0] = 0; m_colsPermutation[1] = 1; m_colsPermutation[2] = 2;

    #pragma unroll 8
    for (int k = 0; k < 3; k++) {
        int temp = 0;
        temp = m_colsPermutation[k];
        m_colsPermutation[k] = m_colsPermutation[ m_colsTranspositions[k] ];
        m_colsPermutation[ m_colsTranspositions[k] ] = temp;
    }
}

template<unsigned int NROW>
__device__ inline void ColPivHouseholderQRNx3<NROW>::MakeHouseHolderInPlace(int col_idx, float& beta) {
    int tail = NROW - col_idx;
    float tailSqNorm = float(0);
    if( tail >= 2 ) {
        tailSqNorm = mat_col_tail_squared_norm<NROW, 3>(m_qr, col_idx, tail - 1); // 对角线元素下面的部分，也就是essential_part
    }
    float c0 = m_qr[col_idx][col_idx]; // 对角线元素
    const float tol = STD_NUMERIC_LIMITS_FLOAT_MIN; // (std::numeric_limits<float>::min)();

    // tailSqNorm <= tol && Eigen::numext::abs2(Eigen::numext::imag(c0)) <= tol
    if ( tailSqNorm <= tol && Eigen::numext::abs2(c0) <= tol ) {
        m_hCoeffs[col_idx] = float(0);
        beta = c0;
        mat_col_tail_set_scalar<NROW, 3>(m_qr, col_idx, tail, 0.0f);
    } else {
        beta = sqrt(Eigen::numext::abs2(c0) + tailSqNorm);
        if ( c0 >= float(0) ) {
            beta = -beta;
        }
        mat_col_tail_div_scalar<NROW, 3>(m_qr, col_idx, tail - 1, c0 - beta);
        m_hCoeffs[col_idx] = (beta - c0) / beta;
    }
}

template<unsigned int NROW>
__device__ inline void ColPivHouseholderQRNx3<NROW>::ApplyHouseholderOnTheLeft(int col_idx) {
    int brc_rows = NROW - col_idx;

    float tau = m_hCoeffs[col_idx];

    if (brc_rows == 1) {
        #pragma unroll 8
        for (int r = col_idx; r < NROW; r++) {
            #pragma unroll 8
            for (int c = col_idx + 1; c < 3; c++) {
                m_qr[r][c] *= (float(1) - tau);
            }
        }
    } else if( tau != float(0) ) {
        #pragma unroll 8
        for (int i = col_idx + 1; i < 3; i++) {
            float temp_dot_product = 0.0f;
            #pragma unroll 8
            for (int j = col_idx + 1; j < NROW; j++) {
                temp_dot_product += m_qr[j][col_idx] * m_qr[j][i];
            }
            m_temp[i] = temp_dot_product;
        }

        #pragma unroll 8
        for (int i = col_idx + 1; i < 3; i++) {
            m_temp[i] += m_qr[col_idx][i];
        }

        #pragma unroll 8
        for (int i = col_idx + 1; i < 3; i++) {
            m_qr[col_idx][i] -= tau * m_temp[i];
        }

        #pragma unroll 8
        for (int i = col_idx + 1; i < NROW; i++) {
            #pragma unroll 8
            for (int j = col_idx + 1; j < 3; j++) {
                m_qr[i][j] -= tau * m_qr[i][col_idx] * m_temp[j];
            }
        }
    }
}

template<unsigned int NROW>
__device__ inline void ColPivHouseholderQRNx3<NROW>::SetUpHouseholderSequence() {
    #pragma unroll 8
    for (int r = 0; r < NROW; r++) {
        m_hhs.m_vectors[r][0] = m_qr[r][0]; m_hhs.m_vectors[r][1] = m_qr[r][1]; m_hhs.m_vectors[r][2] = m_qr[r][2];
    }
    m_hhs.m_coeffs[0] = m_hCoeffs[0]; m_hhs.m_coeffs[1] = m_hCoeffs[1]; m_hhs.m_coeffs[2] = m_hCoeffs[2];
    m_hhs.m_length = m_nonzero_pivots;
    m_hhs.m_reverse = true;
}

template<unsigned int NROW>
__device__ inline Eigen::Vector3f ColPivHouseholderQRNx3<NROW>::Solve(const Eigen::Matrix<float, NROW, 1>& rhs) {
    SetUpHouseholderSequence();
    ApplyHouseholderSequenceOnTheLeftToRHS(rhs);

    if (m_nonzero_pivots == 0) {
        m_solution[0] = 0.0f;
        m_solution[1] = 0.0f;
        m_solution[2] = 0.0f;
        return Eigen::Vector3f{0.0f, 0.0f, 0.0f};
    }

    float x[3];
    x[2] = 2 < m_nonzero_pivots ? ((m_c[2]) / m_qr[2][2]) : 0.0f;
    x[1] = 1 < m_nonzero_pivots ? ((m_c[1] - m_qr[1][2] * x[2]) / m_qr[1][1]) : 0.0f;
    x[0] = 0 < m_nonzero_pivots ? ((m_c[0] - m_qr[0][1] * x[1] - m_qr[0][2] * x[2]) / m_qr[0][0]) : 0.0f;

    m_solution[ m_colsPermutation[0] ] = 0 < m_nonzero_pivots ? x[0] : 0.0f;
    m_solution[ m_colsPermutation[1] ] = 1 < m_nonzero_pivots ? x[1] : 0.0f;
    m_solution[ m_colsPermutation[2] ] = 2 < m_nonzero_pivots ? x[2] : 0.0f;

    return Eigen::Vector3f{m_solution[0], m_solution[1], m_solution[2]};
}

template<unsigned int NROW>
__device__ inline void ColPivHouseholderQRNx3<NROW>::ApplyHouseholderSequenceOnTheLeftToRHS(const Eigen::Matrix<float, NROW, 1>& rhs) {
    #pragma unroll 8
    for (int r = 0; r < NROW; r++) {
        m_c[r] = rhs(r);
    }

    #pragma unroll 8
    for (int k = 0; k < m_hhs.m_length; k++) {
        int actual_k = m_hhs.m_reverse ? k : m_hhs.m_length - k - 1;
        int dstStart = NROW - actual_k;

        int start = actual_k + 1;

        ApplyHouseholderOnTheLeftToRHS(actual_k);
    }
}

template<unsigned int NROW>
__device__ inline void ColPivHouseholderQRNx3<NROW>::ApplyHouseholderOnTheLeftToRHS(int actual_k) {
    int brc_rows = NROW - actual_k;

    float tau = m_hhs.m_coeffs[actual_k];

    if (brc_rows == 1)  {
        #pragma unroll 8
        for (int r = actual_k; r < NROW; r++) {
            m_c[r] *= (float(1) - tau);
        }
    } else {
        m_rhs_workspace = m_c[actual_k];
        #pragma unroll 8
        for (int r = actual_k + 1; r < NROW; r++) {
            m_rhs_workspace += m_hhs.m_vectors[r][actual_k] * m_c[r];
        }
        m_c[actual_k] -= tau * m_rhs_workspace;
        #pragma unroll 8
        for (int r = actual_k + 1; r < NROW; r++) {
            m_c[r] -= tau * m_rhs_workspace * m_hhs.m_vectors[r][actual_k];
        }
    }
}

#endif