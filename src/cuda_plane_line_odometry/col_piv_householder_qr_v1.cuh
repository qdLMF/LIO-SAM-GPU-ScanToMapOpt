//
// Created by lmf on 23-9-6.
//

#ifndef LIO_SAM_CUDA_COL_PIV_HOUSEHOLDER_QR_V1_CUH
#define LIO_SAM_CUDA_COL_PIV_HOUSEHOLDER_QR_V1_CUH

#include <eigen3/Eigen/Core>

#define STD_NUMERIC_LIMITS_FLOAT_MIN 0.0000000000000000000000000000000000000117549435082228750796873653722224567781866555677208752150875170627841725945472717285156250000

struct HouseholderSequence5x3
{
    Eigen::Matrix<float, 5, 3> m_vectors;
    Eigen::Vector3f m_coeffs;
    bool m_reverse{};
    int m_length{};
    int m_shift{};
};

struct ColPivHouseholderQR5x3
{
    static const int rows = 5;
    static const int cols = 3;
    static const int size = 3;	// diag size

    Eigen::Matrix<float, 5, 3> m_qr;
    Eigen::Vector3f m_hCoeffs{0.0, 0.0, 0.0};
    Eigen::Vector3f m_temp;
    Eigen::Vector3i m_colsTranspositions;
    int number_of_transpositions{};
    Eigen::Vector3f m_colNormsUpdated;
    Eigen::Vector3f m_colNormsDirect;

    int m_nonzero_pivots{};

    Eigen::Vector3i m_colsPermutation;

    bool m_isInitialized = false;
    bool m_isSolutionUpdated = false;

    HouseholderSequence5x3 m_hhs;
    Eigen::Matrix<float, 5, 1> m_c;
    Eigen::Matrix<float, 1, 1> m_rhs_workspace;
    Eigen::Vector3f m_solution;

    // funcs
    __device__ ColPivHouseholderQR5x3();
    __device__ explicit ColPivHouseholderQR5x3(const Eigen::Matrix<float, 5, 3>& A);
    __device__ void Compute(const Eigen::Matrix<float, 5, 3>& A);
    __device__ void ApplyHouseholderOnTheLeft(int col_idx);
    __device__ void MakeHouseHolderInPlace(int col_idx, float& beta);
    __device__ void SetUpHouseholderSequence();
    __device__ void ApplyHouseholderSequenceOnTheLeftToRHS(const Eigen::Matrix<float, 5, 1>& rhs);
    __device__ void ApplyHouseholderOnTheLeftToRHS(int actual_k, int dstStart, int start, int dst_num_cols);
    __device__ Eigen::Vector3f Solve(const Eigen::Matrix<float, 5, 1>& rhs);
};

__device__ ColPivHouseholderQR5x3::ColPivHouseholderQR5x3() = default;

__device__ ColPivHouseholderQR5x3::ColPivHouseholderQR5x3(const Eigen::Matrix<float, 5, 3>& A)
{
    Compute(A);
}

__device__ Eigen::Vector3f ColPivHouseholderQR5x3::Solve(const Eigen::Matrix<float, 5, 1>& rhs)
{
    SetUpHouseholderSequence();
    ApplyHouseholderSequenceOnTheLeftToRHS(rhs);

    if(m_nonzero_pivots == 0)
    {
        m_solution.setZero();
        return m_solution;
    }

    Eigen::Matrix<float, 3, 1> x{0.0, 0.0, 0.0};
    Eigen::Matrix<float, 3, 1> b = m_c.topRows(3).eval();
    Eigen::Matrix<float, 3, 3> A = m_qr.topLeftCorner( 3 , 3 ).eval();
    A(1,0) = 0.0; A(2,0) = 0.0; A(2,1) = 0.0;

    Eigen::Vector3i flag{0, 0, 0};
    for(int i = 0; i < 3; i++) {
        flag(i) = i < m_nonzero_pivots ? 1 : 0;
    }

    x(2) = flag(2) == 1 ? ((b(2)) / A(2, 2)) : 0.0;
    x(1) = flag(1) == 1 ? ((b(1) - A(1, 2) * x(2)) / A(1, 1)) : 0.0;
    x(0) = flag(0) == 1 ? ((b(0) - A(0,1) * x(1) - A(0,2) * x(2)) / A(0,0)) : 0.0;

    for(int i = 0; i < m_nonzero_pivots; i++)
    {
        m_solution( m_colsPermutation(i) ) = x(i);    // x.row(i);
    }
    for(int i = m_nonzero_pivots; i < m_hhs.m_vectors.cols(); i++)
    {
        m_solution( m_colsPermutation(i) ) = 0.0;
    }

    m_isSolutionUpdated = true;

    return m_solution;
}

__device__ void ColPivHouseholderQR5x3::SetUpHouseholderSequence()
{
    m_hhs.m_vectors = m_qr;
    m_hhs.m_coeffs = m_hCoeffs.conjugate();
    m_hhs.m_reverse = false;
    m_hhs.m_length = m_qr.diagonalSize();
    m_hhs.m_shift = 0;

    m_hhs.m_length = m_nonzero_pivots;

    m_hhs.m_vectors = m_qr;
    m_hhs.m_coeffs = m_hCoeffs.conjugate();
    m_hhs.m_reverse = true;
    m_hhs.m_length = m_hhs.m_length;
    m_hhs.m_shift = m_hhs.m_shift;
}

__device__ void ColPivHouseholderQR5x3::ApplyHouseholderOnTheLeftToRHS(
        int actual_k,
        int dstStart,
        int start,
        int dst_num_cols
) {
    int brc_rows = dstStart;

    float tau = m_hhs.m_coeffs.coeff(actual_k);

    if(brc_rows == 1) {
        m_c.bottomRightCorner(dstStart, dst_num_cols) *= float(1) - tau;
    }
    else
    {
        m_rhs_workspace = m_hhs.m_vectors.col(actual_k).tail(m_hhs.m_vectors.rows() - start).adjoint() * m_c.col(0).tail(dstStart - 1);
        m_rhs_workspace += m_c.bottomRightCorner(dstStart, dst_num_cols).row(0);
        m_c.bottomRightCorner(dstStart, dst_num_cols).row(0) -= tau * m_rhs_workspace;
        m_c.col(0).tail(dstStart - 1) -= tau * m_hhs.m_vectors.col(actual_k).tail(m_hhs.m_vectors.rows() - start) * m_rhs_workspace;
    }
}

__device__ void ColPivHouseholderQR5x3::ApplyHouseholderSequenceOnTheLeftToRHS(const Eigen::Matrix<float, 5, 1>& rhs)
{
    m_c = rhs;

    Eigen::Matrix<float, 1, 1, Eigen::RowMajor, 1, 1> workspace;

    bool inputIsIdentity = false;

    workspace.resize(m_c.cols());   // ???
    for(int k = 0; k < m_hhs.m_length; k++)
    {
        int actual_k = m_hhs.m_reverse ? k : m_hhs.m_length - k - 1;
        int dstStart = m_hhs.m_vectors.rows() - m_hhs.m_shift - actual_k;

        int start = actual_k + 1 + m_hhs.m_shift;
        auto essential_vector = m_hhs.m_vectors.col(actual_k).tail(m_hhs.m_vectors.rows() - start);

        ApplyHouseholderOnTheLeftToRHS(actual_k, dstStart, start, inputIsIdentity ? dstStart : m_solution.cols());
    }
}

__device__ void ColPivHouseholderQR5x3::Compute(const Eigen::Matrix<float, 5, 3>& A)
{
    m_qr = A;
    number_of_transpositions = 0;

    for (int k = 0; k < cols; ++k)
    {
        m_colNormsDirect.coeffRef(k) = m_qr.col(k).norm();
        m_colNormsUpdated.coeffRef(k) = m_colNormsDirect.coeffRef(k);
    }

    float threshold_helper =  Eigen::numext::abs2<float>( m_colNormsUpdated.maxCoeff() * Eigen::NumTraits<float>::epsilon() ) / float(rows);
    float norm_downdate_threshold = Eigen::numext::sqrt( Eigen::NumTraits<float>::epsilon() );

    m_nonzero_pivots = size;

    for(int k = 0; k < size; k++)
    {
        // first, we look up in our table m_colNormsUpdated which column has the biggest norm
        int biggest_col_index;
        float biggest_col_sq_norm = Eigen::numext::abs2( m_colNormsUpdated.tail(cols - k).maxCoeff( &biggest_col_index ) );
        biggest_col_index += k;

        // Track the number of meaningful pivots but do not stop the decomposition to make
        // sure that the initial matrix is properly reproduced. See bug 941.
        if(    m_nonzero_pivots == size
            && biggest_col_sq_norm < threshold_helper * float(rows - k) )
        {
            m_nonzero_pivots = k;
        }

        // apply the transposition to the columns
        m_colsTranspositions.coeffRef(k) = biggest_col_index;
        if( k != biggest_col_index )
        {
            m_qr.col(k).swap( m_qr.col(biggest_col_index) );
            float temp = 0.0;
            temp = m_colNormsUpdated.coeffRef(k);
            m_colNormsUpdated.coeffRef(k) = m_colNormsUpdated.coeffRef(biggest_col_index);
            m_colNormsUpdated.coeffRef(biggest_col_index) = temp;
            temp = m_colNormsDirect.coeffRef(k);
            m_colNormsDirect.coeffRef(k) = m_colNormsDirect.coeffRef(biggest_col_index);
            m_colNormsDirect.coeffRef(biggest_col_index) = temp;

            ++number_of_transpositions;
        }

        // generate the householder vector, store it below the diagonal
        float beta;
        MakeHouseHolderInPlace(k, beta);

        // apply the householder transformation to the diagonal coefficient
        m_qr.coeffRef( k , k ) = beta;

        // apply the householder transformation
        ApplyHouseholderOnTheLeft(k);

        // update our table of norms of the columns
        for(int j = k + 1; j < cols; ++j)
        {
            // The following implements the stable norm downgrade step discussed in
            // http://www.netlib.org/lapack/lawnspdf/lawn176.pdf
            // and used in LAPACK routines xGEQPF and xGEQP3.
            // See lines 278-297 in http://www.netlib.org/lapack/explore-html/dc/df4/sgeqpf_8f_source.html
            if( m_colNormsUpdated.coeffRef(j) != float(0) )
            {
                float temp = abs( m_qr.coeffRef(k, j) ) / m_colNormsUpdated.coeffRef(j);
                temp = ( float(1) + temp ) * ( float(1) - temp );
                temp = temp < float(0) ? float(0) : temp;
                float temp2 = temp * Eigen::numext::abs2<float>( m_colNormsUpdated.coeffRef(j) / m_colNormsDirect.coeffRef(j) );
                if( temp2 <= norm_downdate_threshold )
                {
                    m_colNormsDirect.coeffRef(j) = m_qr.col(j).tail(rows - k - 1).norm();
                    m_colNormsUpdated.coeffRef(j) = m_colNormsDirect.coeffRef(j);
                }
                else
                {
                    m_colNormsUpdated.coeffRef(j) *= Eigen::numext::sqrt(temp);
                }
            }
        }
    }

    m_colsPermutation(0) = 0; m_colsPermutation(1) = 1; m_colsPermutation(2) = 2;
    for(int k = 0; k < size; k++)
    {
        int temp = 0;
        temp = m_colsPermutation(k);
        m_colsPermutation(k) = m_colsPermutation( m_colsTranspositions.coeff(k) );
        m_colsPermutation(m_colsTranspositions.coeff(k)) = temp;
    }

    m_isInitialized = true;
    m_isSolutionUpdated = false;
}

__device__ void ColPivHouseholderQR5x3::MakeHouseHolderInPlace(int col_idx, float& beta) {
    int tail = rows - col_idx;
    float tailSqNorm = float(0);
    if( tail >= 2 ) {
        tailSqNorm = m_qr.col(col_idx).tail(tail - 1).squaredNorm();	// 对角线元素下面的部分，也就是essential_part
    }
    float c0 = m_qr.col(col_idx).tail(tail).coeff(0);   // 对角线元素
    const float tol = STD_NUMERIC_LIMITS_FLOAT_MIN;     // (std::numeric_limits<float>::min)();

    if( tailSqNorm <= tol && Eigen::numext::abs2(c0) <= tol )	// tailSqNorm <= tol && Eigen::numext::abs2(Eigen::numext::imag(c0)) <= tol
    {
        m_hCoeffs.coeffRef(col_idx) = float(0);
        beta = c0;  // numext::real(c0);
        m_qr.col(col_idx).tail(tail - 1).setZero();
    }
    else
    {
        beta = sqrt(Eigen::numext::abs2(c0) + tailSqNorm);
        if( c0 >= float(0) )	// numext::real(c0) >= RealScalar(0)
        {
            beta = -beta;
        }
        m_qr.col(col_idx).tail(tail - 1) /= (c0 - beta);
        m_hCoeffs.coeffRef(col_idx) = (beta - c0) / beta;		// tau = conj((beta - c0) / beta); // tau = m_hCoeffs.coeffRef(k);
    }
}

__device__ void ColPivHouseholderQR5x3::ApplyHouseholderOnTheLeft(int col_idx)
{
    int brc_rows = m_qr.rows() - col_idx;

    float tau = m_hCoeffs.coeffRef(col_idx);

    if(brc_rows == 1)
    {
        m_qr.bottomRightCorner( m_qr.rows() - col_idx , m_qr.cols() - col_idx - 1 ) *= (float(1) - tau);
    }
    else if( tau != float(0) )
    {
        m_temp.tail(m_temp.size() - col_idx - 1) = \
		m_qr.col(col_idx).tail(m_qr.rows() - col_idx - 1).adjoint() * m_qr.bottomRightCorner( m_qr.rows() - col_idx - 1, m_qr.cols() - col_idx - 1 );
        m_temp.tail(m_temp.size() - col_idx - 1) += m_qr.row(col_idx).tail(m_qr.cols() - col_idx - 1);
        m_qr.row(col_idx).tail(m_qr.cols() - col_idx - 1) -= tau * m_temp.tail(m_temp.size() - col_idx - 1);
        m_qr.bottomRightCorner( m_qr.rows() - col_idx - 1, m_qr.cols() - col_idx - 1 ) -= \
        tau * m_qr.col(col_idx).tail(m_qr.rows() - col_idx - 1) * m_temp.tail(m_temp.size() - col_idx - 1).transpose();
    }
}

#endif //LIO_SAM_CUDA_COL_PIV_HOUSEHOLDER_QR_CUH
