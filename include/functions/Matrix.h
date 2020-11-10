#ifndef MATRIX_H
#define MATRIX_H

#include <immintrin.h>
#include <vector>

#include "DataTypeHandling.h"

class Matrix {
private:
    unsigned int N;
    // There are two main reasons why we use an std::vector here instead of a raw pointer
    //  (1) std::vector is the more modern c++ style and raw pointers are somewhat discuraged (see next comment)
    //  (2) It turns our there is a good reason why we should not use raw pointers. It makes it really difficult to implement appropriate copy / move constructors. I would sometimes run into weird memory issues, because the compiler provided an implicit copy / move c'tor. Of course it would be possible to properly implement move/copy/assignment operators (rule of 0/3/5 https://en.cppreference.com/w/cpp/language/rule_of_three) but thats more work than I need. 
    std::vector<data_t> data;

public:

    /**
     * @brief  Copies the upper left N_sub x N_sub matrix from other into the new object. Caller has to make sure that N_sub <= other.size()
     * @note   
     * @param  &other: 
     * @param  N_sub: 
     * @retval 
     */
    Matrix(Matrix const &other, unsigned int N_sub) : N(N_sub), data(N_sub * N_sub) {
        for (unsigned int i = 0; i < N_sub; ++i) {
            for (unsigned int j = 0; j < N_sub; ++j) {
                this->operator()(i, j) = other(i,j);
            }
        }
    }

    Matrix(unsigned int _size) : N(_size), data(_size * _size, 0){}

    ~Matrix() { }

    inline unsigned int size() const { return N; }

    void replace_row(unsigned int row, data_t const * const x) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(i, row) = x[i];
        }
    }

    void replace_column(unsigned int col, data_t const * const x) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(col, i) = x[i];
        }
    }

    void rank_one_update(unsigned int row, data_t const * const x) {
        for (unsigned int i = 0; i < N; ++i) {
            if (row == i) {
                this->operator()(i,i) += x[i];
            } else {
                this->operator()(i,row) += x[i];
                this->operator()(row,i) += x[i];
            }
        }
    }

    data_t & operator [](int i) {return  data[i*N];}
    data_t operator [](int i) const {return data[i*N];}

    data_t & operator()(int i, int j) { return data[i*N+j]; }
    data_t operator()(int i, int j) const { return data[i*N+j]; }
};


inline std::string to_string(Matrix const &mat, unsigned int N_sub) {
    std::string s = "[";

    for (unsigned int i = 0; i < N_sub; ++i) {
        s += "[";
        for (unsigned int j = 0; j < N_sub; ++j) {
            if (j < N_sub - 1) {
                s += std::to_string(mat(i,j)) + ",";
            } else {
                s += std::to_string(mat(i,j));
            }
        }

        if (i < N_sub - 1) {
            s += "],\n";
        } else {
            s += "]";
        }
    }

    return s + "]";
}

inline std::string to_string(Matrix const &mat) {
    return to_string(mat,mat.size());
}

inline Matrix cholesky(Matrix const &in, unsigned int N_sub) {
    Matrix L(in, N_sub);

    for (unsigned int j = 0; j < N_sub; ++j) {
        data_t sum = 0.0;

        for (unsigned int k = 0; k < j; ++k) {
            sum += L(j,k)*L(j,k);
        }

        L(j,j) = std::sqrt(in(j,j) - sum);

        for (unsigned int i = j + 1; i < N_sub; ++i) {
            data_t sum = 0.0;

            for (unsigned int k = 0; k < j; ++k) {
                sum += L(i,k) * L(j,k);
            }
            L(i,j) = (in(i,j) - sum) / L(j,j);
        }
    }
    return L;
}

template <typename T> 
T sum_2_vec_product_avx(const T * u, const T * v, unsigned int size);

// Adapted from https://github.com/pashminacameron/optimization_examples/blob/master/cpp_cholesky/cholesky_avx.cpp
template <>
float sum_2_vec_product_avx<float>(const float * u, const float * v, unsigned int size) {
    // Process 8 elements in one lane
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int groups_8 = size / 8;  // groups of 8 elements
    int groups_1 = size % 8; // remaining groups of 1

    __m256 singleLane = _mm256_setzero_ps();
    _mm256_zeroall();
    for (int it = 0; it < groups_8; it++) {
        __m256 a1 = _mm256_load_ps(u + 8 * it);
        __m256 b1 = _mm256_load_ps(v + 8 * it);
        singleLane = _mm256_add_ps(singleLane, _mm256_mul_ps(a1, b1));
    }
    singleLane = _mm256_hadd_ps(singleLane, singleLane);
    singleLane = _mm256_hadd_ps(singleLane, singleLane);
    singleLane = _mm256_hadd_ps(singleLane, singleLane);
    _mm256_storeu_ps(&acc[0], singleLane); // we have the answer in acc[0] as we have already done the horizontal add

    // Add last few after multiples of 8
    if (groups_1) {
        for (unsigned int i = groups_8 * 8; i < size; i++) {
            acc[0] += u[i] * v[i];
        }
    }
    return acc[0];
}

template <>
double sum_2_vec_product_avx<double>(const double * u, const double * v, unsigned int size) {
    // Process 4 elements in one lane
    double acc[4] = { 0, 0, 0, 0 };
    int groups_4 = size / 4;  // groups of 4 elements
    int groups_1 = size % 4; // remaining groups of 1

    __m256d singleLane = _mm256_setzero_pd();
    _mm256_zeroall();
    for (int it = 0; it < groups_4; it++) {
        __m256d a1 = _mm256_load_pd(u + 4 * it);
        __m256d b1 = _mm256_load_pd(v + 4 * it);
        singleLane = _mm256_add_pd(singleLane, _mm256_mul_pd(a1, b1));
    }
    singleLane = _mm256_hadd_pd(singleLane, singleLane);
    singleLane = _mm256_hadd_pd(singleLane, singleLane);
    singleLane = _mm256_hadd_pd(singleLane, singleLane);
    _mm256_storeu_pd(&acc[0], singleLane); // we have the answer in acc[0] as we have already done the horizontal add

    // Add last few after multiples of 8
    if (groups_1) {
        for (unsigned int i = groups_4 * 4; i < size; i++) {
            acc[0] += u[i] * v[i];
        }
    }
    return acc[0];
}

double dotprod_avx_step(const double* const a, const double* const b) {
    double sum;
    __m256d apd = _mm256_loadu_pd(&a[0]);
    __m256d bpd = _mm256_loadu_pd(&b[0]);
    __m256d cpd = _mm256_mul_pd(apd, bpd);
    __m256d hsum = _mm256_add_pd(cpd,
        _mm256_permute2f128_pd(cpd, cpd, 0x1));
    _mm_store_sd(&sum,
        _mm_hadd_pd(
            _mm256_castpd256_pd128(hsum),
            _mm256_castpd256_pd128(hsum)));
    return sum;
}

double dotprod_avx(double* a, double* b, int begin, int end) {
    double sum = 0;

    for (int i = begin; i + 4 <= end; i += 4) {
        sum += dotprod_avx_step(a + i, b + i);
    }

    for (int i = end % 4; i > 0; --i) {
        sum += a[end - i] * b[end - 1];
    }
    return sum;
}

void subtr_avx_step(double const * const a, double* b, double* c) {
    __m256d apd = _mm256_loadu_pd(a);
    __m256d bpd = _mm256_loadu_pd(b);
    __m256d cpd = _mm256_sub_pd(apd, bpd);
    _mm256_store_pd(c, cpd);
}

void subtr_avx(double const * const a, double* b, double* c, int n) {
    for (int i = 0; i + 4 <= n; i += 4) {
        subtr_avx_step(a + i, b + i, c + i);
    }

    for (int i = n % 4; i > 0; --i) {
        c[n - i] = a[n - i] - b[n - 1];
    }
}

inline Matrix cholesky_vectorized(Matrix &in, unsigned int N_sub) {
    Matrix L(in, N_sub);
    double* sum_buff = new double[N_sub];

    for (int i = 0; i < N_sub; ++i) {
        for (int j = 0; j < (i + 1); ++j) {
            sum_buff[j] = dotprod_avx(&L[i], &L[j], 0, j);
        }

        subtr_avx(&in[i], sum_buff, &L[i], i + 1);

        L(i,i) = std::sqrt(L(i,i));
        
        for (int j = 0; j < i; ++j) {
            L(i,j) = 1.0 / L(j,j) * L(i,j);
        }
    }

    delete[] sum_buff;
    return L;
}

// // Adapted form https://github.com/pashminacameron/optimization_examples/blob/master/cpp_cholesky/cholesky.hpp 
// inline Matrix cholesky_vectorized(Matrix const &in, unsigned int N_sub) {
//     Matrix L(in, N_sub);

//     for (unsigned int j = 0; j < N_sub; j++){
//         //float sum = 0;
//         //for (int k = 0; k < j; k++)
//         //	sum += m_chol(j, k) * m_chol(j, k);
        
//         L(j, j) = std::sqrt( L(j, j) - sum_2_vec_product_avx<data_t>(&L[j], &L[j], j) );
//         //L(j, j) = std::sqrt(in(j, j) - sum_2_vec_product_avx<data_t>(&L.data[j*N_sub], &L.data[j*N_sub], j));
//         //m_chol(j, j) = std::sqrt(m_chol(j, j) - sum2VecProductWrapper(&m_chol.data[j*stride], &m_chol.data[j*stride], j, m_LLt_Impl));

//         data_t invDiag = 1.0 / L(j, j);
//         for (unsigned int i = j + 1; i < N_sub; i++) {	
//             // i > j
//             //float sum = 0;
//             //for (int k = 0; k < j; k++)
//             //	sum += m_chol(i, k) * m_chol(j, k);
//             //m_chol(i, j) = invDiag * (m_chol(i, j) - sum);

//             L(i, j) = invDiag * (L(i, j) - sum_2_vec_product_avx<data_t>(&L[i], &L[j], j));
//         }
//     }

//     return L;
// }

inline Matrix cholesky_vectorized(Matrix &in) { return cholesky_vectorized(in, in.size()); }

inline Matrix cholesky(Matrix const &in) {return cholesky(in, in.size()); }

inline data_t log_det_from_cholesky(Matrix const &L) {
    data_t det = 0;

    for (size_t i = 0; i < L.size(); ++i) {
        det += std::log(L(i,i));
    }

    return 2*det;
}

inline data_t log_det_vectorized(Matrix &mat, unsigned int N_sub) {
    Matrix L = cholesky_vectorized(mat, N_sub);
    return log_det_from_cholesky(L);
}

inline data_t log_det_vectorized(Matrix &mat) {
    return log_det_vectorized(mat, mat.size());
}

inline data_t log_det(Matrix const &mat, unsigned int N_sub) {
    Matrix L = cholesky(mat, N_sub);
    return log_det_from_cholesky(L);
}

inline data_t log_det(Matrix const &mat) {
    return log_det(mat, mat.size());
}

#endif