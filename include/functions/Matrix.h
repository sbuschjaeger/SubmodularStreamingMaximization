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

inline Matrix cholesky(Matrix const &in) {return cholesky(in, in.size()); }

inline data_t log_det_from_cholesky(Matrix const &L) {
    data_t det = 0;

    for (size_t i = 0; i < L.size(); ++i) {
        det += std::log(L(i,i));
    }

    return 2*det;
}

inline data_t log_det(Matrix const &mat, unsigned int N_sub) {
    Matrix L = cholesky(mat, N_sub);
    return log_det_from_cholesky(L);
}

inline data_t log_det(Matrix const &mat) {
    return log_det(mat, mat.size());
}

#endif