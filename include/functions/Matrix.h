#ifndef MATRIX_H
#define MATRIX_H

#include <immintrin.h>
#include <vector>

#include "DataTypeHandling.h"

/**
 * @brief  This is a simple Matrix class for quadratic \f$ N \times N \f$ matrices. The Matrix is implemented with a 1d (column major) `std::vector`. There are also some linear algebra functions available
 * @note   
 * @retval None
 */
class Matrix {
private:

    // The size of the matrix
    unsigned int N;

    // There are two main reasons why we use an std::vector here instead of a raw pointer
    //  (1) std::vector is the more modern c++ style and raw pointers are somewhat discouraged (see next comment)
    //  (2) It turns our there is a good reason why we should not use raw pointers. It makes it really difficult to implement appropriate copy / move constructors. I would sometimes run into weird memory issues, because the compiler provided an implicit copy / move c'tor. Of course it would be possible to properly implement move/copy/assignment operators (rule of 0/3/5 https://en.cppreference.com/w/cpp/language/rule_of_three) but that's more work than I need. 
    std::vector<data_t> data;

public:

    /**
     * @brief  Copies the upper left N_sub x N_sub matrix from other into the new object. Caller has to make sure that N_sub <= other.size()
     * @note   
     * @param  &other: The matrix from which we want to copy entries
     * @param  N_sub: The size of the sub matrix. Caller has to make sure that N_sub <= other.size()
     * @retval A newly constructed N_sub x N_sub Matrix object.
     */
    Matrix(Matrix const &other, unsigned int N_sub) : N(N_sub), data(N_sub * N_sub) {
        for (unsigned int i = 0; i < N_sub; ++i) {
            for (unsigned int j = 0; j < N_sub; ++j) {
                this->operator()(i, j) = other(i,j);
            }
        }
    }

    /**
     * @brief  Creates a new _size x _size matrix. The matrix elements are initialized with zeros.
     * @note   
     * @param  _size: The number of rows / columns of the matrix.
     * @retval The newly created object.
     */
    Matrix(unsigned int _size) : N(_size), data(_size * _size, 0){}

    /**
     * @brief  Destroys the current matrix object.
     * @note   
     * @retval 
     */
    ~Matrix() { }

    /**
     * @brief  Returns the number of row / columns of the matrix
     * @note   
     * @retval 
     */
    inline unsigned int size() const { return N; }

    /**
     * @brief  Replaces the row at position row with the given vector. Caller has to make sure that row < N and that x has at-least N elements.
     * @note   There are no safety checks performed.
     * @param  row: The row which should be replaced.
     * @param  x: The vector which the row should be replaced with
     */
    void replace_row(unsigned int row, data_t const * const x) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(i, row) = x[i];
        }
    }

    /**
     * @brief  Replaces the column at position col with the given vector. Caller has to make sure that col < N and that x has at-least N elements.
     * @note   There are no safety checks performed.
     * @param  col: The column which should be replaced.
     * @param  x: The vector which the row should be replaced with
     */
    void replace_column(unsigned int col, data_t const * const x) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(col, i) = x[i];
        }
    }

    /**
     * @brief  Adds the given vector x to the j-th row and column of the the matrix. Caller has to make sure that j < N and that x has at-least N elements.
     * @note   There are no safety checks performed.
     * @param  j: The column / row to manipulate
     * @param  x: The vector to be added to the row and column
     * @retval None
     */
    void rank_one_update(unsigned int j, data_t const * const x) {
        for (unsigned int i = 0; i < N; ++i) {
            if (j == i) {
                this->operator()(i,i) += x[i];
            } else {
                this->operator()(i,j) += x[i];
                this->operator()(j,i) += x[i];
            }
        }
    }

    /**
     * @brief  Access the i-th row of the matrix. Caller has to make sure that i < N.
     * @note   There are no safety checks performed.
     * @param  i:  The row to be accessed
     * @retval A reference to the i-th row
     */
    data_t & operator [](int i) {return  data[i*N];}

    /**
     * @brief  Access the i-th row of the matrix. Caller has to make sure that i < N.
     * @note   There are no safety checks performed.
     * @param  i:  The row to be accessed
     * @retval A reference to the i-th row
     */
    data_t operator [](int i) const {return data[i*N];}

    /**
     * @brief  Access the i-th row and j-th column of the matrix. Caller has to make sure that i, j < N.
     * @note   There are no safety checks performed.
     * @param  i:  The row to be accessed
     * @param  j:  The column to be accessed
     * @retval A reference to the (i,j) entry of the matrix
     */
    data_t & operator()(int i, int j) { return data[i*N+j]; }

    /**
     * @brief  Access the i-th row and j-th column of the matrix. Caller has to make sure that i, j < N.
     * @note   There are no safety checks performed.
     * @param  i:  The row to be accessed
     * @param  j:  The column to be accessed
     * @retval A reference to the (i,j) entry of the matrix
     */
    data_t operator()(int i, int j) const { return data[i*N+j]; }
};

/**
 * @brief  Converts the given (sub-)matrix into a python / numpy compatible string, e.g. you can copy this string directly into the interactive python console for debugging if necessary. If you want to print the entire matrix supply N_sub = N.
 * @note   
 * @param  &mat: The matrix which should ne converted to a string.
 * @param  N_sub: The N_sub x N_sub matrix which should be printed. The caller has to make sure that N_sub <= N. If you want to print the entire matrix supply N_sub = N.
 * @retval A string representation of the sub-matrix
 */
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

/**
 * @brief  Converts the given matrix into a python / numpy compatible string, e.g. you can copy this string directly into the interactive python console for debugging if necessary. 
 * @note   
 * @param  &mat: The matrix which should ne converted to a string. 
 * @retval A string representation of the matrix
 */
inline std::string to_string(Matrix const &mat) {
    return to_string(mat,mat.size());
}

/**
 * @brief  Computes the choleksy decomposition of the N_sub x N_sub sub matrix and returns the lower triangular matrix L with LL^T = in.
 * @note   
 * @param  &in: The matrix which should be decomposed.
 * @param  N_sub: The N_sub x N_sub sub-matrix. The caller has to make sure that N_sub <= N. If you want to print the entire matrix supply N_sub = N.
 * @retval Returns the cholesky decomposition
 */
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

/**
 * @brief  Computes the choleksy decomposition given matrix and returns the lower triangular matrix L with LL^T = in.
 * @note   
 * @param  &in: The matrix which should be decomposed.
 * @retval Returns the cholesky decomposition
 */
inline Matrix cholesky(Matrix const &in) {return cholesky(in, in.size()); }

/**
 * @brief  Computes the log-determinant from the lower triangular matrix L which previously has been computed via a cholesky decomposition
 * @note   
 * @param  &L: The lower triangular matrix with LL^T = in, where in is the original matrix
 * @retval The log-determinant of the matrix in
 */
inline data_t log_det_from_cholesky(Matrix const &L) {
    data_t det = 0;

    for (size_t i = 0; i < L.size(); ++i) {
        det += std::log(L(i,i));
    }

    return 2*det;
}

/**
 * @brief  Computes the log-determinant of the N_sub x N_sub sub-matrix of the given matrix mat  
 * @note   
 * @param  &mat: The base matrix from which the N_sub x N_sub sub-matrix is used.
 * @param  N_sub: The N_sub x N_sub sub-matrix. The caller has to make sure that N_sub <= N. If you want to use the entire matrix supply N_sub = N.
 * @retval The log-determinant of the  N_sub x N_sub sub-matrix of mat
 */
inline data_t log_det(Matrix const &mat, unsigned int N_sub) {
    Matrix L = cholesky(mat, N_sub);
    return log_det_from_cholesky(L);
}

/**
 * @brief  Computes the log-determinant of the given matrix mat  
 * @note   
 * @param  &mat: The matrix of which the log-determinant should be computed
 * @retval The log-determinant of the mat
 */
inline data_t log_det(Matrix const &mat) {
    return log_det(mat, mat.size());
}

#endif