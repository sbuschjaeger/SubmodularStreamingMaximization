#ifndef INFORMATIVE_VECTOR_MACHINE_H
#define INFORMATIVE_VECTOR_MACHINE_H

#include <mutex>
#include <vector>
#include <functional>
#include <math.h>
#include <cassert>
#include "DataTypeHandling.h"
#include "SubmodularFunction.h"
#include "kernels/Kernel.h"

class Matrix {
private:
    unsigned int N;
    // There are two main reasons why we use an std::vector here instead of a raw pointer
    //  (1) std::vector is the more modern c++ style and raw pointers are somewhat discuraged (see next comment)
    //  (2) It turns our there is a good reason why we should not use raw pointers. It makes it really difficult to implement appropriate copy / move constructors. I would sometimes run into weird memory issues, because the compiler provided an implicit copy / move c'tor. Of course it would be possible to properly implement move/copy/assignment operators (rule of 0/3/5 https://en.cppreference.com/w/cpp/language/rule_of_three) but thats more work than I need. 
    std::vector<data_t> val;

public:

    // Matrix(Matrix const &other) {
    //     N = other.N;
    //     val = new data_t[N*N];
    //     for (unsigned int i = 0; i < N; ++i) {
    //         val[i] = other.val[i];
    //     }
    // }

    /**
     * @brief  Copies the upper left N_sub x N_sub matrix from other into the new object. Caller has to make sure that N_sub <= other.size()
     * @note   
     * @param  &other: 
     * @param  N_sub: 
     * @retval 
     */
    Matrix(Matrix const &other, unsigned int N_sub) : N(N_sub), val(N_sub * N_sub) {
        for (unsigned int i = 0; i < N_sub; ++i) {
            for (unsigned int j = 0; j < N_sub; ++j) {
                this->operator()(i, j) = other(i,j);
            }
        }
    }

    Matrix(unsigned int _size) : N(_size), val(_size * _size, 0){}

    ~Matrix() {
    }

    // // TODO THIS IS A DEEP COPY AND NO MOVE OR ANYTHING LIKE THAT!
    // Matrix& operator=(Matrix const& other) {
    //     if (this != &other) { // protect against invalid self-assignment
    //         // 1: allocate new memory and copy the elements
    //         data_t* new_val = new data_t[other.N * other.N];
    //         std::copy(other.val, other.val + other.N, new_val);

    //         // 2: deallocate old memory
    //         delete[] val;

    //         // 3: assign the new memory to the object
    //         val = new_val;
    //         N = other.N;
    //     }
    //     // by convention, always return *this
    //     return *this;
    // }

    inline unsigned int size() const { return N; }

    void replace_row(unsigned int row, data_t const * const data) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(i, row) = data[i];
        }
    }

    void replace_column(unsigned int col, data_t const * const data) {
        for (unsigned int i = 0; i < N; ++i) {
            this->operator()(col, i) = data[i];
        }
    }

    void rank_one_update(unsigned int row, data_t const * const data) {
        for (unsigned int i = 0; i < N; ++i) {
            if (row == i) {
                this->operator()(i,i) += data[i];
            } else {
                this->operator()(i,row) += data[i];
                this->operator()(row,i) += data[i];
            }
        }
    }

    data_t & operator()(int i, int j) { return val[i*N+j]; }
    data_t operator()(int i, int j) const { return val[i*N+j]; }
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

Matrix cholesky(Matrix const &in, unsigned int N_sub) {
    Matrix L(in, N_sub);

    for (unsigned int j = 0; j < N_sub; ++j) {
        data_t sum = 0.0;

        for (unsigned int k = 0; k < j; ++k) {
            sum += L(j,k)*L(j,k);
            //sum += pOut[j * N + k] * pOut[j * N + k];
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

inline Matrix cholesky(Matrix const &in) {
    return cholesky(in, in.size());
}

inline data_t log_det_from_cholesky(Matrix const &L) {
    data_t det = 0;
    // TODO: THIS CAN BE IMPROVED / MAYBE THERE IS A MKL/LAPACK FUNCTION FOR THIS
    for (size_t i = 0; i < L.size(); ++i) {
        //if (L[i * N + i] == 0) det += std::numeric_limits<T>::max();
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


class IVM : public SubmodularFunction {
protected:

    inline Matrix compute_kernel(std::vector<std::vector<data_t>> const &X) const {
        unsigned int K = X.size();
        Matrix mat(K);

        for (unsigned int i = 0; i < K; ++i) {
            for (unsigned int j = i; j < K; ++j) {
                data_t kval = kernel->operator()(X[i], X[j]);
                if (i == j) {
                    mat(i,j) = 1.0 + kval / std::pow(1.0, 2.0);
                } else {
                    mat(i,j) = kval / std::pow(1.0, 2.0);
                    mat(j,i) = kval / std::pow(1.0, 2.0);
                }
            }
        }

        // TODO CHECK IF THIS USES MOVE
        return mat;
    }

    std::shared_ptr<Kernel> kernel;
    data_t sigma;

public:
    IVM(Kernel const &kernel, data_t sigma) : kernel(kernel.clone()), sigma(sigma) {}

    IVM(std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> kernel, data_t sigma) 
        : kernel(std::unique_ptr<Kernel>(new KernelWrapper(kernel))), sigma(sigma) {
    }

    data_t peek(std::vector<std::vector<data_t>> const& cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
        std::vector<std::vector<data_t>> tmp(cur_solution);

        if (pos >= cur_solution.size()) {
            tmp.push_back(x);
        } else {
            tmp[pos] = x;
        }

        data_t ftmp = this->operator()(tmp);
        return ftmp;
    } 

    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {}

    data_t operator()(std::vector<std::vector<data_t>> const &X) const override {
        // This is the most basic implementations which recomputes everything with each call
        // I would not use this for any real-world problems. 
        Matrix kernel_mat = compute_kernel(X);
        return log_det(kernel_mat);
    }

    std::shared_ptr<SubmodularFunction> clone() const override {
        return std::make_shared<IVM>(*kernel, sigma);
    }

    ~IVM() {}
};

#endif // INFORMATIVE_VECTOR_MACHINE_H

