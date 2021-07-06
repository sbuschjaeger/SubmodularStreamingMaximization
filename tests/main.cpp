#include <iostream>
#include <vector>
#include <math.h>
#include <map>
#include <algorithm>

#include "functions/FastIVM.h"
#include "functions/kernels/RBFKernel.h"
#include "Greedy.h"
#include "Random.h"
#include "ThreeSieves.h"
#include "Salsa.h"
#include "IndependentSetImprovement.h"
#include "SieveStreaming.h"
#include "SieveStreamingPP.h"
#include "DataTypeHandling.h"
#include "functions/IVM.h"

data_t rbf_kernel(const std::vector<data_t>& x1, const std::vector<data_t>& x2) {
    data_t distance = 0;
        if (x1 != x2) {
            distance = std::inner_product(x1.begin(), x1.end(), x2.begin(), data_t(0), 
                std::plus<data_t>(), [](data_t x,data_t y){return (y-x)*(y-x);}
            );
            distance /= 1.0;
        }
        return 1.0 * std::exp(-distance);
}

data_t poly_kernel(std::vector<data_t>const & x1, std::vector<data_t> const & x2) {
    data_t distance = 0;
    // if (x1 != x2) {
    for (unsigned int i = 0; i < x1.size(); ++i) {
        distance += x1[i]*x2[i];
    }
    // }
    return distance / static_cast<data_t>(x1.size());
}

class PolyKernel : public Kernel {
   public:
      PolyKernel() = default;

      inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const override {
            data_t distance = 0;
            // if (x1 != x2) {
            for (unsigned int i = 0; i < x1.size(); ++i) {
                distance += x1[i]*x2[i];
            }
            // }
            return distance / static_cast<data_t>(x1.size());
      }

      std::shared_ptr<Kernel> clone() const override {
         return std::shared_ptr<Kernel>(new PolyKernel());
      }
   };


inline data_t ivm(std::vector<std::vector<data_t>> const &cur_solution) {
    unsigned int K = cur_solution.size();
    Matrix kmat(K);

    for (unsigned int i = 0; i < K; ++i) {
        for (unsigned int j = i; j < K; ++j) {
            data_t kval = rbf_kernel(cur_solution[i], cur_solution[j]);
            if (i == j) {
                kmat(i,j) = 1.0 + kval / std::pow(1.0, 2.0);
            } else {
                kmat(i,j) = kval / std::pow(1.0, 2.0);
                kmat(j,i) = kval / std::pow(1.0, 2.0);
            }
        }
    }
    return log_det(kmat, cur_solution.size());
}

class FastLogDet : public SubmodularFunction {
   private:
      
   protected:
      // Number of items added so far. Required to maintain consistent access to kmat and L
      unsigned int added;

      // The kernel matrix \Sigma. 
      // See Matrix.h for more details
      Matrix kmat;

   public:

      FastLogDet(unsigned int K) : kmat(K+1) {
         added = 0;
      }

      data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
         if (pos >= added) {
               // Peek function value for last line

               for (unsigned int i = 0; i < added; ++i) {
                  data_t kval = rbf_kernel(cur_solution[i], x);

                  kmat(i, added) = kval;
                  kmat(added, i) = kval;
               }
               data_t kval = rbf_kernel(x, x);
               kmat(added, added) = 1.0 + kval;
               return log_det(kmat, added+1);
         } else {
               Matrix tmp(kmat, added);
               for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                  if (i == pos) {
                     data_t kval = rbf_kernel(x, x);
                     tmp(pos, pos) = 1.0 + kval;
                  } else {
                     data_t kval = rbf_kernel(cur_solution[i], x);
                     tmp(i, pos) = kval;
                     tmp(pos, i) = kval;
                  }
               }

               return log_det(tmp, added);
         }
      }

      void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) override {
         if (pos >= added) {
               peek(cur_solution, x, pos);
               added++;
         } else {
               for (unsigned int i = 0; i < cur_solution.size(); ++i) {
                  if (i == pos) {
                     data_t kval = rbf_kernel(x, x);
                     kmat(pos, pos) = 1.0 + kval;
                  } else {
                     data_t kval = rbf_kernel(cur_solution[i], x);
                     kmat(i, pos) = kval;
                     kmat(pos, i) = kval;
                  }
               }
         }

      }

      data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const override {
         return log_det(kmat);
      }

      std::shared_ptr<SubmodularFunction> clone() const override {
         // We want to store k elements. To allow for efficient peeking we will reserve space for K + 1 elements in kmat and L. 
         // Thus we need to call the constructor with one element less
         return std::make_shared<FastLogDet>(kmat.size() - 1);
      }
   };

bool check_is_equal(std::vector<std::vector<data_t>> const &X1, std::vector<std::vector<data_t>> const &X2) {
    if (X1.size() != X2.size()) return false;

    for (unsigned int i = 0; i < X1.size(); ++i) {
        if (X1[i].size() != X2[i].size()) return false;
        for (unsigned int j = 0; j < X1[i].size(); ++j) {
            if (X1[i][j] != X2[i][j]) return false;
        }
    }
    return true;
}

int main() {
    // "Generate" some test data
    std::vector<std::vector<data_t>> X = {
        {0.0,0.0},
        {1.0,1.0},
        {0.0,1.0},
        {0.0,0.0},
        {1.0,1.0},
        {0.0,1.0},
        {0.0,0.0},
        {1.0,1.0},
        {0.0,1.0},
        {0.0,0.0},
        {1.0,1.0},
        {0.0,1.0}, 
    };    

    // "Generate" some unique ids for each data item
    std::vector<idx_t> ids = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    };

    // Target solutions depending on the kernel
    std::vector<std::vector<data_t>> target_rbf = {
        {0.0,0.0},
        {1.0,1.0},
        {0.0,1.0}
    };

    std::vector<std::vector<data_t>> target_poly = {
        {1.0,1.0},
        {1.0,1.0},
        {0.0,1.0}
    };
    
    std::sort(target_rbf.begin(), target_rbf.end());
    std::sort(target_poly.begin(), target_poly.end());

    auto K = 3;
    // Define all the kernel / submodular function combinations
    FastIVM ivm_rbf(K, RBFKernel(), 1.0);
    FastIVM ivm_custom_kernel_class(K, PolyKernel(), 1.0);
    FastIVM ivm_custom_kernel_function(K, poly_kernel, 1.0);

    FastLogDet ivm_custom_class(K);
    auto ivm_custom_function = ivm;

    std::map<std::string, SubmodularOptimizer*> optimizers;

    /* GREEDY */
    optimizers["Greedy with IVM + RBF"] = new Greedy(K, ivm_rbf);
    optimizers["Greedy with IVM + poly kernel class"] = new Greedy(K, ivm_custom_kernel_class);
    optimizers["Greedy with IVM + poly kernel function"] = new Greedy(K, ivm_custom_kernel_function);
    optimizers["Greedy with custom IVM class"] = new Greedy(K, ivm_custom_class);
    optimizers["Greedy with custom IVM function"] = new Greedy(K, ivm_custom_function);

    /* Random */
    optimizers["Random with IVM + RBF"] = new Random(K, ivm_rbf, 12345);
    optimizers["Random with IVM + poly kernel class"] = new Random(K, ivm_custom_kernel_class, 22222);
    optimizers["Random with IVM + poly kernel function"] = new Random(K, ivm_custom_kernel_function, 22222);
    optimizers["Random with custom IVM class"] = new Random(K, ivm_custom_class, 12345);
    optimizers["Random with custom IVM function"] = new Random(K, ivm_custom_function,12345);

    /* IndependentSetImprovement */ 
    optimizers["IndependentSetImprovement with IVM + RBF"] = new IndependentSetImprovement(K, ivm_rbf);
    optimizers["IndependentSetImprovement with IVM + poly kernel class"] = new IndependentSetImprovement(K, ivm_custom_kernel_class);
    optimizers["IndependentSetImprovement with IVM + poly kernel function"] = new IndependentSetImprovement(K, ivm_custom_kernel_function);
    optimizers["IndependentSetImprovement with custom IVM class"] = new IndependentSetImprovement(K, ivm_custom_class);
    optimizers["IndependentSetImprovement with custom IVM function"] = new IndependentSetImprovement(K, ivm_custom_function);

    /* SieveStreaming */ 
    optimizers["SieveStreaming with IVM + RBF"] = new SieveStreaming(K, ivm_rbf, 1.0, 0.1);
    optimizers["SieveStreaming with IVM + poly kernel class"] = new SieveStreaming(K, ivm_custom_kernel_class, 1.0, 0.5);
    optimizers["SieveStreaming with IVM + poly kernel function"] = new SieveStreaming(K, ivm_custom_kernel_function, 1.0, 0.5);
    optimizers["SieveStreaming with custom IVM class"] = new SieveStreaming(K, ivm_custom_class, 1.0, 0.1);
    optimizers["SieveStreaming with custom IVM function"] = new SieveStreaming(K, ivm_custom_function, 1.0, 0.1);

    /* SieveStreamingPP */ 
    optimizers["SieveStreamingPP with IVM + RBF"] = new SieveStreamingPP(K, ivm_rbf, 1.0, 0.1);
    optimizers["SieveStreamingPP with IVM + poly kernel class"] = new SieveStreamingPP(K, ivm_custom_kernel_class, 1.0, 0.1);
    optimizers["SieveStreamingPP with IVM + poly kernel function"] = new SieveStreamingPP(K, ivm_custom_kernel_function, 1.0, 0.1);
    optimizers["SieveStreamingPP with custom IVM class"] = new SieveStreamingPP(K, ivm_custom_class, 1.0, 0.1);
    optimizers["SieveStreamingPP with custom IVM function"] = new SieveStreamingPP(K, ivm_custom_function, 1.0, 0.1);

    /* Salsa */ 
    optimizers["Salsa with IVM + RBF"] = new Salsa(K, ivm_rbf, 1.0, 0.1);
    optimizers["Salsa with IVM + poly kernel class"] = new Salsa(K, ivm_custom_kernel_class, 1.0, 0.1);
    optimizers["Salsa with IVM + poly kernel function"] = new Salsa(K, ivm_custom_kernel_function, 1.0, 0.1);
    optimizers["Salsa with custom IVM class"] = new Salsa(K, ivm_custom_class, 1.0, 0.1);
    optimizers["Salsa with custom IVM function"] = new Salsa(K, ivm_custom_function, 1.0, 0.1);

    /* ThreeSieves */ 
    optimizers["ThreeSieves with IVM + RBF"] = new ThreeSieves(K, ivm_rbf, 1.0, 0.1, "sieve",5);
    optimizers["ThreeSieves with IVM + poly kernel class"] = new ThreeSieves(K, ivm_custom_kernel_class, 1.0, 0.01, "sieve",1);
    optimizers["ThreeSieves with IVM + poly kernel function"] = new ThreeSieves(K, ivm_custom_kernel_function, 1.0, 0.01, "sieve",1);
    optimizers["ThreeSieves with custom IVM class"] = new ThreeSieves(K, ivm_custom_class, 1.0, 0.1, "sieve",5);
    optimizers["ThreeSieves with custom IVM function"] = new ThreeSieves(K, ivm_custom_function, 1.0, 0.1, "sieve",5);

    bool failed = false;
    for (auto& [name, opt] : optimizers) {
        opt->fit(X);
        auto fval = opt->get_fval();
        auto solution = opt->get_solution();
        std::sort(solution.begin(), solution.end());

        std::cout << "Testing " << name << std::endl;
        std::cout << "\tfval is " << fval << std::endl;

        if (name.find("poly") != std::string::npos) {
            if (!check_is_equal(solution, target_poly)) {
                failed = true;
                std::cout << "\tTEST FAILED. Solution does not match target solution!" << std::endl;
                std::cout << "\tSolution was:" << std::endl;
                for (const auto &s : solution) {
                    std::cout << "\t\t[ ";
                    for (const auto &e: s) {
                        std::cout << e << " ";
                    }
                    std::cout << "]" << std::endl;
                }
                std::cout << "\t...but target was:" << std::endl;
                for (const auto &s : target_poly) {
                    std::cout << "\t\t[ ";
                    for (const auto &e: s) {
                        std::cout << e << " ";
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cout << "\tTEST PASSED. Solution matches target solution!" << std::endl;
            }
        } else {
            if (!check_is_equal(solution, target_rbf)) {
                failed = true;
                std::cout << "\tTEST FAILED. Solution does not match target solution!" << std::endl;
                std::cout << "\tSolution was:" << std::endl;
                for (const auto &s : solution) {
                    std::cout << "\t\t[ ";
                    for (const auto &e: s) {
                        std::cout << e << " ";
                    }
                    std::cout << "]" << std::endl;
                }
                std::cout << "\t...but target was:" << std::endl;
                for (const auto &s : target_rbf) {
                    std::cout << "\t\t[ ";
                    for (const auto &e: s) {
                        std::cout << e << " ";
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cout << "\tTEST PASSED. Solution matches target solution!" << std::endl;
            }
        }

        delete opt;
    }

    return failed == true;
    // First lets check if we compute the correct kernel and its logdet via a cholesky decomposition. 
    // Matrix m = compute_kernel(data);
    // std::cout << "Matrix is: " << std::endl;
    // std::cout << to_string(m) << std::endl;
    
    // Matrix L = cholesky(m);
    // std::cout << "Regular L: " << std::endl;
    // std::cout << to_string(L) << std::endl;
    // std::cout << "log det from chol is: " << log_det_from_cholesky(L) << std::endl;
    // std::cout << "log det directly is: " << log_det(m) << std::endl << std::endl;

    // // Lets select a summary of size K = 5
    // unsigned int K = 5;

    // // Two different IVM variants
    // // IVM slowIVM(RBFKernel(), 1.0);
    // FastIVM fastIVM(K, RBFKernel(), 1.0);
    
    // // Use the Greedy algorithm
    // Greedy greedy(K, fastIVM);
    // greedy.fit(data, ids);
    // std::vector<std::vector<double>> solution = greedy.get_solution();
    // std::vector<idx_t> solution_ids = greedy.get_ids();
    // double fval = greedy.get_fval();

    // std::cout << "Found a solution with fval = " << fval << std::endl;
    // for (auto x : solution) {
    //     for (auto xi : x) {
    //         std::cout << xi << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "ids = {";
    // for (auto i : solution_ids) {
    //     std::cout << i << " ";
    // }
    // std::cout << "}" << std::endl;

    // // Use the Random algorithm
    // Random random(K, fastIVM);
    // random.fit(data, ids);
    // std::vector<std::vector<double>> solution = random.get_solution();
    // std::vector<idx_t> solution_ids = random.get_ids();
    // double fval = random.get_fval();

    // std::cout << "Found a solution with fval = " << fval << std::endl;
    // for (auto x : solution) {
    //     for (auto xi : x) {
    //         std::cout << xi << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "ids = {";
    // for (auto i : solution_ids) {
    //     std::cout << i << " ";
    // }
    // std::cout << "}" << std::endl;

    // // Use the SieveStreaming algorithm, but now with a lambda function instead of a fastIVM object
    // SieveStreaming sieve(K, [](std::vector<std::vector<data_t>> const &X){
    //     unsigned int K = X.size();
    //     data_t * kmat = new data_t[K*K];

    //     for (unsigned int i = 0; i < K; ++i) {
    //         for (unsigned int j = i; j < K; ++j) {
    //             data_t kval = kernel(X[i], X[j]);
    //             if (i == j) {
    //                 kmat[i+i*K] = 1.0 + kval / std::pow(1.0, 2.0);
    //             } else {
    //                 kmat[i+j*K] = kval / std::pow(1.0, 2.0);
    //                 kmat[j+i*K] = kval / std::pow(1.0, 2.0);
    //             }
    //         }
    //     }

    //     data_t fval = logDet(kmat, X.size(), X.size());
    //     delete [] kmat;
    //     return fval;
    // }, 2.0, 0.1);
    // sieve.fit(data);
    // std::vector<std::vector<double>> solution = sieve.get_solution();
    // std::vector<idx_t> solution_ids = sieve.get_ids();
    // double fval = sieve.get_fval();

    // std::cout << "Found a solution with fval = " << fval << std::endl;
    // for (auto x : solution) {
    //     for (auto xi : x) {
    //         std::cout << xi << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "ids = {";
    // for (auto i : solution_ids) {
    //     std::cout << i << " ";
    // }
    // std::cout << "}" << std::endl;
    //return 0;
}

// Compute the kernel matrix
// inline Matrix compute_kernel(std::vector<std::vector<data_t>> const &X) {
//     unsigned int K = X.size();
//     Matrix mat(K);

//     for (unsigned int i = 0; i < K; ++i) {
//         for (unsigned int j = i; j < K; ++j) {
//             data_t kval = kernel(X[i], X[j]);
//             if (i == j) {
//                 mat(i,j) = 1.0 + kval / std::pow(1.0, 2.0);
//             } else {
//                 mat(i,j) = kval / std::pow(1.0, 2.0);
//                 mat(j,i) = kval / std::pow(1.0, 2.0);
//             }
//         }
//     }

//     return mat;
// }