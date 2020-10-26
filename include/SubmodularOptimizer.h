#ifndef SUBMODULAROPTIMIZER_H
#define SUBMODULAROPTIMIZER_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <cassert>
#include <memory>

#include "SubmodularFunction.h"

class SubmodularFunctionWrapper : public SubmodularFunction {
protected:
    std::function<data_t (std::vector<std::vector<data_t>> const &)> f;

public:

    SubmodularFunctionWrapper(std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : f(f) {
    }

    // SubmodularFunction* clone() {
    //     return new SubmodularFunctionWrapper(f);
    // }

    data_t operator()(std::vector<std::vector<data_t>> const &solution) const {
        return f(solution);
    }
};

class SubmodularOptimizer {
private:
    
protected:
    unsigned int const K;
    SubmodularFunction * wrapper = NULL;
    SubmodularFunction &f;
    // std::unique_ptr<SubmodularFunction> f;

    //std::function<data_t (std::vector<std::vector<data_t>> const &)> f;
    std::vector<std::vector<data_t>> solution;
    data_t fval;
    bool is_fitted;

public:

    // SubmodularOptimizer(unsigned int K, std::unique_ptr<SubmodularFunction> f) : K(K), f(std::move(f)) {
    //     is_fitted = false;
    //     assert(("K should at-least be 1 or greater.", K >= 1));
    // }

    //  SubmodularOptimizer(unsigned int K, data_t (std::vector<std::vector<data_t>> const &) f) 
    //     : SubmodularOptimizer(K, [f](std::vector<std::vector<data_t>> const &X){return f(X);}) {}

    // SubmodularOptimizer(unsigned int K, SubmodularFunction & f) 
    //     : SubmodularOptimizer(K, std::move(std::unique_ptr<SubmodularFunction>(f.clone()))) {
    //     is_fitted = false;
    //     assert(("K should at-least be 1 or greater.", K >= 1));
    // }

    SubmodularOptimizer(unsigned int K, SubmodularFunction & f) 
        : K(K), f(f) {
        is_fitted = false;
        assert(("K should at-least be 1 or greater.", K >= 1));
    }

    SubmodularOptimizer(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) 
        : K(K), wrapper(new SubmodularFunctionWrapper(f)), f(*wrapper) {}
        
    // SubmodularOptimizer(unsigned int K, SubmodularFunction * f) 
    //     : SubmodularOptimizer(K, std::move(std::unique_ptr<SubmodularFunction>(f))) {}

    // SubmodularOptimizer(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) 
    //     : SubmodularOptimizer(K, std::move(std::unique_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f)))) {}

    /**
     *
     * @param dataset
     * @return
     */
    virtual void fit(std::vector<std::vector<data_t>> const & X) = 0;

    /**
     *
     * @param dataset
     * @return
     */
    virtual void next(std::vector<data_t> const &x) = 0;

    /**
     *
     * @param dataset
     * @return
     */
    std::vector<std::vector<data_t>>const &  get_solution() const {
        if (!this->is_fitted) {
             throw std::runtime_error("Optimizer was not fitted yet! Please call fit() or next() before calling get_solution()");
        } else {
            return solution;
        }
    }
    
    data_t get_fval() const {
        return fval;
    }

    /**
     * Destructor.
     */
    virtual ~SubmodularOptimizer() {
        if (wrapper != NULL) {
            delete wrapper;
        }
    }
};

#endif // THREESIEVES_SUBMODULAROPTIMIZER_H
