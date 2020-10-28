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

class SubmodularOptimizer {
private:
    
protected:
    unsigned int K;
    
    /** 
     * Okay lets explain the reasoning behind this a bit more. 
     * 1):  Use SubmodularFunction as an object. This does not work because we are dealing with inheritance / virtual functions
     * 2):  Use a reference to SubmodularFunction. This works well with PyBind and has an easy interface, but breaks does not work well 
     *      together with the SubmodularFunctionWrapper if we want to allow users to pass a std::function directly.
     * 3):  Use SubmodularFunction* which would be a very "pure" and old-school approach. Should be do-able, but is not the modern c++ style
     * 4):  Use std::unique_ptr<SubmodularFunction>. This would IMHO be the best approach as it reflects our intend, that the SubmodularOptimizer owns the SubmodularFunction which it clones beforehand. This does not work well with PyBind since PyBind wants to own the memory. 
     * 5):  Use std::shared_ptr<SubmodularFunction> which is basically a modern version of 3) and works better with PyBind. 
     *
     **/
    //std::unique_ptr<SubmodularFunction> f;
    std::shared_ptr<SubmodularFunction> f;

    bool is_fitted;

public:
    //TODO: Do we want to have this public here?
    std::vector<std::vector<data_t>> solution;
    data_t fval;

    SubmodularOptimizer(unsigned int K, SubmodularFunction & f) 
        : K(K), f(f.clone()) {
        is_fitted = false;
        fval = 0;
        assert(("K should at-least be 1 or greater.", K >= 1));
    }

    SubmodularOptimizer(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) 
        : K(K), f(std::unique_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f))) {
        is_fitted = false;
        fval = 0;
        assert(("K should at-least be 1 or greater.", K >= 1));
    }

    SubmodularOptimizer(unsigned int K, std::shared_ptr<SubmodularFunction> f) : K(K), f(f) {
        is_fitted = false;
        fval = 0;
        assert(("K should at-least be 1 or greater.", K >= 1));
    }

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
    virtual ~SubmodularOptimizer() {}
};

#endif // THREESIEVES_SUBMODULAROPTIMIZER_H
