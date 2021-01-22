#ifndef SUBMODULAROPTIMIZER_H
#define SUBMODULAROPTIMIZER_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <cassert>
#include <memory>
#include <optional>

#include "SubmodularFunction.h"

/**
 * @brief  Interface class which every optimizer should implement. Each optimizer must offer a next() and fit() function. However, if a certain optimizer does not support streaming (`next') or batch (`fit') processing it is okay to throw an exeception with an appropriate message. This class already offers a member to store the best solution (`solution') and its function value (`fval`) including getter functions. You can access the function to be maximized via `f` which is a shared pointer (and thus there is no need for explicit delete in the destructor). Please make sure to set `is_fitted` after the fit / next has been called. Please make sure that you use the `peek` and `update` function of the SubmodularFunction correctly. Always call `peek` if you want to know the function value if you would add a new element to the current solution and call `update` if you know which element to add to the current solution. See SubmodularFunction.h for more details.
 * @note   
 * @retval None
 */
class SubmodularOptimizer {
private:
    
protected:
    // The cardinality constraint you of the optimization problem, that is the number of items selected.
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

    // true if fit() or next() has been called.
    bool is_fitted;

public:
    // The current solution of this optimizer
    std::vector<std::vector<data_t>> solution;
    std::vector<idx_t> ids;

    // The current function value of this optimizer
    data_t fval;

    /**
     * @brief  Creates a submodular optimizer object. 
     * @note   
     * @param  K: The cardinality constraint K of the optimization problem or differently put the number of items to be selected.
     * @param  f: The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
     * @retval A new SubmodularOptimizer object.  
     */
    SubmodularOptimizer(unsigned int K, SubmodularFunction & f) 
        : K(K), f(f.clone()) {
        is_fitted = false;
        fval = 0;
        // assert(("K should at-least be 1 or greater.", K >= 1));
    }

    /**
     * @brief  Creates a submodular optimizer object. 
     * @note   
     * @param  K: The cardinality constraint K of the optimization problem or differently put the number of items to be selected.
     * @param  f: The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state. 
     * @retval A new SubmodularOptimizer object.  
     */
    SubmodularOptimizer(unsigned int K, std::function<data_t (std::vector<std::vector<data_t>> const &)> f) 
        : K(K), f(std::unique_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f))) {
        is_fitted = false;
        fval = 0;
        // assert(("K should at-least be 1 or greater.", K >= 1));
    }

    /**
     * @brief  Find a solution given the entire data set. 
     * @note   
     * @param  X: A constant reference to the entire data set
     * @param iterations: Maximum number of iterations over the entire data-set (default = 1). Tries to select exactly K elements by iterating multiple 
     *                    times over the entire dataset, but at most iterations times and at-least once. Early exits once K elements are found and at-least one iteration is completed. 
     * @retval None
     */
    virtual void fit(std::vector<std::vector<data_t>> const & X, std::vector<idx_t> const & ids, unsigned int iterations = 1) {
        assert(X.size() == ids.size());

        for (unsigned int i = 0; i < iterations; ++i) {
            for (unsigned int j = 0; j < X.size(); ++j) {
                next(X[j], ids[j]);
                // It is verly likely that the lower threshold sieves will fill up early and thus we will probably find a full sieve early on
                // This likely results in a very bad function value. However, only iterating once over the entire data-set may lead to a very
                // weird situation where no sieve is full yet (e.g. for very small datasets). Thus, we re-iterate as often as needed and early
                // exit if we have seen every item at-least once
                if (solution.size() == K && i > 0) {
                    return;
                }
            }
        }
    }

    /**
     * @brief  Find a solution given the entire data set. 
     * @note   
     * @param  X: A constant reference to the entire data set
     * @param iterations: Maximum number of iterations over the entire data-set (default = 1). Tries to select exactly K elements by iterating multiple 
     *                    times over the entire dataset, but at most iterations times and at-least once. Early exits once K elements are found and at-least one iteration is completed. 
     * @retval None
     */
    virtual void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
        for (unsigned int i = 0; i < iterations; ++i) {
            for (auto &x : X) {
                next(x);
                // It is verly likely that the lower threshold sieves will fill up early and thus we will probably find a full sieve early on
                // This likely results in a very bad function value. However, only iterating once over the entire data-set may lead to a very
                // weird situation where no sieve is full yet (e.g. for very small datasets). Thus, we re-iterate as often as needed and early
                // exit if we have seen every item at-least once
                if (solution.size() == K && i > 0) {
                    return;
                }
            }
        }
    }

    /**
     * @brief  Consume the next object in the data stream. This may throw an exception if the optimizer does not support streaming.
     * @note   
     * @param  x: A constant reference to the next object on the stream.
     * @retval None
     */
    virtual void next(std::vector<data_t> const &x, std::optional<idx_t> const id = std::nullopt) = 0;


    /**
     * @brief  Return the current solution.
     * @note   
     * @retval A const reference to the current solution.
     */
    std::vector<std::vector<data_t>>const & get_solution() const {
        if (!this->is_fitted) {
             throw std::runtime_error("Optimizer was not fitted yet! Please call fit() or next() before calling get_solution()");
        } else {
            return solution;
        }
    }
    
    std::vector<idx_t> const &get_ids() const {
        if (!this->is_fitted) {
             throw std::runtime_error("Optimizer was not fitted yet! Please call fit() or next() before calling get_ids()");
        } else {
            return ids;
        }
    }

    virtual unsigned int get_num_candidate_solutions() const {
        return 1;
    }
    
    virtual unsigned long get_num_elements_stored() const {
        return this->get_solution().size();
    }

    /**
     * @brief  Returns the current function value
     * @note   
     * @retval The current function value
     */
    data_t get_fval() const {
        return fval;
    }

    /**
     * @brief  Destroys this object
     * @note   
     * @retval None
     */
    virtual ~SubmodularOptimizer() {}
};

#endif // THREESIEVES_SUBMODULAROPTIMIZER_H
