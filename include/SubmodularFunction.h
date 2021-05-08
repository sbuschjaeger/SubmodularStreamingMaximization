#ifndef SUBMODULARFUNCTION_H
#define SUBMODULARFUNCTION_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <cassert>

#include "DataTypeHandling.h"


/**
 * @brief  Interface class which every submodular function should implement. Is is expected by all optimizers. This interface offers a convenient way to implement stateful submodular functions. Each submodular function must offer four functions: 
 * - `operator()`
 * - `peek` 
 * - `update` 
 * - `clone` 
 * as detailed below. The SubmodularOptimizer class are expected to use `peek` whenever they ask for a function value and to use `update` whenever a new element is added to the solution. The clone function should implement a deep copy of the object. For state-less functions there is also a SubmodlarFunctonWrapper available which expects a lambda / std::function.
 * @note   
 * @retval None
 */
class SubmodularFunction {
public:
    // TODO THIS SHOULD NOT BE NEEDED. WHY DO WE HAVE THIS?!?!
    virtual data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const = 0;

    /**
     * @brief  Returns the function value if x __would__ be added at position "pos" in the current solution. If pos is greater than the number of elements in the current solution we __would__ add x the current solution. Otherwise, we __would__ replace the object at position "pos" with x. 
     * @note   
     * @param  cur_solution: The current solution.
     * @param  x: The item which we would hypothetically add to the solution.
     * @param  pos: The position at which we would add x. Note that it holds: \f$ 0 \le pos < K \f$
     * @retval The function value if we would add x to cur_solution at position pos 
     */
    virtual data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) = 0; 

    /**
     * @brief  Update the function if we add x at position "pos" to the current solution. If pos is greater than the number of elements in the current solution we add x the current solution. Otherwise, we replace the object at position "pos" with x.
     * @note   
     * @param  cur_solution: The current solution.
     * @param  x: The item which we add to the solution.
     * @param  pos: The position at which we would add x. Note that it holds: \f$ 0 \le pos < K \f$
     * @retval None
     */
    virtual void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) = 0;

    /**
     * @brief  This function returns a clone of this Submodular function. Make sure, that the new objet is a valid clone which behaves like a new object and does not reference any members of this object. Some algorithms like SieveStreaming(++) or Salsa utilize multiple optimizers in parallel each with their own unique SubmodularFunction. Moreover, to make for efficient PyBind bindings, we use clone() to give the C++ side more control over the memory.   
     * @note   
     * @retval 
     */
    virtual std::shared_ptr<SubmodularFunction> clone() const = 0;

    /**
     * @brief  Destroys this object
     * @note   
     * @retval None
     */
    virtual ~SubmodularFunction() {}
};

/**
 * @brief  A simple Wrapper class which takse a std::function and uses it to implement the SubmodularFunction interface. This is used as a convience class for the SubmodularOptimizer interface. This wrapper is meant for stateless functions so that the std::function __should not__ have / change / maintain an internal state which depends on the order of function calls. The main reason for this is, that the given std::function is likely to be moved into the member object (and not copied) which makes for very efficient code. However, some optimizers require multiple copies of the same function such as SieveStreaming(++) for multiple sub-optimizers. In this case, _all_ (sub-) optimizers reference the same object, which works fine if the function is stateless but probably breaks for stateful functions. If your submodular function requires some internal states which e.g. depend on the order of items added please consider to implement a `proper' SubmodularFunction. 
 * @note   
 * @retval None
 */
class SubmodularFunctionWrapper : public SubmodularFunction {
protected:
    // The std::function which implements the actual submodular function
    std::function<data_t (std::vector<std::vector<data_t>> const &)> f;

public:

    /**
     * @brief  Creates a new SubmodularFunction from a given std::function object. 
     * @note   
     * @param  f: The (stateless) function which implements the actual submodular function
     * @retval 
     */
    SubmodularFunctionWrapper(std::function<data_t (std::vector<std::vector<data_t>> const &)> f) : f(f) {}

    /**
     * @brief  Implements the () operator by simply delegating the call to the underlying std::function.
     * @note   
     * @param  &cur_solution: 
     * @retval 
     */
    data_t operator()(std::vector<std::vector<data_t>> const &cur_solution) const {
        return f(cur_solution);
    }

    /**
     * @brief  Implements the peek method. This copies the current solution vector to a new one, adds x at the appropriate positon and calls the ()-operator. In most cases the copy is probably not necessary (e.g. if we only append x to the current solution) which makes this code slightly inefficient. 
     * @note   
     * @param  &cur_solution: 
     * @param  &x: 
     * @param  pos: 
     * @retval 
     */
    data_t peek(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) {
        std::vector<std::vector<data_t>> tmp(cur_solution);

        if (pos >= cur_solution.size()) {
            tmp.push_back(x);
        } else {
            tmp[pos] = x;
        }

        data_t ftmp = this->operator()(tmp);
        return ftmp;
    }

    /**
     * @brief  Implements the update method. This class only wraps an std::function so it is state-less and the std::function would have to deal with any stateful behaviour. Thus, we don't do anything here.
     * @note   
     * @param  &cur_solution: 
     * @param  &x: 
     * @param  pos: 
     * @retval None
     */
    void update(std::vector<std::vector<data_t>> const &cur_solution, std::vector<data_t> const &x, unsigned int pos) {}
    
    /**
     * @brief  Implements the clone method. Note, that it is very likely that the std::function `f' has been moved into this object and similarly, we will move it into the clone as-well. This is okay, as long as `f' is a stateless function. However, if `f' has some internal state, then the other optimizers will use the __same__ function with the shared state which will probably lead to weird side-effects. In this case consider implementing a proper SubmodularFunction.  
     * @note   
     * @retval The cloned object.
     */
    std::shared_ptr<SubmodularFunction> clone() const {
        return std::shared_ptr<SubmodularFunction>(new SubmodularFunctionWrapper(f));
    }

    /**
     * @brief  Destroy the wrapper object.
     * @note   
     */
    ~SubmodularFunctionWrapper() {}
};

#endif // SUBMODULARFUNCTION_H
