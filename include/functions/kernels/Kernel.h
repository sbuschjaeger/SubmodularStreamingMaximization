#ifndef KERNEL_H
#define KERNEL_H

#include <cassert>
#include "DataTypeHandling.h"

/**
 * @brief  Virtual base class for Kernels. Usually, I would try to make this a little easier / more accessible 
 * and use raw function pointer / std::function for kernels. However, sometimes kernels have parameters or may hold a state. Thus I decided to use (simple) classes. Do circumvent writing new classes for each kernel, you can use the KernelWrapper to wrap functions / lambdas into this object. 
 */
class Kernel {

public:

    /**
     * @brief  Evaluates the kernel on the two given parameters x1, x2 
     * @note   
     * @param  x1: The first parameter of the kernel.
     * @param  x2: The second parameter of the kernel.
     * @retval Returns the kernel value of the parameter. 
     */
    virtual inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const = 0;

    /**
     * @brief  Clones the current kernel object. 
     * @note   Clones should be a deep copy of the object, because a SubmodularOptimizer might generate multiple copies of this kernel if required. 
     * @retval A shared pointer pointing to the copy. 
     */
    virtual std::shared_ptr<Kernel> clone() const = 0;

    /**
     * @brief  Destroys the current kernel.
     */
    virtual ~Kernel() {}
};

/**
 * @brief  A simple wrapper, which wraps a `std::function' into the kernel object. This allows us to use lambdas / std::functions instead of writing a new class for a new Kernel.  For example:
        KernelWrapper kernel([](const std::vector<data_t>& x1, const std::vector<data_t>& x2) {
            data_t distance = 0;
            if (x1 != x2) {
                for (unsigned int i = 0; i < x1.size(); ++i) {
                    distance += (x1[i]-x2[i])*(x1[i]-x2[i]);
                }
                distance /= 1.0;
            }
            return 1.0 * std::exp(-distance);
        })
 */
class KernelWrapper : public Kernel {
protected:
    /**
     * @brief  The wrapped function.
     * @note   
     * @param  &: First parameter for evaluation.
     * @param  &: Second parameter for evaluation.
     * @retval The computed kernel value between both parameters.
     */
    std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> f;

public:

    /**
     * @brief  Creates a new KernelWrapper object.
     * @note   The supplied std::function is moved into this wrapper. There is no copy involved. 
     * @param  f: The std::function to be wrapped.  
     */
    KernelWrapper(std::function<data_t (std::vector<data_t> const &, std::vector<data_t> const &)> f) : f(f) {}

    /**
     * @brief  Evaluates the wrapped kernel on the given parameters. 
     * @note   
     * @param  x1: First parameter for evaluation.
     * @param  x2: Second parameter for evaluation.
     * @retval The evaluated kernel value on the given parameters.
     */
    inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const override {
        return f(x1, x2);
    }

    /**
     * @brief  Clones this objet. 
     * @note   This is _not_ a deep copy. The internal std::function is moved into the new object 
     * @retval A pointer to the cloned object.
     */
    std::shared_ptr<Kernel> clone() const override {
        return std::shared_ptr<Kernel>(new KernelWrapper(f));
    }

};

#endif // RBF_KERNEL_H
