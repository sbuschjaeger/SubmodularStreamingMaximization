#ifndef RBF_KERNEL_H
#define RBF_KERNEL_H

#include <cassert>
#include "DataTypeHandling.h"
#include "functions/kernels/Kernel.h"
/**
 *
 */
class RBFKernel : public Kernel {
private:
    data_t sigma = 1.0;
    data_t scale = 1.0;

public:
    /**
     * The default constructor for this kernel. The sigma value is 1.0 and the distance function equals to Distances::SquaredEuclidean.
     */
    RBFKernel() = default;

    /**
     * Instantiates a RBF Kernel object with given sigma.
     * @param sigma Kernel sigma value.
     */
    explicit RBFKernel(data_t sigma) : RBFKernel(sigma, 1.0) {
    }

    /**
     * Instantiates a RBF Kernel object with given sigma and a arbitrarily chosen distance function.
     * @param sigma Kernel sigma value.
     * @param l A scaling value.
     */
    RBFKernel(data_t sigma, data_t scale) : sigma(sigma), scale(scale){
        assert(("The scale of an RBF Kernel should be greater than 0!", scale > 0));
        assert(("The sigma value of an RBF Kernel should be greater than  0!", sigma > 0));
    };

    /**
     * Returns the RBF kernel value for two vectors x1 and x2 by using the following formula.
     *
     * \f$k(x_1, x_2) = _l^2 \exp(- \frac{\|x_1 - x_2 \|_2^2}{2\sigma^2})\f$
     *
     * The norm in the equation may be substituted by a different function by constructing this object with another distance function.
     * However, the implementation defaults to the squared L2-norm, which effectively resembles the squared euclidean distance between
     * the input vectors.
     *
     * @param x1 A vector.
     * @param x2 A vector.
     * @return RBF kernel value for x1 and x2.
     */
    inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const override {
        data_t distance = 0;
        if (x1 != x2) {
            for (unsigned int i = 0; i < x1.size(); ++i) {
                distance += (x1[i]-x2[i])*(x1[i]-x2[i]);
            }
            distance /= sigma;
        }
        return scale * std::exp(-distance);
    }

    std::shared_ptr<Kernel> clone() const override {
        return std::shared_ptr<Kernel>(new RBFKernel(sigma, scale));
    }
};

#endif // RBF_KERNEL_H
