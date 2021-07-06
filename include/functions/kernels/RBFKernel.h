#ifndef RBF_KERNEL_H
#define RBF_KERNEL_H

#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <x86intrin.h>

#include "DataTypeHandling.h"
#include "functions/kernels/Kernel.h"

/* These are remains of an AVX implementation for the euclidean distance. However, it was not much faster (sometimes slower)
 * then the code snippet used from the STL. Also, it only supports float at the moment.
 * 
*/ 
// // From https://gist.github.com/matsui528/583925f88fcb08240319030202588c74
// // reads 0 <= d < 4 floats as __m128
// static inline __m128 masked_read (int d, const float *x) {
//     assert (0 <= d && d < 4);
//     __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
//     switch (d) {
//       case 3:
//         buf[2] = x[2];
//       case 2:
//         buf[1] = x[1];
//       case 1:
//         buf[0] = x[0];
//     }
//     return _mm_load_ps (buf);
//     // cannot use AVX2 _mm_mask_set1_epi32
// }

// template<typename T> 
// T squared_distance(const float * x, const float * y, unsigned int d);

// template<> 
// float squared_distance(const float * x, const float * y, unsigned int d) {
//      __m256 msum1 = _mm256_setzero_ps();

//     while (d >= 8) {
//         __m256 mx = _mm256_loadu_ps (x); x += 8;
//         __m256 my = _mm256_loadu_ps (y); y += 8;
//         const __m256 a_m_b1 = mx - my;
//         msum1 += a_m_b1 * a_m_b1;
//         d -= 8;
//     }

//     __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
//     msum2 +=       _mm256_extractf128_ps(msum1, 0);

//     if (d >= 4) {
//         __m128 mx = _mm_loadu_ps (x); x += 4;
//         __m128 my = _mm_loadu_ps (y); y += 4;
//         const __m128 a_m_b1 = mx - my;
//         msum2 += a_m_b1 * a_m_b1;
//         d -= 4;
//     }

//     if (d > 0) {
//         __m128 mx = masked_read (d, x);
//         __m128 my = masked_read (d, y);
//         __m128 a_m_b1 = mx - my;
//         msum2 += a_m_b1 * a_m_b1;
//     }

//     msum2 = _mm_hadd_ps (msum2, msum2);
//     msum2 = _mm_hadd_ps (msum2, msum2);
//     return  _mm_cvtss_f32 (msum2);
// }

// template<> 
// double squared_distance(const float * x, const float * y, unsigned int d) {
    
// }

/**
 * @brief  The RBF Kernel:
 *      \f[
 *          k(x_1, x_2) = scale \cdot \exp\left(- \frac{\|x_1 - x_2 \|_2^2}{sigma}\right)
 *      \f]
 *      where \f$ scale > 0\f$  and \f$sigma > 0\f$.
 */
class RBFKernel : public Kernel {
private:
    /**
     * Sigma hyperparameter. Should be > 0
     */
    data_t sigma = 1.0;
    
    /**
     * Scale hyperparameter. Should be > 0
     */
    data_t scale = 1.0;

public:
    /**
     * @brief   The default constructor for this kernel. The sigma value is 1.0 and the scale is 1.0
     */
    RBFKernel() = default;

    /**
     * @brief  Creates a new RBFKernel with the given sigma parameter and scale 1.0. 
     * @param  sigma: The sigma parameter > 0.
     */
    RBFKernel(data_t sigma) : RBFKernel(sigma, 1.0) {
    }

    /**
     * @brief  Creates a new RBFKernel with the given sigma and scale parameter. 
     * @note   This constructor uses assert to make sure that scale/sigma has the correct range. This may lead to warnings during compilation.
     * @param  sigma: The sigma value > 0.
     * @param  scale: The scale value > 0.
     */
    RBFKernel(data_t sigma, data_t scale) : sigma(sigma), scale(scale){
        assert(("The scale of an RBF Kernel should be greater than 0!", scale > 0));
        assert(("The sigma value of an RBF Kernel should be greater than  0!", sigma > 0));
    };

    /**
     * @brief  Computes the RBF Kernel at the given points x1, x2:
     *      \f$k(x_1, x_2) = scale * \exp(- \frac{\|x_1 - x_2 \|_2^2}{sigma)\f$
     *          where \f$scale > 0\f$ and \f$sigma > 0\f$
     * @param  x1: First argument for the kernel. 
     * @param  x2: Second argument for the kernel
     */
    inline data_t operator()(const std::vector<data_t>& x1, const std::vector<data_t>& x2) const override {
        data_t distance = 0;
        if (x1 != x2) {
            // This is the fastest stl-compatible version I could find / come up with. I am not sure how much 
            // vectorization this utilizes, but for now this shall be enough
            distance = std::inner_product(x1.begin(), x1.end(), x2.begin(), data_t(0), 
                std::plus<data_t>(), [](data_t x,data_t y){return (y-x)*(y-x);}
            );
            // for (unsigned int i = 0; i < x1.size(); ++i) {
            //     auto const d = x1[i] - x2[i];
            //     distance += d * d;
            // }
            distance /= sigma;
        }
        return scale * std::exp(-distance);
    }

    /**
     * @brief  Returns a clone of this kernel. 
     * @note   The clone is a deep copy of this kernel. 
     */
    std::shared_ptr<Kernel> clone() const override {
        return std::shared_ptr<Kernel>(new RBFKernel(sigma, scale));
    }
};

#endif // RBF_KERNEL_H
