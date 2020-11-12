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

    std::shared_ptr<Kernel> clone() const override {
        return std::shared_ptr<Kernel>(new RBFKernel(sigma, scale));
    }
};

#endif // RBF_KERNEL_H
