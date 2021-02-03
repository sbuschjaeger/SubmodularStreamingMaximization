#ifndef DATATYPEHANDLING_H_
#define DATATYPEHANDLING_H_

#include <vector>

// Float data-typed used in the entire project. If you find a hardcoded "float" / "double" its probably a good idea to replace it with data_t
typedef double data_t;

// Index data-typed used in the entire project. If you find a hardcoded "size_t" / "unsigned int" etc. its probably a good idea to replace it with idx_t
typedef long idx_t;

// struct Data {
//     std::vector<data_t> x;
//     unsigned long id = 0;
// };

/*typedef Eigen::MatrixXd MatrixX;
typedef Eigen::VectorXd VectorX;

typedef Eigen::Ref<VectorX, 0, Eigen::InnerStride<>> VectorXRef;
typedef Eigen::Ref<const VectorX, 0, Eigen::InnerStride<>> ConstVectorXRef;
typedef std::shared_ptr<MatrixX> SharedMatrixX;*/

#endif
