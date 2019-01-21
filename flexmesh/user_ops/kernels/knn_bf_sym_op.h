// ComputerGraphics Tuebingen, 2018

#ifndef LIB_KNN_BF_SYM_OP_H_
#define LIB_KNN_BF_SYM_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
class OpKernelContext;
class Tensor;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
}

namespace tensorflow {
namespace functor {

template<typename Device, typename Dtype, typename NBtype>
struct KNNBfSymFunctor {
	void operator ()(::tensorflow::OpKernelContext* ctx,
			const Tensor& positions_x, const Tensor& positions_y, Tensor *neighborhood_out, Tensor *distances, Tensor *timings);

	bool return_timings;
};

} // namespace functor
} // namespace tensorflow

#endif  // LIB_KNN_BF_SYM_OP_H_
