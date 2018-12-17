// ComputerGraphics Tuebingen, 2018

#include "tensorflow/core/framework/op.h"
#include "knn_bf_sym_op.h"

namespace tensorflow {

namespace functor {

template<typename Dtype, typename NBtype>
struct KNNBfSymFunctor<CPUDevice,Dtype, NBtype> {
	void operator ()(::tensorflow::OpKernelContext* ctx,
			const Tensor& positions_x, const Tensor& positions_y, const Tensor& neighborhood_in, Tensor *neighborhood_out, Tensor *distances, Tensor *timings) {
		printf("CPU: KNNBFSym not implemented yet! \n");
	}
};

template struct KNNBfSymFunctor<CPUDevice, float, int>;

} // namespace functor
} // namespace tensorflow
