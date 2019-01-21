// ComputerGraphics Tuebingen, 2018

#include "knn_bf_sym_op.h"

#include <stdio.h>
#include <type_traits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow
{

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template<typename Device, typename Dtype, typename NBtype>
class KNNBfSymOp: public OpKernel
{
public:
	explicit KNNBfSymOp(OpKernelConstruction* ctx) :
			OpKernel(ctx)
	{
		OP_REQUIRES_OK(ctx, ctx->GetAttr("K", &K));
		OP_REQUIRES_OK(ctx,
				ctx->GetAttr("return_timings", &return_timings));
	}

	void Compute(OpKernelContext* ctx) override
	{
		const Tensor& positions_x = ctx->input(0);
		const Tensor& positions_y = ctx->input(1);
		const Tensor& neighborhood_in = ctx->input(2);

		const int B = positions_x.shape().dim_size(0);
		const int N = positions_x.shape().dim_size(2);

		Tensor* neighborhood_out = nullptr;
		Tensor* distances = nullptr;
		Tensor* timings = nullptr;

		OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape(
				{ B, N, K }), &neighborhood_out));

		OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape(
		{ B, N, K }), &distances));

		OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape(
				{1}), &timings));

		::tensorflow::functor::KNNBfSymFunctor<Device, Dtype, NBtype> KNNBfSymF;
		KNNBfSymF.return_timings = return_timings;
		KNNBfSymF(ctx, positions_x, positions_y, neighborhood_out, distances, timings);
	}

private:
	TF_DISALLOW_COPY_AND_ASSIGN(KNNBfSymOp);

	int K;
	bool return_timings;
//	int subBatch;
};

#define OPNAME(NAME) NAME ## Op
//#define REGISTER(NAME, Dtype)                                          \
//  REGISTER_KERNEL_BUILDER(                                             \
//      Name(#NAME).Device(DEVICE_CPU).TypeConstraint<Dtype>("T"),       \
//      OPNAME(NAME)<CPUDevice, Dtype>);                                 \
//  REGISTER_KERNEL_BUILDER(                                             \
//      Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("T"),       \
//      OPNAME(NAME)<GPUDevice, Dtype>);

//#define REGISTER(NAME, Dtype)                                          \
//  REGISTER_KERNEL_BUILDER(                                             \
//      Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("T"),   \
//      OPNAME(NAME)<GPUDevice, Dtype>);
//REGISTER(KNNGraph, float);

#define REGISTER(NAME, Dtype, NBtype)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("T")			\
      .TypeConstraint<NBtype>("NBtype"),   							\
      OPNAME(NAME)<GPUDevice, Dtype, NBtype>);
REGISTER(KNNBfSym, float, int);

} // namespace tensorflow
