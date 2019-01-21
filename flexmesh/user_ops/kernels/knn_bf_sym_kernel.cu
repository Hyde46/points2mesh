// ComputerGraphics Tuebingen, 2018

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <vector>

#include <helper_cuda.h>
#include <cuda.h>

#include <cub/cub.cuh>

#include <curand.h>
#include <curand_kernel.h>

#include <limits>

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "knn_bf_sym_op.h"

//template<typename Dtype, typename NBtype>
//struct DiagDistanceDiffKernel;
//
//template<typename Dtype, typename NBtype>
//__global__ void runSymmetricHashKernel(
//		const DiagDistanceDiffKernel<Dtype, NBtype> kernel)
//{
//	kernel();
//}
//
//template<typename Dtype, typename NBtype>
//struct DiagDistanceDiffKernel
//{
//	enum
//	{
//		C_NW = 32, C_NH = 32
//	};
//
//	void launch(int B)
//	{
//
////		dim3 block(K, C_Threads / K);
////		dim3 grid((N - 1) / block.y + 1, 1, B);
////
////		runSymmetricHashKernel<<<grid, block>>>((*this));
//	}
//
//	__device__ __forceinline__ void operator()() const
//	{
//		extern __shared__ Dtype s_points[];
//
//		int b = blockIdx.z;
//
//		//skip complete block if over diag
//		if (blockIdx.x * C_NW > (blockIdx.y + 1) * C_NH - 1)
//			return;
//
//		//load shm
//		for (int tid = threadIdx.y*C_NW+threadIdx.x; tid < (C_NW+C_NH)*Dp; tid+=C_NW+C_NH) {
//			if(tid<C_NW*Dp)
//			{
//				int dpi = tid/C_NW;
//				int ni = tid%C_NW;
//				s_points[tid] = d_points[(offsetB + b) * N * Dp + dpi * N + blockIdx.x * C_NW + ni];
//			}
//			else
//			{
//				int dpi = tid/C_NH;
//				int ni = tid%C_NH;
//				s_points[tid] = d_points[(offsetB + b) * N * Dp + dpi * N + blockIdx.x * C_NH + ni];
//			}
//		}
//		__syncthreads();
//
//
//		int x = blockIdx.x * C_NW + threadIdx.x;
//		int y = blockIdx.y * C_NH + threadIdx.y;
//
//		if (x > y || x >= N || y >= N)
//			return;
//
//		if (x == y)
//		{
//			d_diff[b * N * N + y * N + x] = 0;
//			d_idx[b * N * N + y * N + x] = x;
//			return;
//		}
//
//		float tmp = length(s_points[tidx] - s_points[blockDim.x + tidy]);
//
//		Dtype sum = 0.f;
//		for (int dpi = 0; dpi < Dp; ++dpi) {
//			Dtype val = s_points[dpi*C_NW+threadIdx.x] - s_points[Dp*C_NW+dpi*C_NH+threadIdx.y];
//			sum += val * val;
//		}
//		sum = sqrt(sum);
//
//		d_diff[b * N * N + y * N + x] = sum;
//		d_diff[b * N * N + x * N + y] = sum;
//
//		d_idx[b * N * N + y * N + x] = x;
//		d_idx[b * N * N + x * N + y] = y;
//	}
//
//	const Dtype* d_points;
//
//	Dtype* d_diff;
//	NBtype* d_idx;
//
//	int N;
//	int Dp;
//	int offsetB;
//
//};


template<typename Dtype, typename NBtype, int C_THREADS=256, int C_VPT=2>
struct BlockBFSymKernel;

template<typename Dtype, typename NBtype, int C_THREADS, int C_VPT>
__global__ void runBlockBFSymKernel(
		const BlockBFSymKernel<Dtype, NBtype,C_THREADS,C_VPT> kernel)
{
	kernel();
}

template<typename Dtype, typename NBtype, int C_THREADS, int C_VPT>
struct BlockBFSymKernel
{

	void launch(int B)
	{
		dim3 block(C_THREADS, 1);
		dim3 grid(N, 1, B);

		size_t shm_size = Dp * sizeof(Dtype);

		runBlockBFSymKernel<<<grid, block, shm_size>>>((*this));
	}

	__device__ __forceinline__ void operator()() const
	{

		extern __shared__ Dtype s_shm[];
		Dtype* s_point = (Dtype*) &s_shm[0];

		if(M>C_THREADS*C_VPT)
		{
			printf("<BlockBFSymKernel> Critical problem!!!!! Not enough resources spend for M points!!! %d < %d \n",C_THREADS*C_VPT,N);
			return;
		}

		int b = blockIdx.z;
		int y = blockIdx.x;

		int tid = threadIdx.x;
		for (int dpi = tid; dpi < Dp; dpi+=blockDim.x)
		{
			s_point[dpi] = d_posX[b * N * Dp + dpi * N + y]; //not aligned with data!
		}

		__syncthreads();

		Dtype thread_dists[C_VPT]; 	//keys
		NBtype thread_ids[C_VPT];	//values

		typedef cub::BlockRadixSort<Dtype, C_THREADS, C_VPT, NBtype> BlockRadixSort;
		typedef cub::BlockStore<Dtype, C_THREADS, C_VPT, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreDists;
		typedef cub::BlockStore<NBtype, C_THREADS, C_VPT, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreIds;

		// Allocate shared memory
		__shared__ union
		{
			typename BlockRadixSort::TempStorage sort;
			typename BlockStoreDists::TempStorage store_dists;
			typename BlockStoreIds::TempStorage store_ids;

		}temp_storage;

		for (int vpt_i = 0; vpt_i < C_VPT; ++vpt_i)
		{
			int x = vpt_i * C_THREADS + threadIdx.x;

			if(x<M)
			{
				Dtype sum = 0.f;
				for (int dpi = 0; dpi < Dp; ++dpi)
				{
//					Dtype val = d_data[b * N * Dp + dpi * N + x] - d_data[b * N * Dp + dpi * N + y];
					Dtype val = d_posY[b * M * Dp + dpi * M + x] - s_point[dpi];
					sum += val * val;
				}
				thread_dists[vpt_i] = sqrt(sum);
				thread_ids[vpt_i] = x;
			}
			else
			{
				thread_dists[vpt_i] = std::numeric_limits<Dtype>::max();
				thread_ids[vpt_i] = -1;
			}
		}

		BlockRadixSort(temp_storage.sort).Sort(thread_dists, thread_ids);
		__syncthreads();

		BlockStoreIds(temp_storage.store_ids).Store(&d_knn_ids[b*N*K + y*K], thread_ids, K);
		__syncthreads();

		BlockStoreDists(temp_storage.store_dists).Store(&d_knn_dists[b*N*K + y*K], thread_dists, K);

	}

	const Dtype* d_posX;
	const Dtype* d_posY;

	Dtype* d_knn_dists;
	NBtype* d_knn_ids;

	int N;
	int M;
	int Dp;
	int K;
};

template<typename Dtype, typename NBtype>
struct BlockBFSymKernelAttributesSetter
{
	template<typename T>
	void setAttributes(T& kernel)
	{
		kernel.d_posX = d_posX;
		kernel.d_posY = d_posY;
		kernel.d_knn_dists = d_knn_dists;
		kernel.d_knn_ids = d_knn_ids;

		kernel.N = N;
		kernel.M = M;
		kernel.Dp = Dp;
		kernel.K = K;
	}

	const Dtype* d_posX;
	const Dtype* d_posY;

	Dtype* d_knn_dists;
	NBtype* d_knn_ids;

	int N;
	int M;
	int Dp;
	int K;
};

namespace tensorflow
{
namespace functor
{

template<typename Dtype, typename NBtype>
struct KNNBfSymFunctor<GPUDevice, Dtype, NBtype>
{
	void operator ()(::tensorflow::OpKernelContext* ctx,
			const Tensor& positions_x, const Tensor& positions_y,
			Tensor *neighborhood_out, Tensor *distances, Tensor *timings)
	{
//		printf("GPU: KNNBFSym! \n");

//		printf("return_timings: %d \n",return_timings);

		const int B = positions_x.dim_size(0);
		const int D = positions_x.dim_size(1);
		const int N = positions_x.dim_size(2);
		const int M = positions_y.dim_size(2);

		const int K = neighborhood_out->dim_size(2);
//		printf("B: %d | N: %d | K: %d \n", B, N, K);

		BlockBFSymKernelAttributesSetter<Dtype,NBtype> attr;
		attr.d_posX = positions_x.flat<Dtype>().data();
		attr.d_posY = positions_y.flat<Dtype>().data();
		attr.d_knn_ids = neighborhood_out->flat<NBtype>().data();
		attr.d_knn_dists = distances->flat<Dtype>().data();

		attr.N = N;
		attr.M = M;
		attr.Dp = D;
		attr.K = K;

		cudaEvent_t start, stop;
		if (return_timings)
		{
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
		}

		//TODO: beautify
		if(M <= 32)
		{
			BlockBFSymKernel<Dtype,NBtype,32,1> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=64)
		{
			BlockBFSymKernel<Dtype,NBtype,64,1> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=128)
		{
			BlockBFSymKernel<Dtype,NBtype,128,1> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=256)
		{
			BlockBFSymKernel<Dtype,NBtype,128,2> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=512)
		{
			BlockBFSymKernel<Dtype,NBtype,128,4> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=1024)
		{
			BlockBFSymKernel<Dtype,NBtype,256,4> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=2048)
		{
			BlockBFSymKernel<Dtype,NBtype,256,8> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=4096)
		{
			BlockBFSymKernel<Dtype,NBtype,512,8> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else if(M<=1024*8)
		{
			BlockBFSymKernel<Dtype,NBtype,1024,8> knn;
			attr.setAttributes(knn);
			knn.launch(B);
		}
		else
		{
			printf("point sets greater then 8k are note yet supported!! Change to knn_graph operation!! \n");
		}


		if (return_timings)
		{
			float time;
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
//			printf("complete elapsed time for KNNBFSym: %f ms \n", time);

			cudaMemcpy(timings->flat<Dtype>().data(),&time,sizeof(float),cudaMemcpyHostToDevice);

			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		}
		cudaDeviceSynchronize();
		getLastCudaError("KNNBFSym execution failed");
		checkCudaErrors(cudaDeviceSynchronize());

	}

	bool return_timings;

};

template struct KNNBfSymFunctor<GPUDevice, float, int> ;

} // namespace functor
} // namespace tensorflow

#endif  // GOOGLE_CUDA
