// ComputerGraphics Tuebingen, 2018

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("KNNBfSym")
.Input("position_x: T")						// position_X:	 each datapoint in nd space     	[B, Dp, N].
.Input("position_y: T")						// position_Y:	 each datapoint in nd space     	[B, Dp, M].
.Output("neighborhood: NBtype")				// neighborhood: all K nearest neighbors        	[B, N, K].
.Output("distances: T")						// distances: 	 all K nearest distances 			[B, N, K].
.Output("timings: T")						// timings:											[1]
.Attr("K: int")
.Attr("return_timings: bool = false")
.Attr("T: realnumbertype")
.Attr("NBtype: {int32} = DT_INT32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			const auto position_x = c->input(0);
			const auto position_y = c->input(1);
			const auto neighborhood_out = c->input(2);


			int K;
			c->GetAttr("K", &K);

			auto B = c->Dim(position_x, 0);
			auto N = c->Dim(position_x, 2);

			c->set_output(0, c->MakeShape({B, N, K}));
			c->set_output(1, c->MakeShape({B, N, K}));
			c->set_output(2, c->MakeShape({1}));
			return Status::OK();
		})
.Doc(R"doc(
Find the K nearest neighbors for X in Y.

input
position_x:	 		each datapoint of point cloud X in nd space     		[B, Dp, N].
position_y:	 		each datapoint of point cloud Y in nd space     		[B, Dp, M].

output
neighborhood: 		all K nearest neighbors 				[B, N, K].
distances:    		all K nearest distances        			[B, N, K].
timings				timings									[1]

attributes
K:					K nearest neighbors
return_timings:		return timings (default: false)

)doc");

}

//doc: K:					number of neighbors.
