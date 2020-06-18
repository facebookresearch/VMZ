#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "av_input_op.h"

namespace caffe2 {

namespace video_modeling {

REGISTER_CUDA_OPERATOR(AVInput, AVInputOp<CUDAContext>);

} // namespace video_modeling

} // namespace caffe2
