
#include "bang_sigmoid_sample.h"
#include "customed_ops.h"

#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/MLUStream.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor active_sigmoid_mlu(torch::Tensor x) {
  auto x_contiguous = torch_mlu::cnnl_contiguous(x);
  auto x_impl = getMluTensorImpl(x_contiguous);
  auto x_ptr = mlu_data_ptr(x_impl);

  auto y = at::empty_like(x_contiguous);
  auto y_contiguous = torch_mlu::cnnl_contiguous(y);
  auto y_impl = getMluTensorImpl(y_contiguous);
  auto y_ptr = mlu_data_ptr(y_impl);

  int32_t size = x_contiguous.numel();

  cnrtQueue_t queue = getCurMLUStream();
  // TODO: 请补充Sigmoid主程序函数接口的签名
  bang_sigmoid_kernel_entry(
      queue,
      reinterpret_cast<float*>(y_ptr),
      reinterpret_cast<float*>(x_ptr),
      size);

  return y;
}

PYBIND11_MODULE(libmlu_custom_ext, m) {
  m.def("active_sigmoid_mlu", &active_sigmoid_mlu);
}
