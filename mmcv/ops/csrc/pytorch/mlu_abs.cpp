#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void mlu_abs_forward_impl(Tensor input, Tensor output) {
  DISPATCH_DEVICE_IMPL(mlu_abs_forward_impl, input, output);
}

void mlu_abs_forward(const Tensor& input, Tensor& output) {
  mlu_abs_forward_impl(input, output);
}
