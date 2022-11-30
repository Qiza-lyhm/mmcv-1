#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"
#include "mlu_common_desc.h"
#include "mlu_op.h"


void mlu_abs_forward_mlu(Tensor input, Tensor output) {
  MluOpTensorDescriptor input_desc, output_desc;
  input_desc.set(input);
  output_desc.set(output);

  auto input_impl = torch_mlu::getMluTensorImpl(input);
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  auto handle = mluOpGetCurrentHandle();
  mluOpAbs(handle, input_desc.desc(), input_ptr, output_desc.desc(), output_ptr);
}

void mlu_abs_forward_impl(Tensor input, Tensor output);

REGISTER_DEVICE_IMPL(mlu_abs_forward_impl, MLU, mlu_abs_forward_mlu);
