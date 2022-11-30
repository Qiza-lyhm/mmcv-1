#pragma once
#include <ATen/ATen.h>
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"
#include <c10/core/ScalarType.h>
#include "mlu_op.h"

mluOpDataType_t getMluOpDataType(const caffe2::TypeMeta& data_type);

class MluOpTensorDescriptor {
  public:
    MluOpTensorDescriptor() {
      mluOpCreateTensorDescriptor(&desc_);
    };
    ~MluOpTensorDescriptor() {
      mluOpDestroyTensorDescriptor(desc_);
    }

    void set(at::Tensor);
    mluOpTensorDescriptor_t desc() {return desc_;}

  private:
    mluOpTensorDescriptor_t desc_;
    void set_desc(const at::Tensor&, mluOpTensorLayout_t, mluOpDataType_t,
                  std::vector<int>& dims);
};

mluOpHandle_t mluOpGetCurrentHandle(c10::DeviceIndex device_index = -1);

class MluOpHandle{
  public:
    MluOpHandle(): handle(nullptr) {
      mluOpCreate(&handle);
    }
    ~MluOpHandle() {
      if (handle) {
        mluOpDestroy(handle);
        handle = nullptr;
      }
    }
    void setQueue(cnrtQueue_t queue) {
      mluOpSetQueue(handle, queue);
    }
    mluOpHandle_t handle;
};
