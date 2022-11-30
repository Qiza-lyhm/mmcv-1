import torch
import torch_mlu
import mmcv
from mmcv.ops import mlu_abs

a = torch.randn(7)
print(a)

print(mlu_abs(a.mlu()))
