import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['mlu_abs_forward'])

class MluAbs(Function):
    @staticmethod
    def symbolic(g, input:torch.Tensor):
        return g.op('mmcv::MMCVMluAbs',
                    input)
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output = input.new_zeros(input.size())
        ctx.save_for_backward(input)
        ext_module.mlu_abs_forward(input, output)
        return output

    # demo with fake backward
    @staticmethod
    def backward(ctx, grad_output) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = input.new_zeros(input.size())
        # abs backward is abs(x)/x, which means -1 for x<0 and 1 for x>0

        ext_module.mlu_abs_forward(input, grad_input)
        return grad_output * (grad_input / input)

mlu_abs = MluAbs.apply
