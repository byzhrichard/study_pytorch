#p70 卷积函数
# 来自torch.nn.Conv2d的源码

import torch
class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,   #输入维度
        out_channels: int,  #输出维度
        kernel_size: _size_2_t, #卷积核大小
        stride: _size_2_t = 1,  #步长
        padding: Union[str, _size_2_t] = 0, #填充方式，只能为1或0，决定不同的卷积方式
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)