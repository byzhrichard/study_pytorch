#p70 卷积函数
# 来自torch.nn.AvgPool2d的源码

import torch
class AvgPool2d(_AvgPoolNd):
    def __init__(self,
                 kernel_size: _size_2_t, 
                 stride: Optional[_size_2_t] = None,
                 padding: _size_2_t = 0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 divisor_override: Optional[int] = None
                 ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override