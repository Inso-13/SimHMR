from typing import Optional, Union

import torch.nn as nn
from mmcv.runner.base_module import BaseModule


class ConvDownsamplerX4(BaseModule):
    def __init__(self, in_channels=1024, out_channels=256, ratio=1.50):
        super(ConvDownsamplerX4, self).__init__()
        self.neck = nn.Sequential(nn.Conv2d(in_channels, int(out_channels * ratio), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(),
                                  nn.Conv2d(int(out_channels * ratio), out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1] 
        return self.neck(x)

class ConvDownsamplerX8(BaseModule):
    def __init__(self, in_channels=1024, out_channels=256, ratio=1.50):
        super(ConvDownsamplerX8, self).__init__()
        self.neck = nn.Sequential(nn.Conv2d(in_channels, int(out_channels * ratio), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(),
                                  nn.Conv2d(int(out_channels * ratio), out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1] 
        return self.neck(x)
