import math
import numpy as np
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
    
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
    
    
###### Jiayuan learnable dropout in concat
class Concat_dropout(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1, ch=None):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
        self.weight = nn.Parameter(torch.tensor([5.0]))

        self.conv = nn.Conv2d(sum(ch),
                               ch[0],
                               kernel_size=1,
                               stride=1)

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        if torch.sigmoid(self.weight) > 0.5:
            concatenated = torch.cat(x, self.d)
            return self.conv(concatenated)
        else:
            return x[0]
        
        
class Adapt_concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        x[1] = x[1] * self.weight
        return torch.cat(x, self.d)
    

class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid 
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        if not torch.onnx.is_in_onnx_export():
            z = []  # inference output
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                # print(str(i)+str(x[i].shape))
                bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                x[i] = x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()
                # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                # print(str(i)+str(x[i].shape))

                if not self.training:  # inference
                    # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    #     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    #print("**")
                    #print(y.shape) #[1, 3, w, h, 85]
                    #print(self.grid[i].shape) #[1, 3, w, h, 2]
                    # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy (bs,na,ny,nx,2)
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh (bs,na,ny,nx,2)
                    y = torch.cat((xy, wh, y[..., 4:]), -1)  # (bs,na,ny,nx,2+2+1+nc=xy+wh+conf+cls_prob)

                    """print("**")
                    print(y.shape)  #[1, 3, w, h, 85]
                    print(y.view(bs, -1, self.no).shape) #[1, 3*w*h, 85]"""
                    z.append(y.view(bs, -1, self.no))  # y (bs,na*ny*nx,no=2+2+1+nc=xy+wh+conf+cls_prob)
            return x if self.training else (torch.cat(z, 1), x)
        else:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                x[i] = x[i].view(bs, -1, self.no)
            return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()