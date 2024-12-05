import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
from thop import profile
from torchsummary import summary

sys.path.append(os.getcwd())
# sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
# from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from lib.models.common3 import Conv, SPPF, C2f, Concat, Detect, Adapt_concat
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized

# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [ 
[22, 30, 38],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx (640, 640, 3)
[ -1, Conv, [3, 32, 3, 2]],   #0 (320, 320, 32)
[ -1, Conv, [32, 64, 3, 2]],    #1 (160, 160, 64)
[ -1, C2f, [64, 64, 1, True]],  #2 (160, 160, 64)
[ -1, Conv, [64, 128, 3, 2]],   #3 (80, 80, 128)
[ -1, C2f, [128, 128, 3, True]],    #4 (80, 80, 128)
[ -1, Conv, [128, 256, 3, 2]],  #5 (40, 40, 256)
[ -1, C2f, [256, 256, 3, True]],    #6 (40, 40, 256)
[ -1, Conv, [256, 512, 3, 2]],  #7 (20, 20, 512)
[ -1, C2f, [512, 512, 1, True]], #8 (20, 20, 512)
[ -1, SPPF, [512, 512, 5]],     #9 (20, 20, 512)

[ -1, Upsample, [None, 2, 'nearest']], #10 (40, 40, 512)
[ [-1, 6], Concat, [1]],    #11 (40, 40, 768)
[ -1, C2f, [768, 256, 1, False]],     #12 (40, 40, 256)
[ -1, Upsample, [None, 2, 'nearest']], #13 (80, 80, 256)
[ [-1, 4], Concat, [1]],    #14 (80, 80, 384) Encoder

[ -1, C2f, [384, 128, 1, False]],     #15 (80, 80, 128) 
[ -1, Conv, [128, 128, 3, 2]], #16 (40, 40, 128)
[ [-1, 12], Concat, [1]],    #17 (40, 40, 384)
[ -1, C2f, [384, 256, 1, False]],     #18 (40, 40, 256)
[ -1, Conv, [256, 256, 3, 2]], #19 (20, 20, 256)
[ [-1, 9], Concat, [1]],    #20 (20, 20, 768)
[ -1, C2f, [768, 512, 1, False]],     #21 (20, 20, 512)
[ [15, 18, 21], Detect,  [1, [[2,6,3,9,5,14], [8,20,12,29,19,46], [26,97,39,77,61,147]], [128, 256, 512]]], #Detection head 22

[ 14, Upsample, [None, 2, 'nearest']], #23 (160, 160, 384)
[ [-1, 2], Adapt_concat, [1]],    #24 (160, 160, 448)
[ -1, C2f, [448, 128, 1, False]],     #25 (160, 160, 128)
[ -1, Upsample, [None, 2, 'nearest']], #26 (320, 320, 128)
[ [-1, 0], Adapt_concat, [1]],    #27 (320, 320, 160)
[ -1, C2f, [160, 64, 1, False]],     #28 (320, 320, 64)
[ -1, Upsample, [None, 2, 'nearest']], #29 (640, 640, 64)
[ -1, Conv, [64, 2, 3, 1]],     #30 (640, 640, 2) Driving area segmentation head (640, 640, 2)

[ 14, Upsample, [None, 2, 'nearest']], #31 (160, 160, 384)
[ [-1, 2], Adapt_concat, [1]],    #32 (160, 160, 448)
[ -1, C2f, [448, 128, 1, False]],     #33 (160, 160, 128)
[ -1, Upsample, [None, 2, 'nearest']], #34 (320, 320, 128)
[ [-1, 0], Adapt_concat, [1]],    #35 (320, 320, 160)
[ -1, C2f, [160, 64, 1, False]],     #36 (320, 320, 64)
[ -1, Upsample, [None, 2, 'nearest']], #37 (640, 640, 64)
[ -1, Conv, [64, 2, 3, 1]],     #38 (640, 640, 2) Lane line segmentation head (640, 640, 2)
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:     #save driving area and lane line segment result
                m = nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)

        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs): 
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 640, 640))

    macs, params = profile(model, inputs=(input_,))
    print("FLOPs:", macs)  # FLOPs模型复杂度
    print("params:", params)  # params参数量

    # gt_ = torch.rand((1, 2, 640, 640))
    # metric = SegmentationMetric(2)
    # model_out = model(input_)
    # detects, dring_area_seg, lane_line_seg = model_out
    #
    # for det in detects:
    #     print(det.shape)
    # print(dring_area_seg.shape)
    # print(lane_line_seg.shape)

# FLOPs: 18792857600.0
# params: 10069278.0