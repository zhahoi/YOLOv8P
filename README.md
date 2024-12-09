# YOLOv8P
尝试使用YOLOV8使用的模块替换YOLOP的主干网络（Attempt to replace the backbone network of YOLOP with the modules used in YOLOV8.）



## 网络结构

YOLOP是基于YOLOv5的模块构建的，想着如果使用yolov8的模块对其进行替换，可能会提升检测结果。在Github上搜索一番之后，发现有一个仓库为[YOLOv8-multi-task](https://github.com/JiayuanWang-JW/YOLOv8-multi-task)，该仓库在YOLOv8基础之上将其修改成多任务的模型，其原理上和YOLOP一致，但是有了些许创新。本仓库设计的主干结构参考了该仓库创新结构设计。

本仓库设计的YOLOv8P网络结构如下：

```python
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
```

在YOLOP的基础上替换了YOLOv8使用的`C2f` 和 `SPPF`模块，而`Adapt_concat`参考自[YOLOv8-multi-task](https://github.com/JiayuanWang-JW/YOLOv8-multi-task)，没有详细考究该模块是否有用，但是既然发了论文应该有一定效果。此外，原始的YOLOP中对于车道线检测和可行驶区域分支网络的设计，我觉得过于草率了，因为对于分割网络，融合上下文信息非常重要（如Unet)，我自己训练原始YOLOP网络时（部分数据和较少的epoch数），发现可行使区域和车道线检测预测的结果并不好。而[YOLOv8-multi-task](https://github.com/JiayuanWang-JW/YOLOv8-multi-task)融合了backbone的特征层，对于预测可行使区域和车道线检测有很大的帮助。

本仓库设计的YOLOv8P的网路结构设计，参考了二者的结构，融合成了最后的网络。为了减少计算量，尽量减少了通道数和模块的重复次数。



## 训练和配置

本仓库的代码是在YOLOP官方仓库上修改的，只修改了网络结构和部分内容，训练和配置可以参考官网的[README](https://github.com/hustvl/YOLOP/blob/main/README.md)文档。本仓库网络结构设计，不保证比原始模型更优，因为手头现有GPU资源的限制，没有办法进行详细测试。仅供学习参考。



-注：在获取bdd100k数据集时，发现原始的伯克利官方下载的网站不能使用了，但是可以从以下链接下载到：https://dl.cv.ethz.ch/bdd100k/data/



## Reference

-**[YOLOv8-multi-task](https://github.com/JiayuanWang-JW/YOLOv8-multi-task)**

-**[YOLOP](https://github.com/hustvl/YOLOP)**
