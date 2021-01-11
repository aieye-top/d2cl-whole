# ShuffleNet

网络是Megvii Inc. (Face++)提出。ShuffleNet pursues the best accuracy in very limited computational budgets at tens or hundreds of MFLOPs
In ARM device, ShuffleNet achieves 13× actual speedup over AlexNet while maintaining comparable accuracy.[6] 2018 CVPR : 300 citations.

Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of 40 MFLOPs.[2]


CNN
分组卷积
ResNet

分组点卷积
通道重排
shufflenet v2
pytorch复现

## 动机

数百层和数千通道，Billions of FLOPs

## 方法

属于直接训练而不是压缩

## 分组点卷积Group convolutions`


给点卷积也分组

分组点卷积某个通道的输出仅来及一部分输入通道，阻止了信息流动，特征表示。


Group convolutions are used in AlexNet and ResNeXt.
(a): There is no channel shuffle, each output channel only relates to the input channels within the group. This property blocks information flow between channel groups and weakens representation.
(b): If we allow group convolution to obtain input data from different groups, the input and output channels will be fully related.
(c): The operations in (b) can be efficiently and elegantly implemented by a channel shuffle operation. Suppose a convolutional layer with g groups whose output has g×n channels; we first reshape the output channel dimension into (g, n), transposing and then flattening it back as the input of next layer.
And channel shuffle is also differentiable, which means it can be embedded into network structures for end-to-end training.[6]

## 通道重排(channel shuffle)

如果我们允许组卷积从不同组中获取输入数据，则输入和输出通道讲完全相关。

对于从上一个组层生成的特征图，可以先将每一个组中的通道划分为几个子组，
然后在下一层中的每个组中使用不同的子组作为输入。


ShuffleNet中的Channel Shuffle操作可以将组间的信息进行交换，并且可以实现端到端的训练。


Group convolution。其中输入特征通道被为G组(图4)，并且对于每个分组的信道独立地执行卷积，则分组卷积计算量是HWNK²M/G，为标准卷积计算量的1/G。
Channel shuffle。Grouped Convlution导致模型的信息流限制在各个group内，组与组之间没有信息交换，这会影响模型的表示能力。因此，需要引入group之间信息交换的机制，即Channel Shuffle操作。

引入的问题：[5]

channel shuffle在工程实现占用大量内存和指针跳转，这部分很耗时。
channel shuffle的规则是人工设计，分组之间信息交流存在随意性，没有理论指导。


The motivation of ShuffleNet is the fact that conv1x1 is the bottleneck of separable conv as mentioned above. While conv1x1 is already efficient and there seems to be no room for improvement, grouped conv1x1 can be used for this purpose!

The above figure illustrates the module for ShuffleNet. The important building block here is the channel shuffle layer which “shuffles” the order of the channels among groups in grouped convolution. Without channel shuffle, the outputs of grouped convolutions are never exploited among groups, resulting in the degradation of accuracy.[7]



## FLOPS

![FLOPs] (img\Shuffle_Flops.jp)

```py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes/4
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out

def ShuffleNetG2():
    cfg = {
        'out_planes': [200,400,800],
        'num_blocks': [4,8,4],
        'groups': 2
    }
    return ShuffleNet(cfg)

def ShuffleNetG3():
    cfg = {
        'out_planes': [240,480,960],
        'num_blocks': [4,8,4],
        'groups': 3
    }
    return ShuffleNet(cfg)


def test():
    net = ShuffleNetG2()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)
```

- The proposed network is mainly composed of a stack of ShuffleNet units grouped into three stages.
- The number of bottleneck channels is set to 1/4 of the output channels for each ShuffleNet unit.
- A scale factor s is applied on the number of channels. The networks in the above table is denoted as “ShuffleNet 1×”, then ”ShuffleNet s×” means scaling the number of filters in ShuffleNet 1× by s times thus overall complexity will be roughly s² times of ShuffleNet 1×.

ShuffleNet也从宏观和微观两个层面分别对网络进行了优化。无独有偶，其瞄准的主要优化对象其实也是卷积核，ShuffleNet不仅采用了更小的卷积核，而且还采用了一种分组卷积的概念组合小型的卷积核，以求减少计算的复杂度。

ShuffleNet和ResNet结构可知，ShuffleNet计算量降低主要是通过分组卷积实现。ShuffleNet虽然降低了计算量，但是引入两个新的问题：[4]

1、channel shuffle在工程实现占用大量内存和指针跳转，这部分很耗时。
2、channel shuffle的规则是人工设计，分组之间信息交流存在随意性，没有理论指导。

## 什么是分组

所谓分组就是将输入与输出的通道分成几组，比如输出与输入的通道数都是4个且分成2组，那第1、2通道的输出只使用第1、2通道的输入，同样第3、4通道的输出只使用第3、4通道的输入。也就是说不同组之间的输入和输出之间完全没有了关系，减少联系势必减少计算量（有联系就说明要进行运算）。当然这种方式的副作用就是会损失信息，可能导致准确率下降

### 计算量可减少多少

在分组之前，每一层的参数数量是 $N \times C \times H \times W$,
如果将输入输出分成 $g$ 组，那么每一组的参数数量就会变成 $\frac{N \times C \times H \times W}{g}$ 个，虽然每层特征输出总数量
依然不变，但是每一组自己运算的计算次数也会变成原来的 $\frac{1}{g},$ 也就是说分组之后计算量可以降低到 原来的 $\frac{1}{g^{2}},$ 而参数数量可以降低到原来的 $\frac{1}{g}$ 。

以上是单样本输入的情况，那么如果同时输入多个样本呢？ 如果是在内存资源充足的服务器端，我们
可以利用数据并行的思路, 让k个样本多线程同时执行，速度自然可以提高K倍。但是在移动平台我
们往往没有那么充足的内存资源, CPU也不支持太多线程同时执行, 因此很有可能每个样本依然是独
立执行的，速度变化和单样本没有什么差距。

不过这些都是理论分析，实际上移动平台的计算效率并不能提高如此之多，一方面卷积运算一般为了减少预算复杂度，都是先通过im2col转成向量，然后执行矩阵乘法，而im2col和矩阵运算时间其实相差无几，同时现代化的线性代数库都极大优化了矩阵运算性能，因此实际的性能提升肯定会受到影响。

## ShuffleNet-V2[8]

由上图可以看到，相同FLOPs的两个模型, 各部分的运行时间存在着明显的差异。这种不一致主要归结为两个原因:
1) 影响速度的不仅仅是FLOPs，还有内存访问成本（Memory Access cost, MAC） ;
2）模型的并行

程度也会影响速度, 并行度高的模型速度相对更快。因此作者结合理论与实践得到了四条实用的设计原则。
1. 同等通道大小最小化内存访问成本一一使用1 $\times 1$ 卷积平衡输入和输出的通道大小
2. 过量使用分组卷积会增加MAC一一分组卷积要谨慎实用, 注意分组数
3. 网络碎片化会降低并行度, 一些网络如inception等倾向于采用"多路"结构, 既存在一个block中有很多不同 的小卷积或pooling，这容易造成网络碎片化, 降低并行度。一文避免网络碎片化
4. 不能忽略元素级别的操作，例如ReLU和Add等操作，这些操作虽然FLOPs较小，但是MAC较大。——减少元素级运算


(a): the basic ShuffleNet-V1 unit; (b) the ShuffleNet-V1 unit for spatial down sampling $(2 \times) ;$ (c) ShuffleNet-V2
basic unit; (d) ShuffleNet-V2 unit for spatial down sampling ( $2 \times$ )

ShuffleNet-V2 相对与V1，引入了一种新的运算: channel split。具体来说，在开始时先将输入特征图在通道 维度分成两个分支：通道数分别为 $C^{\prime}$ 和 $C-C^{\prime},$ 实际实现时 $C^{\prime}=C / 2$ 。左边分支做同等映射, 右边的
分支包含3个连续的卷积, 并且输入和输出通道相同，这符合准则1。而且两个1x1卷积不再是组卷积, 这符合
准则2，另外两个分支相当于已经分成两组。两个分支的输出不再是Add元素，而是concat在一起, 紧接着是
对两个分支concat结果进行channle shuffle, 以保证两个分支信息交流。其实concat和channel shuffle可以和
下一个模块单元的channel split合成一个元素级运算，这符合准则4。整体网络结果如下表:

## Comparison with MobileNetV1[6]

- ShuffleNet models are superior to MobileNetV1 for all the complexities.
- Though ShuffleNet network is specially designed for small models (< 150 MFLOPs), it is still better than MobileNetV1 for higher computation cost, e.g. 3.1% more accurate than MobileNetV1 at the cost of 500 MFLOPs.
- The simple architecture design also makes it easy to equip ShuffeNets with the latest advances such as Squeeze-and-Excitation (SE) blocks. (Hope I can review SENet in the future.)
- ShuffleNets with SE modules boosting the top-1 error of ShuffleNet 2× to 24.7%, but are usually 25 to 40% slower than the “raw” ShuffleNets on mobile devices, which implies that actual speedup evaluation is critical on low-cost architecture design.


它在移动端低功耗设备提出了一种更为高效的卷积模型结构，在大幅降低模型计算复杂度的同时仍然保持了较高的识别精度，并在多个性能指标上均显著超过了同类方法。[9]
　　ShuffleNet Series涵盖以下6个模型：
　　（1） ShuffleNetV1: ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
　　论文链接：https://arxiv.org/abs/1707.01083
　　解读链接：为移动 AI 而生——旷视最新成果 ShuffleNet 全面解读
　　（2） ShuffleNetV2: ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
　　论文链接：https://arxiv.org/abs/1807.11164
　　解读链接：ECCV 2018 | 旷视提出新型轻量架构ShuffleNet V2：从理论复杂度到实用设计准则
　　（3） ShuffleNetV2+: ShuffleNetV2 的增强版
　　（4） ShuffleNetV2.Large: ShuffleNetV2 的深化版
　　（5） OneShot: Single Path One-Shot Neural Architecture Search with Uniform Sampling
　　论文链接：https://arxiv.org/abs/1904.00420
　　解读链接：AutoML | 旷视研究院提出One-Shot模型搜索框架的新变体
　　（6） DetNAS: DetNAS: Backbone Search for Object Detection
　　论文链接：https://arxiv.org/abs/1903.10979

[1]: https://ai.deepshare.net/detail/v_5ee645312d94a_eMNJ5Jws/3?from=p_5ee641d2e8471_5z8XYfL6&type=6
[2]: https://arxiv.org/abs/1707.01083
[3]: https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenet.py
[4]: https://zhuanlan.zhihu.com/p/45496826
[5]: https://cygao.xyz/2019/07/12/lightweight/
[6]: https://towardsdatascience.com/review-shufflenet-v1-light-weight-model-image-classification-5b253dfe982f
[7]: https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d
[10]: https://aistudio.baidu.com/aistudio/projectdetail/56879?channelType=0&channel=0
[8]: https://github.com/megvii-model/ShuffleNet-Series
[9]: http://os.aiiaorg.cn/open/article/1201782277957726210
