# Server

一直使用 Google Colab 和 Kaggle Kernel 提供的免费 GPU（Tesla K80）训练模型（最近 Google 将 Colab 的 GPU 升级为 Tesla T4，计算速度又提升了一个档次），不过由于内地网络的原因，Google 和 Kaggle 连接十分不稳定，经常断线重连，一直是很令人头痛的问题，而且二者均有很多限制，例如 Google Colab 一个脚本运行的最长时间为 12h，Kaggle 的为 6h，数据集上传也存在问题，需要使用一些 Trick 才能达成目的，模型的保存、下载等都会耗费很多精力，总之体验不是很好

Google Colab, https://colab.research.google.com/

20种小技巧，玩转Google Colab:
https://www.jiqizhixin.com/articles/2020-09-27-2

https://amitness.com/vscode-on-colab/

从 Kaggle 上传数据的步骤[3]

将「kaggle.json」文件保存在本地计算机上。

安装 Kaggle 软件包：

!pip install -q kaggle

导入包：

from google.colab import files

上传本地文件「kaggle.json」：

files.upload()

## DJL

DJL（Deep Java Library ）是亚马逊在2019年宣布推出的开源Java深度学习开发包，它是在现有深度学习框架基础上使用原生Java概念构建的开发库。它为开发者提供了深度学习的最新创新和使用前沿硬件的能力，例如GPU、MKL等。简单的API抽象并简化了开发深度学习模型所涉及的复杂性，使得DJL更易于学习和应用。有了model-zoo中绑定的预训练模型集，开发者可以立即开始将深度学习的SOTA成果集成到Java应用当中。

DJL秉承了Java的座右铭：「Write once, run anywhere」，不依赖于具体的引擎和深度学习框架，可以随时切换框架。原则上，基于DJL开发人员可以编写在任何引擎上运行的代码。DJL目前提供了MXNet,、PyTorch和TensorFlow的实现。DJL通过调用JNI或者JNA来调用相应的底层操作。DJL 编排管理基础设施，基于硬件配置来提供自动的 CPU/GPU 检测，以确保良好的运行效果。

[1]: https://www.guoyaohua.com/deeplearning-workstation.html#%E4%B8%BB%E6%9D%BF
[2]: https://www.jiqizhixin.com/articles/2020-10-30-12
[3]: https://www.jiqizhixin.com/articles/2020-11-16-11
