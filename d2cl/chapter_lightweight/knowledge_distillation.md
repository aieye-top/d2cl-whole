# Knowledge Distillation

背景

集成来提升任务性能，耗时耗力，不利于部署。
将知识压缩到方便部署单个模型是可行的，性能相近。

distill 压缩模型，利用大模型生成的类别概率作为soft targets，待压缩 hard targets。

61.1%
60.8%

旨在把一个大模型或者多个模型ensemble学到的知识迁移到另一个轻量级单模型上，方便部署。简单的说就是用新的小模型去学习大模型的预测结果，改变一下目标函数。听起来是不难，但在实践中小模型真的能拟合那么好吗？所以还是要多看看别人家的实验，掌握一些trick。[3]

知识蒸馏(knowledge distillation，KD)是指对于一个训练好的较大的teacher net，训练一个较小的student net去拟合teacher net的输出(分布)：[8]

蒸馏的目标是让student学习到teacher的泛化能力，理论上得到的结果会比单纯拟合训练数据的student要好。另外，对于分类任务，如果soft targets的熵比hard targets高，那显然student会学习到更多的信息。

## 什么是distillation (或者用Hinton的话说，dark knowledge）



## Transfer Set和Soft target

实验证实，Soft target可以起到正则化的作用（不用soft target的时候需要early stopping，用soft target后稳定收敛）
数据过少的话无法完整表达teacher学到的知识，需要增加无监督数据（用teacher的预测作为标签）或进行数据增强，可以使用的方法有：1.增加[MASK]，2.用相同POS标签的词替换，2.随机n-gram采样，具体步骤参考文献2

由于有teacher network的存在，student network的训练也和普通的监督学习有所不同。

论文：

《Articulatory and Spectrum Features Integration using Generalized Distillation Framework》

### Soft target

hard target 包含的信息量（信息熵）很低，[11]
soft target包含的信息量大，由于加入了关于想要拟合的mapping的prior knowledge，所以拥有不同类之间关系的信息；例子：2像3、2像7那学习的很好的大网络会给label“3”和“7”都有一定的概率值。[10]


soft target的作用在于generalization。同dropout、L2 regularization、pre-train有相同作用。

- dropout是阻碍神经网络学习过多训练集pattern的方法
- L2 regularization是强制让神经网络的所有节点均摊变体的方法。
- pretrain和soft target的方式比较接近，是加入prior knowledge，降低搜索空间的方法。


## 超参数T

T越大越能学到teacher模型的泛化信息。比如MNIST在对2的手写图片分类时，可能给2分配0.9的置信度，3是1e-6，7是1e-9，从这个分布可以看出2和3有一定的相似度，因此这种时候可以调大T，让概率分布更平滑，展示teacher更多的泛化能力
T可以尝试1～20之间

## BERT蒸馏

蒸馏单BERT[4]：模型架构：单层BiLSTM；目标函数：logits的MSE
蒸馏Ensemble BERT[5]：模型架构：BERT；目标函数：soft prob+hard prob；方法：MT-DNN。该论文用给每个任务训练多个MT-DNN，取soft target的平均，最后再训一个MT-DNN，效果比纯BERT好3.2%。但感觉该研究应该是刷榜的结晶，平常应该没人去训BERT ensemble吧。。
BAM[6]：Born-aging Multi-task。用多个任务的Single BERT，蒸馏MT BERT；目标函数：多任务loss的和；方法：在mini-batch中打乱多个任务的数据，任务采样概率为  ，防止某个任务数据过多dominate模型、teacher annealing、layerwise-learning-rate，LR由输出层到输出层递减，因为前面的层需要学习到general features。最终student在大部分任务上超过teacher，而且上面提到的tricks也提供了不少帮助。文献4还不错，推荐阅读一下。
TinyBERT[7]：截止201910的SOTA。利用Two-stage方法，分别对预训练阶段和精调阶段的BERT进行蒸馏，并且不同层都设计了损失函数。与其他模型的对比如下：


https://github.com/0809zheng/Hung-yi-Lee-ML2020-homework/blob/master/hw7_Network_Compression/hw7_Knowledge_Distillation.ipynb

方法
知识蒸馏的两种方法：[8]

Logit Distillation：学生网络学习教师网络的logit输出值
Feature Distillation：学生网络学习教师网络的feature中间值
Relational Distillation：学生网络学习样本之间的关系

loss是KL divergence，用来衡量两个分布之间距离。而KL divergence在展开之后，第一项是原始预测分布的熵，由于是已知固定的，可以消去。第二项是 -q log p，叫做cross entropy，就是平时分类训练使用的loss。与标签label不同的是，这里的q是teacher model的预测输出连续概率。而如果进一步假设q p都是基于softmax函数输出的概率的话，求导之后形式就是 q - p。直观理解就是让student model的输出尽量向teacher model的输出概率靠近[9]





[1]: 蒸馏开山鼻祖Hinton@NIPS2014：Distilling the Knowledge in a Neural Network https://arxiv.org/abs/1503.02531 Distilling the Knowledge in a Neural Network
[2]: https://ai.deepshare.net/detail/v_5f164b66e4b0aebca61a59e3/3?from=p_5ee641d2e8471_5z8XYfL6&type=6
[3]: https://zhuanlan.zhihu.com/p/71986772?utm_source=wechat_session&utm_medium=social&utm_oi=772887009306906624&utm_campaign=shareopn
[4]: BERT -> 单层LSTM：Distilling Task-Specific Knowledge from BERT into Simple Neural Networks https://arxiv.org/abs/1903.12136
[5]: MT-DNN ensemble -> MT-DNN：Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding https://arxiv.org/abs/1904.09482
[6]: Google Single-task ensemble -> Multi-task：BAM! Born-Again Multi-Task Networks for Natural Language Understanding https://arxiv.org/abs/1907.04829
[7]: Huawei -> TinyBERT: Distilling BERT for Natural Language Understanding https://arxiv.org/abs/1909.10351
[8]: https://0809zheng.github.io/2020/05/01/network-compression.html
[9]: https://www.zhihu.com/question/305220135/answer/552545851
[10]: https://antkillerfarm.github.io/dl%20acceleration/2019/07/26/DL_acceleration_5.html
[11]: https://www.zhihu.com/question/50519680/answer/136406661


补充一些资源，还没仔细看：

[dkozlov/awesome-knowledge-distillation](https://github.com/dkozlov/awesome-knowledge-distillation)
[Distilling BERT Models with spaCy](http://www.nlp.town/blog/distilling-bert/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter)
[DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)
[Multilingual MiniBERT: Tsai et al. (EMNLP 2019)](https://arxiv.org/pdf/1909.00100)
BERT蒸馏完全指南｜原理/技巧/代码: https://zhuanlan.zhihu.com/p/273378905
