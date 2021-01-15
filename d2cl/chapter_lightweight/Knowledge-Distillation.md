# Knowledge-Distillation

知识蒸馏最早由Buciluǎ等人[146]提出训练了带有伪数据分类器的压缩模型,复制了原始分类器的输出.与其他压缩与加速方法只使用需要被压缩的目标网络不同知识蒸馏法需要两种类型的网络教师模和学生模型预先训练好的教师模型通常网络模型具有很好的性能如图6所示将教师模型的 softmax层输出作为soft生模型的 softmax层输出作为 hard target一同送total losss计算指生模型训缭将教师模型的知识迁移到学生模型中,使学生模型达到与教当的性能学生模型更加紧凑效,起到模型压缩的目的.知识蒸馏法能使深层网络变浅.大大降低计算成本,但也有其局限性.由于使softmax层输出作为知识,所以一般多用于具有s损失函数的分类任务,在其它任务的泛化性不好;并就目前来看其压缩比与蒸馏后的模型性能还存在较大进步空间


## 学生模型的网络结构

知识蒸馏法的研究方向之一就是如生模型选择合适的网络结构,帮助学生模型更好教师模型的知识Ba等人[147]提出在保证教师模型和学生模型网络参数数的情况下,设计更浅的学生模型,每层变得更宽 Romero等人[148与[147观点不同认为更深的学生模型分类效果更好提使用教师网络的中间层输出 Hints作为监督信息训练学生网络的前半部分Chen等人[149]提出使用生长式网络结构以复制的方式重用预训练的网络参数,在此基础上进行结构拓展i等人[150]与[149]观点一致提出分别从宽度深度上进行网络生长C等人[151]提出将知识蒸馏与设计更紧凑的网络结构结合将原网络作为教师模型,将使用简化卷积的网络作为学生模型zhu等提出基于原始网络构造多分支结构将每个分支作为学生网络融合生成推理性能更强的教师网络2教师模型的学习信除了使用 softmax层输岀作为教师模型的学习信息,一些研究者认为可以使用教师模型中的其他信息帮助高晗等:深度学习模型压縮与加速综述知识迁移 Hinton等人[153]首先提出使用教师模型的类别概率输出计算 soft target.为了方便计算还引入温度参数Yim等人[154]将教师模型网络层之间的数据流信息作为息,定义为两层特征的内积Chen等人[155将教师模型在某一类的不同样本间的排序关系作为学习信息传递给学生模型



模型蒸馏直接设计了一个简单结构的小网络，那小网络的准确率怎么和大网络比呢？


模型蒸的主要思想是用预训练好的网络(通常结构较复杂，准确率较高)，来指导小网络的训练，并使小网络达到与复杂网络相近的准确率。

大网络类比于老师，小网络类比于学生，老师经过漫长时间的“训练”摸索出一套适用于某个任务的方法，于是将方法提炼成“知识”传授给学生，帮助学生更快地学会处理相似的任务。

整个思想中最大的难题在于如何有效地表达“知识”，并有效地指导小网络的训练。


## 背景

集成来提升任务性能，耗时耗力，不利于部署。
将知识压缩到方便部署单个模型是可行的，性能相近。

distill 压缩模型，利用大模型生成的类别概率作为soft targets，待压缩 hard targets。

61.1%
60.8%

旨在把一个大模型或者多个模型ensemble学到的知识迁移到另一个轻量级单模型上，方便部署。简单的说就是用新的小模型去学习大模型的预测结果，改变一下目标函数。听起来是不难，但在实践中小模型真的能拟合那么好吗？所以还是要多看看别人家的实验，掌握一些trick。[3]

知识蒸馏(knowledge distillation，KD)是指对于一个训练好的较大的teacher net，训练一个较小的student net去拟合teacher net的输出(分布)：[8]

蒸馏的目标是让student学习到teacher的泛化能力，理论上得到的结果会比单纯拟合训练数据的student要好。另外，对于分类任务，如果soft targets的熵比hard targets高，那显然student会学习到更多的信息。


## Transfer Set和Soft target

实验证实，Soft target可以起到正则化的作用（不用soft target的时候需要early stopping，用soft target后稳定收敛）
数据过少的话无法完整表达teacher学到的知识，需要增加无监督数据（用teacher的预测作为标签）或进行数据增强，可以使用的方法有：1.增加[MASK]，2.用相同POS标签的词替换，2.随机n-gram采样，具体步骤参考文献2

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

$$L_{TS}={\frac{1}{2}}\Vert{u_{Teacher}-r_{student}}\Vert ^2$$

- $L_{TS}(Block)$ 表示指导损失函数
- $u_{Teacher}$ 表示大网络输出特征图
- $r_{student}$ 表示小网络的输出特征图

整体网络的损失函数如下式所示：

$$L_{total} =\lambda L_{orig}+（1-\lambda） L_{TS}$$

- $L_{orig}$ 为直接训练网络的损失函数
- $\lambda$ 为提前设定的超参数，表示大网络对小网络指导损失函数的重要性
- 对于 $\lambda$ 的取值：

- 当 $\lambda$ 过小时，总损失函数与原损失函数几乎相同
- 当 $\lambda$ 过大时，总损失函数与指导损失函数几乎相同，每次迭代的参数更新值几乎全部取决于指导损失函数，这种训练将完全陷入模仿训练误区。此时，小网络学习重点偏向于模仿大网络而忽略了任务本身，导致实际训练效果下降甚至发生错误。
- 推荐 $0.1至0.5$


“蒸馏”最简单的形式就是：以从复杂模型得到的“软目标”为目标（这时T比较大），用“转化”训练集训练小模型。训练小模型时T不变仍然较大，训练完之后T改为1。

知ton等人[153]首先提出使用教师模型的类别概率输出计算 soft target为了方便计算还引入温度参Yim等人[154]将教师模型网络层数据流信息作定义为两层特征的内积Chen等将教师模型在某一类的不同样本间的排序关系作信息传递给学生模型训练技巧Czarnecki等人[156提出 Sobole训练方法,将目标函数的导数融入到神经网络函数逼近器的训练练数据由于隐私等问题对于学生可用时opes等人[157]提出如何通过 extra metadata解决zhou等人58]主要有两点创新第—不用预训练教师模型而是教师模型和学生模型同时训缭;第二教师模型和学生模型共享网络参数4其他场限制知识蒸馏法被局限于分类任务的使用场景.但近年来研究人员提出多种策略使其能应用于其他深度学习场景在目标检测任务提出匹万法,C用[148]和[153]提出的方法提升多分类目标检测网络的性能.在解决人脸检测任务时,Luo等提出将更隐层的神经元作知识其与类别输出概率信息量相同但更紧凑. Gupta等人[162]提出跨模态迁移知识的做法将在RGB数据集学习到的知识迁移到深度学习的场提出一种多任务指导预测和蒸馏网络( PAD-Net)结构,产生一组中间辅助任务为标任务提供丰富的多模态数据

由于softmax层的限制,知识蒸馏法被局限于分类任务的使用场景能应用于其他深度学习场景.在目标检测任务中,Li等人用[148]和[153]提出的方法,提升多分类目标检测网络的性能高隐层的神经元作为学习知识,其与类别输出概率信息量相同的做法,将在RGB数据集学习到的知识迁移到深度学习的场景中馏网络(PAD-Net)结构,产生一组中间辅助任务,为学习目标任务提供丰富的多模态数据2.7  混合方式以上这些压缩与加速方法单独使用时能起到很好效果补充.研究人员通过组合使用不同的压缩与加速方法或者针对不同网络层选取不同的压缩与加速方法体化的压缩与加速框架,能够获得更好的压缩比与加速效果合使用,极大降低模型的内存需求和存储需求,方便模型部署到计算资源有限的移动平台紧凑网络组合使用,为学生模型选择紧凑的网络结构够综合各类压缩与加速方法的优势,进一步加强压缩与加速效果的重要研究方向.2.7.1  组合参数剪枝和参数量化Ullrich等人[165]基于Soft  weight  sharing的正则化项在枝.Tung等人[166]提出参数剪枝和参数量化的一体化Pruning-Quantization(CLIP-Q).如图7所示,Han等人夫曼编码结合,达到很好的压缩效果,并在其基础上考虑到软硬件的协同压缩设计Engine(Eie)框架[168].Dubey等人[169]同样利用这三种方法的组合进行网络压缩Fig.7    Theflow chart of 图7    Deep Compression[13首先提出使用教师模型的类别概率输出计算soft target,为了方便计算还引入温度参将教师模型网络层之间的数据流信息作为学习信息,定义为两层特征的内积.Chen等人[155]将教师模型在某一类的不同样本间的排序关系作为学习信息传递给学生模型.将目标函数的导数融入到神经网络函数逼近器的训练中.当训,Lopes等人[157]提出如何通过extra  metadata解决.Zhou等人而是教师模型和学生模型同时训练；第二教师模型和学生模知识蒸馏法被局限于分类任务的使用场景.但近年来,研究人员提出多种策略使其等人[159]提出匹配proposal的方法,Chen等人[160]结合使提升多分类目标检测网络的性能.在解决人脸检测任务时,Luo等人[161]提出将更其与类别输出概率信息量相同,但更紧凑.Gupta等人[162]提出跨模态迁移知识数据集学习到的知识迁移到深度学习的场景中.Xu等人[163]提出一种多任务指导预测和蒸为学习目标任务提供丰富的多模态数据


[1]: 蒸馏开山鼻祖Hinton@NIPS2014：Distilling the Knowledge in a Neural Network https://arxiv.org/abs/1503.02531 Distilling the Knowledge in a Neural Network
[2]: https://ai.deepshare.net/detail/v_5f164b66e4b0aebca61a59e3/3?from=p_5ee641d2e8471_5z8XYfL6&type=6
[3]: https://zhuanlan.zhihu.com/p/71986772?utm_source=wechat_session&utm_medium=social&utm_oi=772887009306906624&utm_campaign=shareopn
[4]: BERT -> 单层LSTM：Distilling Task-Specific Knowledge from BERT into Simple Neural Networks https://arxiv.org/abs/1903.12136
[5]: MT-DNN ensemble -> MT-DNN：Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding https://arxiv.org/abs/1904.09482
[6]: Google Single-task ensemble -> Multi-task：BAM! Born-Again Multi-Task Networks for Natural Language Understanding https://arxiv.org/abs/1907.04829
[7]: Huawei -> TinyBERT: Distilling BERT for Natural Language Understanding https://arxiv.org/abs/1909.10351
[8]: https://0809zheng.github.io/2020/05/01/network-compression.html
[9]: https://www.zhihu.com/question/305220135/answer/552545851
[10]: https://www.hhyz.me/2018/06/26/ModelCompression/
补充一些资源，还没仔细看：

[dkozlov/awesome-knowledge-distillation](https://github.com/dkozlov/awesome-knowledge-distillation)
[Distilling BERT Models with spaCy](http://www.nlp.town/blog/distilling-bert/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter)
[DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)
[Multilingual MiniBERT: Tsai et al. (EMNLP 2019)](https://arxiv.org/pdf/1909.00100)
BERT蒸馏完全指南｜原理/技巧/代码: https://zhuanlan.zhihu.com/p/273378905
https://github.com/FLHonker/Awesome-Knowledge-Distillation

https://github.com/peterliht/knowledge-distillation-pytorch
https://github.com/AberHu/Knowledge-Distillation-Zoo
