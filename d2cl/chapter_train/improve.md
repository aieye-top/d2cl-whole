# 改进

试想一下，当你构建了一个准确率已达90%的机器学习系统，我们应该怎样进一步改进系统以取得实际应用？可能你有很多想法：收集更多的训练数据；保持训练数据集的多样性；使用更大规模的神经网络；使用不同的优化算法；采用dropout层或使用L2正则化；更改神经网络的架构等。面对如此多的选择，我们究竟应该如何抉择？哪一种选择是最有希望改善系统性能的？如果你的团队花费了近半年的时间去收集数据，结果发现更多的数据并不是该系统性能提升的决定性因素，这样你们就浪费了大量如同你生命一样宝贵的时间。

## 策略一：正交化过程

构建机器学习系统的最大挑战包括调参，调参的过程中你有许多种选择，如果时间允许的话，你可以各种尝试。但是经验表明，最高效的机器学习人员他们很清楚什么样的方式最有效，这就要用到正交化的思想。举个例子，一台老式的电视机，上面有很多旋钮，你可以采用各种方式调整照片。但是，最有效的调整方式就是，每个旋钮只拥有一个功能，然后再调整旋钮。电子工程师已经花费了大量的时间来保证每个设计的旋钮都有相对可解释的功能：一个旋钮控制图像的高度，一个旋钮控制图像的宽度，一个旋钮控制视频的左右位置，其它的分别控制图像的旋转、平移——这样的设置确保了人能更加容易地操纵图像；如果一个旋钮控制着所有的功能，你几乎不可能调整电视图像。这就是正交化的设计思维，它能确保我们的操作简单、具有可解释性。那么，这与机器学习有什么关系呢？

在构建有监督机器学习的系统中，我们也在做类似的过程。首先，我们必须确保我们的系统能在训练集上表现很好，然后在交叉验证集上，最后是测试集。如果你的算法在训练集上表示不好，这时你需要一个专用的“旋钮”来解决这个问题，这可能是你需要训练更大的神经网络，也或许是要采用更好的优化算法，如 Adam 优化算法；如果你的系统在交叉验证集上表示不好，那么你也需要开发出另一组正交化的“旋钮”来解决这个问题，这可能需要你采用更大的训练集，这有助于你的机器学习算法更好地泛化到交叉验证集上；如果你的系统在交叉验证集上也表现很好，但测试集上性能较差又怎么办呢？在这种情况下，你应该调整旋钮得到一个更大的交叉验证集；如果你的系统在测试集上也表现不错，但是用户体验较差，这意味着你可能需要回去重新选择交叉验证集或成本函数。

在构建高性能的机器学习系统时，我们应该清楚哪个模块出现了问题，针对这些不同的模块采取不同的措施，机器学习系统的构建应充分考虑正交化的概念，确保各个流程、各个功能不交叉，保持相对的独立性，具有可解释的功能性。

## 策略二：确定单一的模型性能评价指标和选择合适的训练集、交叉验证集/测试集

无论你是调整超参数，还是尝试不同的学习算法，甚至是使用不同的机器学习模型，你都需要确定一个模型性能的评价指标。有了性能评价指标，你能快速地判断出新的尝试是否有效，这会加速你构建机器学习系统的步伐。应用机器学习需要丰富的经验，一个比较好的做法是，先编写代码，然后运行代码查看实验结果，并据此来继续改进你的算法。比如，你先建立了一个简单的分类器A，然后通过调整模型中的某些超参数或一些其它东西，你可以训练出一个新的分类器B，然后根据性能评估指标评估你分类器的优劣。

采用精确率和召回率评估分类器的性能可能是合理的，但是可能会出现这样的问题：分类器A上的精确率更高，这意味着分类器A的预测结果准确率更高；分类器B的召回率更高，意味着分类器B的误分类概率更低。那么分类器A和分类器B谁更优呢？另外，如果你尝试许多想法，采用不同的参数，这通常会得到更多的分类器，如何从这些分类器中选择最好的分类器呢？这时如果仍然使用两个指标衡量模型性能，就显得不太合理，所以最好的方式是综合考虑精确率和召回率，如F1分数就是这样的一个指标。采用这样的评估指标，会加速机器学习的迭代过程。

有时，我们并不能只是关注精确率、召回率，甚至是F1分数，我们需要考虑训练机器学习系统的更多限制因素，如时间、金钱。这时，单一的性能评估指标并不能满足我们的要求。更一般地说，我们需要在多个目标中确定出一个最合理的优化目标，将其它目标或条件作为限制性因素，然后在训练集或交叉验证集上或测试集上进行评估或计算，选出在有限制的条件下的最优算法。

训练集、交叉验证集和测试集的选择对构建机器学习应用产品具有巨大的影响。那么，我们应该怎样选取这些数据集来提高团队开发机器学习系统的效率？训练机器学习模型时，通常会尝试许多不同的想法，在训练集上训练不同的模型，然后使用交叉验证集验证你的想法并选择一个模型。继续保持训练，在验证集上提高你的模型，最后在测试集上评估模型。

事实证明，交叉验证集和测试集应该来自于同一数据分布。确定好单一的性能评估指标和交叉验证集后，机器学习团队就可以快速地进行创新，尝试不同的想法，运行实验，选择最好的模型。所以建议是，在划分数据集前，先将数据均匀混洗，再划分交叉验证集和测试集，这样就能保证两个数据集来自相同的分布。

事实证明，验证集和测试集的规模选择也正在改变机器学习领域。你可能听说过机器学习领域内70/30的经验法则之一，即将所有的数据按70：30的比例划分为训练集和测试集，或者将数据集的60%划为训练集，20%划为交叉验证集，20%划为测试集。这在早期的机器学习领域相当合理，尤其是当数据量很少的时候。

但是，现代的机器学习领域，数据集的规模极其庞大，这种情况下 70/30 的经验规则不再适用。假如你有 100 万个样例，可以选取其中的 98% 作为训练集， 1%作为交叉验证集，剩余 1%作为测试集才合理。这是因为其中的 1% 也有 10000 个样例，对于测试集和交叉验证集已经足够了。所以在现代大规模数据集的机器学习或深度学习过程中，测试集和交叉验证集的划分比例远远小于 20%。另外，深度学习算法对数据的需求量极大，所以大部分的数据都可以划分入训练集。而使用测试集的目的就是验证最终机器学习系统的可靠性，增加你对机器学习系统整体性能的信心。除非你需要非常准确地衡量系统的最终表现，否则你的测试集不需要数以百万计的样例，成千上万的数据足矣。

## 策略三：调整性能评估指标和交叉验证集/测试集

选择交叉验证集和评估指标就如同打靶。很多时候，你的项目已经进行了一部分，你才发现你将目标选错了，这如同你在打靶过程看错了靶子的位置。因此，建立机器学习系统时，第一步应该是确定评估指标，哪怕确定的评估指标并不一定最好，但你要有这个意识；然后采用正交化过程中的各个步骤，使算法能很好地工作。

例如，你建立了两个识别猫的分类器A和B，它们在交叉验证集上的分类错误率分别是3%，5%，但当你在实际中测试你的算法的时候，你发现分类器B却做的更好——这可能是你选择的交叉验证集是高质量的猫图片，而建立的机器学习应用程序要处理的却是大量低质量的猫图像，这些图片不仅模糊，也可能包含各种奇怪的猫表情。这种情况下，你就需要改变你的交叉验证集以使你的数据能更好地反映出你实际情况。

总而言之，如果目前的评估指标和数据不能很好地应对你真正关心的事情，这就是改变你的评估指标和/或交叉验证集/测试集的好时机。这会使你的算法更好地捕获你所关心的信息；这对于确定算法的优劣，加速想法的迭代过程，提高算法效率很有帮助。

## 策略四：将系统的表现与人类的表现相比，确定提升系统性能的方法

我们经常发现，当机器学习系统在某些方面超越人之前，系统的性能提升最快，在超越人之后，进步就变得缓慢。人类非常擅长分类，识别事物，听音乐，做阅读理解。当系统的性能超越人之前，人类总能使用特定的策略帮助改善系统的性能。这些策略包括使用更多标签化的数据；从误差分析中获得启示；进行偏差和方差分析。当系统的表现超越人之后，这些策略将更难奏效。

这就是为什么要将机器学习系统的表现行为与人类的表现行为相比，尤其是在人类擅长的任务上。一个监督的机器学习系统要正常工作，从根本上来说，要保证两个方面：第一，很好地拟合训练集，低偏差；第二，训练的系统要有很强的泛化能力，即要在交叉验证集或测试集上很好地工作，低方差。

在提升机器学习系统性能之前，首先要明白机器学习系统的训练误差与人类误差之间的差异，辨别是否存在高偏差。解决高偏差可以使用以下几种方法：使用更大的神经网络模型；训练更长的时间；使用 ADAM，Momentum，NAG,Adagrad, Adadelta,RMSProp 等更好的算法；调整神经网络架构，这包括神经网络的层数、神经元的连接方式、神经元个数，激活函数等超参数。然后，比较交叉验证集或测试集与训练集上的误差，确定是否具有高方差。方差的高低意味着你的模型是否具有很好的泛化能力。遇到高方差，可以采用以下几种方式解决：收集更多的数据训练模型；使用正则化方法，这包括采用dropout，数据增强，权重正则化等；调整神经网络结构等。

[1]: http://www.mittrchina.com/news/973