# Edge

对5G+AI时代的新型算力平台（边缘计算）与网络连接（算力网络）进行了系统性的介绍。

边缘计算的定义是：在靠近物或数据源头的网络边缘侧，融合网络、计算、存储、应用核心能力的分布式开放平台（架构），就近提供边缘智能服务，满足行业数字化在敏捷连接、实时业务、数据优化、应用智能、安全与隐私保护等方面的关键需求。它可以作为连接物理和数字世界的桥梁，使能智能资产、智能网关、智能系统和智能服务。

边缘计算是在网络边缘提供计算、存储等资源，通过分布式的系统满足上层应用对时延等性能指标的要求，同时降低资源消耗，降低综合支出。因此边缘计算无法像传统云计算那样通过集中化、规模化管控，需要找到新的途径实现对离散资源的管控与资源互通，并实现网络资源和计算资源的协同调度，从而向用户提供有质量保证的服务。[1]

## 算力

用于AI和图形处理的每秒浮点运算次数，FLOP/s），智能社会对算力的需求主要是浮点运算能力，专用AI芯片如华为昇腾910采用7nm工艺，半精度FP16算力达256TFLOP/s，低功耗的12nm芯片昇腾310半精度FP16算力也达到了8 TFLOP/s。过去5年，随着深度学习算法的演进，AI训练对算力的需求增加了30万倍，一些互联网厂家已经将算力作为服务提供给用户，从1 FP 32 TFLOP/s或8 FP 16 TFLOP/s到4FP 32 TFLOP/s或32 FP 16 TFLOP/s的AI推理加速服务，简单的语音语义识别或单流视频分析 8 FP16 TFLOP/s即可满足，复杂的推荐引擎或风险检测则需要32 FP 16 TFLOP/s的算力[2]

## 新型算力平台：边缘计算

中国电信在深圳召开的5G创新合作大会上对外展示了自主研发的基于分布式开放平台的多接入边缘计算（Multi-access Edge Computing，MEC）平台。该平台就近提供边缘智能服务，支持固定/移动网络接入、第三方能力/应用灵活部署及边缘能力统一开放，可应用于工业互联网、高清视频、车联网等行业。[3]

边缘计算凭借“边缘”的特性，可以更好地支撑云端的应用，而云计算则能够基于大数据分析，完成边缘节点无法胜任的计算任务，助力边缘计算更加满足本地化的需求。

## 端计算端

即用户终端，如PC、手机和物联网终端设备等。用户终端设备具有一定的计算能力，能够对采集的数据进行实时处理，进行本地优化控制、故障自动处理、负荷识别和建模等操作。在和网络进行连接后，用户终端设备可以把加工汇集后的高价值数据与云端进行交互，在云端进行全网的安全和风险分析、大数据和人工智能的模式识别、节能和策略改进等操作。同时，如果遇到网络覆盖不到的情况，可以先在边缘侧进行数据处理，当有网络时再将数据上传到云端，在云端进行数据存储和分析[3]。

## 算力网络

算力网络是一种通过网络分发服务节点的算力信息、存储信息、算法信息等，结合网络信息（如路径、时延等），针对用户需求，提供最佳的资源分配及网络连接方案，并实现整网资源最优化使用的解决方案。算力网络将具备以下四个基本特征。

1. 资源抽象：算力网络需要将计算资源、存储资源、网络资源（尤其是广域范围内的连接资源）及算法资源等都抽象出来，作为产品的组成部分提供给用户。
2. 业务保证：以业务需求划分服务等级，而不是简单地以地域划分，向用户承诺诸如网络性能、算力大小等服务等级的协议（Service-LevelAgreement，SLA），屏蔽底层的差异性（如异构计算、不同类型的网络连接等）。
3. 统一管控：统一管控云计算节点、边缘计算节点、网络资源（含计算节点内部网络和广域网络）等，根据业务需求对算力资源及相应的网络资源、存储资源等进行统一调度。
4. 弹性调度：实时监测业务流量，动态调整算力资源，完成各类任务，高效处理和整合输出，并在满足业务需求的前提下实现资源的弹性伸缩，优化算力分配

## KubeEdge[4]

KubeEdge构建于Kubernetes之上，是将Kubernetes原生的容器编排能力扩展到了边缘节点上，并增加了对边缘设备的管理功能。它由云端部分和边缘部分组成，核心基础架构提供了对网络、应用部署和云边之间元数据同步的支持。它同时支持MQTT，使得边缘设备可以通过边缘节点接入集群。
https://github.com/kubeedge/kubeedge

[1]: https://weread.qq.com/web/reader/eab32840721a4865eab660dka87322c014a87ff679a21ea
[2]: https://weread.qq.com/web/reader/eab32840721a4865eab660dk16732dc0161679091c5aeb1
[3]: https://weread.qq.com/web/reader/eab32840721a4865eab660dk8f132430178f14e45fce0f7
[4]: https://weread.qq.com/web/reader/eab32840721a4865eab660dkc0c320a0232c0c7c76d365a