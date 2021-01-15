# save_load

在训练完成之前，我们需要每隔一段时间保存模型当前参数值，一方面可以防止断电重跑，另一方面可以观察不同迭代次数模型的表现；在训练完成以后，我们需要保存模型参数值用于后续的测试过程。所以，保存的对象包含网络参数值、优化器参数值、epoch值等等。

```py
model.load_state_dict(torch.load("models/net.pth"))
print(model.state_dict())
```

pytorch中state_dict()和load_state_dict()函数配合使用可以实现状态的获取与重载，load()和save()函数配合使用可以实现参数的存储与读取。

[1]: 一文梳理pytorch保存和重载模型参数攻略 - 机器学习入坑者的文章 - 知乎 https://zhuanlan.zhihu.com/p/94971100
