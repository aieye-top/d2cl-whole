
# Pretrain


```{.python .input  n=1}
import torch
torch.hub.list('pytorch/vision')
```

https://github.com/chsasank/pytorch-hub-model-zoo

```{.python .input  n=2}
import torch
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)

model.eval()
input_tensor = torch.rand(1,3,224,224)

script_model = torch.jit.trace(model,input_tensor)
script_model.save("mobilenet-v2.pt")
```

---

https://github.com/topics/pretrained-language-model
https://github.com/Separius/awesome-sentence-embedding

[2]: https://github.com/johnolafenwa/PytorchMobile/blob/master/convert.py
