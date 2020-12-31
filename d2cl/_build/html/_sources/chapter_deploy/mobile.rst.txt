
mobile
======

https://mp.weixin.qq.com/s/bndECrtEcNCkCF5EG0wO-A

移动端机器学习资源合集

Embedded and mobile frameworks are less fully featured than full
PyTorch/Tensorflow Have to be careful with architecture Interchange
format Embedded and mobile devices have little memory and
slow/expensivecompute Have to reduce network size /quantize weights
/distill knowledge

Embedded and mobile devices have low-processor with little memory, which
makes the process slow and expensive to compute. Often, we can try some
tricks such as reducing network size, quantizing the weights, and
distilling knowledge. Both pruning and quantization are model
compression techniques that make the model physically smaller to save
disk space and make the model require less memory during computation to
run faster. Knowledge distillation is a compression technique in which a
small “student” model is trained to reproduce the behavior of a large
“teacher” model. Embedded and mobile PyTorch/TensorFlow frameworks are
less fully featured than the full PyTorch/TensorFlow frameworks.
Therefore, we have to be careful with the model architecture. An
alternative option is using the interchange format. Mobile machine
learning frameworks are regularly in flux: Tensorflow Lite, PyTorch
Mobile, CoreML, MLKit, FritzAI. The best solution in the industry for
embedded devices is NVIDIA. The Open Neural Network Exchange (ONNX for
short) is designed to allow framework interoperability.[2]

Deploy

Take the compressed flite file and load itinto a mobile or embedded
device

[1]: [2]:
https://course.fullstackdeeplearning.com/course-content/testing-and-deployment/hardware-mobile
