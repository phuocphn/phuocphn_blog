---
layout: post
title:  "FLOPs calculation in Convolutional Neural Network."
date:   2021-11-01 14:00:00 +0900
categories: ml
tags: tnse, reduction
author: Phuoc. Pham
comment: true
---


#### **Different of FLOPS and FLOPs**



FLOP**S**: Abbreviation for floating point operations per second, which means the number of floating-point operations per second, which is understood as the calculation speed. Is a measure of hardware performance.



FLOP**s**: Abbreviation for floating point operations (s table plural), meaning floating-point operands, understood as the amount of calculation. Can be used to measure the complexity of the algorithm/model.

<p align="center">
  <img src="https://images4.programmersought.com/992/d8/d83e5a1f963df52826e8592241dc8358.png">
</p>

 
The following of this article will show the formula for calculation of different layers in a standard convolution neural network.

#### **Convolution Layer**


$$FLOPs = 2 * H * W * (C_{in} * K^{2} + 1) * C_{out} $$

Note this formula is contradict with [2], (*this formula is seem to be the correct way*)

$$FLOPs = [K^{2} * C_{in} ) * C_{out} + C_{out}] * (H * W)$$  

$$FLOPs = [(K^{2} * C_{in}  + 1)  * C_{out}] * (H * W) $$



$$ C_{in} $$ is the number of input channels, $$ C_{out} $$ is the number of output channels, [latex] H, W [/latex] are the height and width of input feature map. $$ K$$ is the kernel size (height &amp; width).


<p align="center">
  <img src="https://www.programmersought.com/images/281/01a524af9415fc43bf6a2d61b4218619.png">
</p>


#### **Fully Connected Layer**



$$ FLOPs = (2 * I - 1) * O $$



$$O$$ is the output dimension, $$I$$ is the input dimension.


We construct 101- layer and 152-layer ResNets by using more 3-layer blocks (Table 1). Remarkably, although the depth is significantly increased, the 152-layer ResNet (11.3 billion FLOPs) still has lower complexity than VGG-16/19 nets (15.3/19.6 billion FLOPs)

**What is the relationship between GMACs and GFLOPs?** 


>I think GFLOPs = 2 * GMACs as general each MAC contains one multiplication and one addition.
><sup>https://github.com/sovrasov/flops-counter.pytorch/issues/16#issuecomment-567327330</sup>








#### **Calculate memory usage.**


There is a inconsistent results between articles. For more details, please refer to the following link [Floating point operations per second (FLOPS) of Machine Learning models ](https://iq.opengenus.org/floating-point-operations-per-second-flops-of-machine-learning-models/)





**FLOPS of VGG models**

- VGG19 has 19.6 billion FLOPs
- VGG16 has 15.3 billion FLOPs


**FLOPS of ResNet models**

- ResNet 152 model has 11.3 billion FLOPs
- ResNet 101 model has 7.6 billion FLOPs
- ResNet 50 model has 3.8 billion FLOPs
- ResNet 34 model has 3.6 billion FLOPs
- ResNet 18 model has 1.8 billion FLOPs


**FLOPS of GoogleNet and AlexNet**
- GooglenNet model has 1.5 billion FLOPs
- AlexNet model has 0.72 billion FLOPs


#### **Implementation**


```python


import torch
import torch.nn as nn
import torchvision.models as models
import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np


__all__ = ['AlexNet', 'alexnet']

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model




class ModelBenchmark(object):

    def __init__(self, net, batch_data):
        self.net = net
        self.batch_data = batch_data

        self._module_inputs = {}
        self._module_outputs  = {}
        self._module_infos  = {}

        self.add_flops_counter_hook()


    def flops_counter_hook(self, name):
        def hook(conv_module, input, output):
            self._module_inputs[name] = input[0].detach().shape
            self._module_outputs[name] = output.detach().shape
            self._module_infos[name] = conv_module

        return hook

    def add_flops_counter_hook(self):
        for n, m in self.net.named_modules():
          # print ("checking...", type(m))
          # print ("assert...", isinstance(m, torch.nn.modules.linear.Linear))
          if isinstance(m, nn.Conv2d) or \
          		isinstance(m, nn.Linear) or \
          		isinstance(m, nn.ReLU) or \
          		isinstance(m, nn.BatchNorm2d):
              m.register_forward_hook(self.flops_counter_hook(n))


    def flops_to_string(self, flops, units='GFLOPs', precision=2):
        if units is None:
            if flops // 10**9 > 0:
                return str(round(flops / 10.**9, precision)) + ' GFLOPs'
            elif flops // 10**6 > 0:
                return str(round(flops / 10.**6, precision)) + ' MFLOPs'
            elif flops // 10**3 > 0:
                return str(round(flops / 10.**3, precision)) + ' KFLOPs'
            else:
                return str(flops) + ' FLOPs'
        else:
            if units == 'GFLOPs':
                return str(round(flops / 10.**9, precision)) + ' ' + units
            elif units == 'MFLOPs':
                return str(round(flops / 10.**6, precision)) + ' ' + units
            elif units == 'KFLOPs':
                return str(round(flops / 10.**3, precision)) + ' ' + units
            else:
                return str(flops) + ' FLOPs'

    def human_size(self, bytes, \
    	units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
        """ Returns a human readable string representation of bytes """
        return str(bytes) + units[0] if bytes < 1024 else \
        		self.human_size(bytes>>10, units[1:])


    def start(self, show_report=True):
        self.net(self.batch_data)

        flops = 0
        required_mem = 0
        for k, v in self._module_infos.items():
            if  isinstance(self._module_infos[k], nn.Conv2d):
                conv_layer = self._module_infos[k]
                conv_weight = conv_layer.weight.shape
                flops +=  ((conv_weight[1] * np.prod(conv_layer.kernel_size)) * \
                	conv_weight[0] + conv_weight[0])  * np.prod(self._module_outputs[k][2:])
                # flops += (conv_weight[1] * conv_weight[2] * conv_weight[3]  + 1) * \self._module_outputs[k][2] * self._module_outputs[k][3] * self._module_outputs[k][1]
                required_mem += np.prod(conv_layer.weight.shape)
            

            elif isinstance(self._module_infos[k], nn.Linear):
                fc_module = self._module_infos[k]
                flops += (fc_module.in_features * fc_module.out_features) + fc_module.out_features
                required_mem += np.prod(fc_module.weight.shape)

            elif isinstance(self._module_infos[k], nn.ReLU):
                flops += np.prod(self._module_inputs[k][2:]) * self._module_outputs[k][1]

            elif isinstance(self._module_infos[k], nn.BatchNorm2d):
                flops += np.prod(self._module_inputs[k]) 
                required_mem += np.prod(self._module_infos[k].weight.shape) 


        print ("Total of FLOPs: ", self.flops_to_string(flops), "~", self.flops_to_string(flops, "MFLOPs"))
        print ("Required Memory: ", self.human_size(required_mem*4))


benchmark = ModelBenchmark(models.resnet34(), batch_data=torch.randn(1, 3, 224, 224))
benchmark.start()
```


#### **References:**

1. [Neural network calculation amount FLOPs](https://www.programmersought.com/article/6010108964/)
2. [FLOPs calculation in CNN](https://www.programmersought.com/article/7565986745/)
3. [MXNet-Python/CalculateFlopsTool](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/MXNet-Python/CalculateFlopsTool)
4. [Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/pdf/1810.00736.pdf)
5. [sovrasov/flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)
6. [PRUNING CONVOLUTIONAL NEURAL NETWORKS FOR RESOURCE EFFICIENT INFERENCE](https://arxiv.org/pdf/1611.06440.pdf)
7. [Swall0w/torchstat](https://github.com/Swall0w/torchstat)
8. [Tramac/torchscope](https://github.com/Tramac/torchscope)
9. [Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
10. [telecombcn-2016-dlcv/slides/D2L1-memory.pdf](http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf)
11. [Memory usage and computational considerations](http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf)
12. [albanie/convnet-burde](https://github.com/albanie/convnet-burden)
13. [networks/model_list/alexnet.py](https://github.com/wogong/pytorch-alexnet/blob/master/networks/model_list/alexnet.py)
14. [lsq_quantizer/flops_counter.py](https://github.com/yashbhalgat/QualcommAI-MicroNet-submission-MixNet/blob/master/lsq_quantizer/flops_counter.py)
15. [How fast is my model?](https://machinethink.net/blog/how-fast-is-my-model/)
16. [pytorch_segmentation_detection/utils/flops_benchmark.py](https://raw.githubusercontent.com/warmspringwinds/pytorch-segmentation-detection/master/pytorch_segmentation_detection/utils/flops_benchmark.py)
17. [pascal_voc/segmentation/flops_counter.ipynb](https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/d5df5e066fe9c6078d38b26527d93436bf869b1c/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/flops_counter.ipynb)
18. [Tramac/torchscope](https://github.com/Tramac/torchscope)
19. [CNN 模型所需的计算力（flops）和参数（parameters）数量是怎么计算的？](https://www.zhihu.com/question/65305385)