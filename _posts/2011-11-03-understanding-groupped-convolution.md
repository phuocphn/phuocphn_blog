---
layout: post
title:  "A Deeper Understanding of Grouped Convolution."
date:   2021-11-03 22:24:00 +0900
categories: ml
tags: convolution
author: Phuoc. Pham
---

### **What Is Grouped Convolution ?**
Grouped Convolution is a method of performing convolution operation independently by **dividing input channels into several groups**. The benefits of this method are simple implementation and also offer the advantageous for parallel processing.

<p align="center">
  <img src="https://i.ibb.co/3WBSbbP/grouped-conv.png">
</p>



The idea of grouped convolution was originally introduced in AlexNet paper. The motivation is to allow the training of the network over two Nvidia GTX 580 GUS with 1.5GB of memory each. (The model requires under 3GB of GPU RAM for training). Groups convolution allowed more efficient model-parellization across the GPUs.

The architecture of AlexNet as illustrated in the original paper, showing two separate convolutional filter groups across most of the layers (Alex Krizhevsky et al. 2012). The **<span style="color:green;">green color rectangle</span>** is the set of 48 kernels with size 11 x 11 and **<span style="color:red">the red color rectangle</span>** is the **different set of 48 kernels** with size 11 x 11.

![AlexNet Architecture](https://i.ibb.co/10rvpMt/alexnet-paper.png)




**But it is not just an engineering hack...**

There is an interesting side-effect to this engineering hack, the `conv1`  filters being easily interpreted, it  noted that filter groups seemed to consistently divide  `conv1`  into two seperate and distinct tasks: black-white filters and colour filters.

<ins>AlexNet  `conv1`  filter separation</ins>: as noted by the authors, filter groups appear to structure learned filters into two distinct groups, black-and-white and colour filters ([Alex Krizhevsky et al. 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)).
What isn't noted explicitly in the AlexNet paper is the more important side-effect of convolutional groups, that **it learns better representations**. This is an empirical evidence from the authors without much theoretical explanation.

AlexNet trained with varying numbers of filter groups, from 1 (i.e. no filter groups), to 4. When trained with 2 filter groups, AlexNet is more efficient and yet achieves the same if not lower validation error.

![ValidationEror](https://i.ibb.co/QJsm4NG/validation-error.png)

### **How does grouped groups work ?**

The following picture is the illustration for the nomarl convolution layer, with no filter groups.  We have $$c_1$$  is `#input’s channels`, and $$c_2$$  is `#filters` or `#output channels`.

![StandardConvolutionLayer](https://i.ibb.co/2tdDNqL/image-49.png)




Let's give an concrete example to see how grouped convolution layer work. For the convolutonal layer with 2 filter groups (or `groups = 2`). This is a grouped convolution layer, it will work as follows:

![GroupedConvWorking](https://i.ibb.co/t87Ywsy/image-50.png)

Firstly, it will group filters into 2  groups $$\frac{c_2}{groups}=\frac{\#num\_filters}{groups=2}$$, and also we need to group the input’s channels into **2  groups**. 

Secondly,  the **first** of $$\frac{c_2}{groups}$$ kernels will convolute in the **first** group of input's channels (the output channel results are denoted by  **<span style="color:red;">the red line</span>**). And similarly, the **second** of $$\frac{c_2}{groups}$$ kernels will convolute in the **second** group of input's channels (the output channel results are denoted by  **<span style="color:green;">the green line</span>**). 


#### **Pytorch convention:**

Convolutional Layer:   $$(batch\_size, out\_channels, height, width ) \to (batch\_size, \frac{out\_channels}{groups=2}, height, width)$$


Input :  $$(batch\_size, in\_channels, height, width) \to (batch\_size, \frac{in\_channels}{groups=2}, height, width)$$

**A simple way of explanation when `groups=2`**:  Instead of convoluting a deep filter over the input chanel, we can convolute 2 times, the first time with the half of the filter over half of the input channels, the second time with the remaining half of the filter over the remaining of the input channels.

In short,  with grouped convolution ~ rather than creating filters with the full channel depth of the input, the input is split channel-wise into groups. It was discovered that using grouped convolutions led to a degree of specialization among groups where separate groups focused on different characteristics of the input image.

**Understanding the groups parameter in nn.Conv2d** 

So, for $$K$$ groups in the setting of `Conv2d`, we <ins>must ensure that the number of input channels and the number of output channels are both divisible by</ins> $$K$$.

Say you want to take `(1, 3, 64, 64)` input image (tensor) and produce `(1, 9, 64, 64)` output tensor. You can have at most 3 groups (since 9mod3 = 0, 3mod3=0) in this situation and each channel will be convolved with `9/3=3 filters`.

So, if you have `(N, M, H, W)` input shape where $$N$$ is the batch size and $$M$$ is the number of input channels, and you want to produce $$(N, L, H’, W’)$$ output, the maximum number of groups you can have the **Greatest Common** Divisor of these two numbers.


#### **A Visualization Example**

In this following we have the  $$in_channels = 4 ,  number\_of\_kernels = out\_channels = 8$$. We can observe there are lot of parameters for kernels, assume each kernel has shape 3 x 3, the total learnable parameters would be  `3 * 3 * 4 * 8 = 288` 

{% highlight python %}
conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), groups=1)
input = torch.randn(1, 4, 32, 32) # (batch_size, in_channels, height, width)
>>> conv.weight.shape
torch.Size([8, 4, 3, 3])
{% endhighlight %}

When we set groups=2

The input channels will be divided by `groups=2` into  $$in\_channels = \frac{4}{groups=2} = 2$$.

The total number of kernels is still the same ( number_kernels = 8) but for each kernel, the depth is divided by `groups = 2` , and  $$in\_channels = \frac{4}{groups=2} = 2$$.

The total learnable parameters:  `3 * 3 * 2 * 8 = 144`

{% highlight python %}
conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), groups=2)
input = torch.randn(1, 4, 32, 32) # (batch_size, in_channels, height, width)
>>> conv.weight.shape
torch.Size([8, 2, 3, 3])
{% endhighlight %}


- Filters will be divided into 2 groups (groups=2), each group will have  $$\frac{8}{2} = 4$$  filters
- The depth of filters will be reduced by factors of 2 (groups=2)
- The input channels will be divided into 2 groups (groups=2), each groups contains  $$\frac{4}{groups=2} = 2$$ channels.
- <ins>The first group filters</ins> ( 4 filters of 3×3) will convolute over the <ins>the first group of input</ins> (2 x h x w)
- <ins>The second group filters</ins> ( 4 filters of 3×3) will convolute over the <ins>the second group of input</ins> (2 x h x w)

If the input image has 3 channels, and the convolutional layer is defined as follows:

{% highlight python %}
conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3, 3), groups=3)
input = torch.randn(1, 3, 32, 32) # (batch_size, in_channels, height, width)
>>> conv.weight.shape
torch.Size([9, 1, 3, 3])
{% endhighlight %}




**Formal Analysis with Pytorch nn.Conv2d Layer**

For example, we have a convolution layer is defined as follows:


`nn.Conv2d(in_channels, cardinality * bottlenect_width, kernel_size=3, stride=1, padding=1, groups=cardinality)`


Input $$\textbf{X}$$ will be divided into  cardinality  sub-inputs $$\textbf{X}_1, \textbf{X}_2, \textbf{X}_3, ... , \textbf{X}_{cardinality}$$ , each sub-input $$\textbf{X}_i$$ will have $$\frac{in\_channels}{cardinality}$$  channels.

Kernel filters will be grouped into cardinality  groups $$\textbf{G}_1,\textbf{G}_2,.... \textbf{G}_{cardinality}$$. For each group $$\textbf{G}_i$$ will have  $$\textbf{M} = \frac{out\_channels}{cardinality}$$  filters denoted as:  $$\textbf{G}_i^{(1)}, \textbf{G}_i^{(2)}, \textbf{G}_i^{(3)}, ..., \textbf{G}_i^{(M)}$$

Each filter in the kernel group  $$\textbf{G}_i$$  will convolute over the corresponding sub-input $$\textbf{X}_i$$ , results in  $$M$$  output feature maps.

We have total  caridinality  groups, therefore we have  $$\textbf{M} * cardinality = \frac{out\_channels}{cardinality} * cardinality = out\_channels$$  output channels (which are concatenated).


### **References**
1. [A Tutorial on Filter Groups (Grouped Convolution)](https://blog.yani.io/filter-group-tutorial/)
2. [How groups work in PyTorch convolutions](https://mc.ai/how-groups-work-in-pytorch-convolutions/)
3. [Conv2d certain values for groups and out_channels don’t work](https://discuss.pytorch.org/t/conv2d-certain-values-for-groups-and-out-channels-dont-work/14228)