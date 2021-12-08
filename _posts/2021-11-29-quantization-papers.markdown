---
layout: post
title:  "INU Research Papers - A Brief Summarization & Implementation."
date:   2021-11-29 21:10:04 +0900
categories: research
tags: quantization
author: Phuoc. Pham
---


This main objective of this article is to make a concise and informative summarization of deep learning research papers that I have read  during my time as a Master student. These papers can be classified into three different categories/topics: Model Compression, Deep Neural Network Generalization Training and Reinforcement Learning.

*Last updated: December 06, 2021*

## **Model Compression Papers**


#### **From Quantized DNNs to Quantizable DNNs**
Propose Quantizable DNNs, a special type of quantized DNNs that can *flexibly adjust its bit-width on the fly* and use only a single optimal set of CNN+FC weights. The authours also introduced `consistency loss` to encourage different bit modes to produce consistent predictions to 32-bit mode. Moreover, `Bit-specific Batch Normalization` is used to alleviate the distribution difference among different bit modes.

Ref: [From Quantized DNNs to Quantizable DNNs](https://www.bmvc2020-conference.com/assets/papers/0400.pdf)


#### **EasyQuant: Post-Training Quantization via Scale Optimization**
Introduce an a simple post-training quantization method (named: EasyQuant) via effectively optimizing the scales of weights and activations alternately.

For entire network optimization, they sequentially optimize scales layer by layer by maximizing the cosine similarity between FP32 and INT8 outputs. The scales of weights and activations are jointly optimized in each layer, and the scales of the next layer are optimized based on the quantized results of the previous 
layers.

Ref: [https://arxiv.org/abs/2006.16669](https://arxiv.org/abs/2006.16669)


#### **Improving Neural Network Quanzation without Retraining using Outlier Channel Splitting.**

DNN weights and activations follow a bell-shaped distribution after training. Extending the quantization range or clipping is naive approaches. OCS solves this problem by identifying a small number of channels containing outliers, duplicates them, then halves the values in those channels. This creates a functionally identicall network, but moves the affected outliers towards the center of the distribution.

Ref: [https://arxiv.org/abs/1901.09504](https://arxiv.org/abs/1901.09504)


#### **Toward Accurate Post-Training Network Quantization via Bit-split and Stitching**

Introduce a novel framework for post-training network quantization.The basic idea is to split integers into multiple bits, then optimize each it, and finally stitch all bits back to integers.

They also proposed `Error Compensated Activation Quantization (ECAQ)` method, which could lower the quantization error for activations. They evaluated the proposed method on classification, object detection, instance segmentation using various neural networks, showing that Bit-split could archive close to full-precison accuracy even for 3-bit quantization, setting new SOTA for post-training quantization.

Ref: [mlr](http://proceedings.mlr.press/v119/wang20c/wang20c.pdf)

#### **Data-Free Quantization Through Weight Equalization and Bias Correction.**

Themain focus is about 8-bit post-training quatization. The proposed quantization relies on equalizing the weight ranges in the network by making use of a scale equivariance property of activation functions (i.e ReLU). In addition the method corrects biases in the error that are introduced during quantization. This improves quantization accuracy performace, and can be applied to many common computer vision architectures with a straight forward API call.

Ref: [https://arxiv.org/abs/1906.04721](https://arxiv.org/abs/1906.04721)


#### **Compressing Neural Networks with the Hashing Trick**

The authors present a novel network architecture, HashedNets that exploits inherent reductions in model sizes. HashedNets uses a low-cost hash function to randomly group connection weights into hash buckets, and all connections within the same hash bucket share a single parameter value.

Ref: [https://arxiv.org/abs/1504.04788](https://arxiv.org/abs/1504.04788)

#### **Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss**

The authours tried to parameterize the quantization intervals and obtain their optimal values by directly minimizing the task loss of the network. (CVPR 2018)

Ref: [cvf.com](https://openaccess.thecvf.com/content_CVPR_2019/papers/Jung_Learning_to_Quantize_Deep_Networks_by_Optimizing_Quantization_Intervals_With_CVPR_2019_paper.pdf)


#### **Differentiable Soft Quantization (DSQ): Bridging Full-Precision and Low-Bit Neural Networks.**
Discrete quantized representation makes the backward propagation can hardly access the accurate gradients and also brings large deviations between the original data and their quantization values, thus often make the training process is more challenging and causes the performance decrease. 

DSQ uses a series of hyperbolic tangent functions to gradually approach the staircase function for low-bit quantization and meanwhile keeps the smoothness for easy gradient calculation.

Ref: [https://arxiv.org/abs/1908.05033](https://arxiv.org/abs/1908.05033)


#### **QKD: Quantization-aware Knowledge Distillation**

Previous works try to combine of quantization + knowledge distillation but it isn't work as desired. (due to the regularization effect of KQ further dimishes the already reduced representation power.

This paper proposes `QKD` technique wherein quantization and KD are carefully coordinated in three phases:

- First, self-studying (SS) phase fine-tunes a quantized low-precision student network without KD to obtain a good initialization.
- Second, co-studying (CS) phase tries to train a teacher to make it more quantization-friendly and poweful than a fixed teacher. (by using the knowledge of the student network).
- Finally, tutoring (TU) phase transfers knowledge from the trained teacher to the student.

Ref: [https://arxiv.org/abs/1911.12491](https://arxiv.org/abs/1911.12491)

#### **Towards Efficient Training for Neural Network Quantization**

The authors discovered two critical rules for efficient training (Efficient Training Rules I and II), recent works violates lead to lower accuracy. They proposed `SAT` to comply with this for efficient training. They archived the SoTA record(2021) by combining SAT+PACT.

#### **AdderNet: Do we really need multiplications in Deep Learning ?**
Multiplication operation is much higher computation complexity than addition operation. The conventional convolutions are exactly cross-correlation to measure the similarity between input feature and convolution filters.

AdderNets takes the L1-norm distance between filters and input feature as the output response.
Ref: [https://arxiv.org/abs/1912.13200](https://arxiv.org/abs/1912.13200)

#### **DeepCache: Principle Cache for Mobile Deep Vision**

DeepCache benefits model execution efficiency by exploiting **temporal locality** in input video streams. At the *input of a model*, DeeepCache discovers video temporal locality by exploiting the video's internal structure, for which it borrows proven heuristics from video compression; into the model, DeepCache propagates regions of reusable results by exploiting the model's internal structure.

Ref: [https://arxiv.org/pdf/1712.01670.pdf](https://arxiv.org/pdf/1712.01670.pdf)

#### **LCNN: Lookup-based Convolutional Neural Network.**

To leverage **the correlation between the parameters** and **represent the space of parameters by a compact set of weight vectors**, called dictionary. This paper presents LCNN, a lookup-based convolutional neural network that encodes convolutions by few lookups to a dictionary that is trained to cover the space of weights in CNN.

Ref: [https://arxiv.org/abs/1611.06473](https://arxiv.org/abs/1611.06473)

#### **AdaBits: Neural Network Quantization with Adaptive Bit-widths.**

In this paper, the authors investigate a novel option to achieve this goal by enabling adaptive bit-widths of weights and activations in the model. 

Ref: [https://arxiv.org/abs/1912.09666](https://arxiv.org/abs/1912.09666)

#### **Slimmable neural networks.**
This paper introduces a deep neural network that provides different inference paths with respect to different widths for accuracy-efficiency trade-off at test time.

Ref: [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Universally_Slimmable_Networks_and_Improved_Training_Techniques_ICCV_2019_paper.pdf)

#### **Universally Slimmable Networks and Improved Training Techniques.**

Slimmable network is the type of NN that the width can be chosen from a predefined widths set to adaptively optimize accuracy-efficiency trade-offs at runtime. This paper proposes a systematic approach to train universally slimmable networks (US-Nets), extending slimmable networks to execute at arbitrary width, and generalizing to networks both with and without batch normalization layers.

Ref: [https://arxiv.org/abs/1903.05134](https://arxiv.org/abs/1903.05134)


#### **AQD: Towards Accurate Quantized Object Detection**

This paper introduces an Accurate Quantized object Detection solution, termed AQD, to fully get rid of floating-point computation. To this end, fixed-point operations are used in all kinds of layers, including the convolutional layers, normalization layers, and skip connections, allowing the inference to be executed using integer-only arithmetic.

Ref: [https://arxiv.org/abs/2007.06919](https://arxiv.org/abs/2007.06919)


#### **Defensive Quantization: When Efficiency Meets Robustness**
This paper presents an approach  for quantising neural networks such that the resulting quantised model is robust to adversarial and random perturbations.

The core idea of the paper is to enforce the Lipschitz constant of each linear layer of the network approximately close to 1. Since the Lipschitz constant of the neural network is bounded by the product of the Lipschitz constant of its linear layer (assuming Lipschitz 1 activation functions) the Lipschitz constant of the trained neural network is bounded by 1. 

Ref: [https://arxiv.org/abs/1904.08444](https://arxiv.org/abs/1904.08444)

<br/>

## **Deep Learning Training & Generalization Papers**

#### **Deep Mutual Learning**

In mutual learning, we start with a pool of untrained students who simultaneously learn to solve the task together. Specifically, each student is trained with two losses: a conventional supervised learning loss, and a mimicry loss that aligns each student’s class posterior with the class probabilities of other students.

![Introduction](http://i.imgur.com/hFC69aE.png)


#### **Self-training with Noisy Student improves ImageNet classification (Paper Explained)**

This paper presents `Noisy Student Training`, a semi-supervised learning approach that works well even when labeled data is abundant.

Noisy Student Training extends the idea of self-training and distillation with the use of equal-or-larger student models and noise added to the student during learning. On ImageNet, we first train an EfficientNet model on labeled images and use it as a teacher to generate pseudo labels for 300M unlabeled images. We then train a larger EfficientNet as a student model on the combination of labeled and pseudo labeled images. We iterate this process by putting back the student as the teacher. During the learning of the student, we inject noise such as dropout, stochastic depth, and data augmentation via RandAugment to the student so that the student generalizes better than the teacher.


*Additional Ablation Study Summarization*

- *Finding #1*: Using a large teacher model with better performance leads to better results.
- *Finding #2*: A large amount of unlabeled data is necessary for better performance.
- *Finding #3*: Soft pseudo labels work better than hard pseudo labels for out-of-domain data in certain cases
- *Finding #4*: A large student model is important to enable the student to learn a more powerful model.
- *Finding #5*: Data balancing is useful for small models.
- *Finding #6*: Joint training on labeled data and unlabeled data outperforms the pipeline that first pretrains with unlabeled data and then finetunes on labeled data.
- *Finding #7*: Using a large ratio between unlabeled batch size and labeled batch size enables models to train longer on unlabeled data to achieve a higher accuracy.
- *Finding #8*: Training the student from scratch is sometimes better than initializing the student with the teacher and the student initialized with the teacher still requires a large number of training epochs to perform well.

![Algorithms](http://i.imgur.com/BR5vER3.png)


#### **Revisiting Unreasonable Effectiveness of Data in Deep Learning Era.**

Since 2012, the size of the biggest datasets has remained constant. What will happen if we increase the dataset size by 10x or 100x ?. This paper takes a step towards to make clear the mystery surrounding the relationship between "enormous data" and visual deep learning.

The paper delivers some surprising findings:

- The performance on vision tasks increases logarithmically based on volume of training data size.
- Representation learning (pretraining) still holds a lot of promise. Once can improve performance on many vision tasks by just training a better base model.
- Present a SOTA results for different vision tasks.

<p align="center">
  <img src="http://i.imgur.com/RXl8gx2.png">
</p>


#### **Deep Learning of Binary Hash Codes for Fast Image Retrieval.**

Approximate nearest neighbor search NNS is an efficient strategy for large-scale image retrieval. This paper proposes an CNN-based method generate binary hash codes for fast image retrieval. When data labels available, **binary codes** can be learned by employing **a hidden layer** for representing the **latent concepts** that dominate the class labels.

![Image Retrieval](http://i.imgur.com/RotMrS0.png)

Ref: [CPVR 2015 Paper](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/papers/Lin_Deep_Learning_of_2015_CVPR_paper.pdf)


#### **Snapshot Ensembles: Train 1, Get M for Free**

A method to obtain ensembling multiple neural networks at no additional training cost. Instead of training M neural networks independently from scratch, we let SGD converge M times to local minima along its optimization path. Each time the model converges, we save the weights and add the corresponding network to our ensemble. We then restart the optimization with a large learning rate to escape the current local minimum. 

![Train 1, Get M for Free](http://i.imgur.com/X1XTmAx.png)

Ref: [https://arxiv.org/abs/1704.00109](https://arxiv.org/abs/1704.00109)



#### **Train longer, generalize better: closing the generalization gap in large batch training of neural networks**

Deep learning models are typically trained using stochastic gradient descent or one of its variants. These methods update the weights using their gradient, estimated from a small fraction of the training data. It has been observed that when using large batch sizes there is a persistent degradation in generalization performance - known as the "generalization gap" phenomenon. Identifying the origin of this gap and closing it had remained an open problem. 

Ref: [NIPS 2017](https://proceedings.neurips.cc/paper/2017/file/a5e0ff62be0b08456fc7f1e88812af3d-Paper.pdf)

#### **Averaging Weights Leads to Wider Optima and Better Generalization.**

This paper shows that simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, leads to better generalization than conventional training. Moreover, Stochastic Weight Averaging (SWA) procedure finds much flatter solutions than SGD, and opproximates the recent Fast Geometrics Ensembling (FGE) approach with a single model.

Ref: [https://arxiv.org/abs/1803.05407](https://arxiv.org/abs/1803.05407)