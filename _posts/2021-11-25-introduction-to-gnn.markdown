<!-- ---
layout: post
title:  "Introduction to Graph Neural Network"
date:   2021-11-25 15:10:04 +0900
categories: research
tags: graph neural network
author: Phuoc. Pham
---  -->
**What is difference between transductive and inductive in GNN?**

The key difference between induction and transduction is that **induction** refers to learning a function that *can be applied to any novel inputs*, while **transduction** is only concerned with *transferring some property onto a specific set of test inputs*

Ref: https://www.researchgate.net/post/What-is-difference-between-transductive-and-inductive-in-GNN

**Inductive learning** is the same as what we commonly know as traditional supervised learning. We build and train a machine learning model based on a labelled training dataset we already have. Then we use this trained model to predict the labels of a testing dataset which we have never encountered before.

In contrast to inductive learning, **transductive learning** techniques have observed all the data beforehand, both the training and testing datasets. We learn from the already observed training dataset and then predict the labels of the testing dataset. Even though we do not know the labels of the testing datasets, we can make use of the patterns and additional information present in this data during the learning process.




Ref: https://towardsdatascience.com/inductive-vs-transductive-learning-e608e786f7d
