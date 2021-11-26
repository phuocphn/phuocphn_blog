---
layout: post
title:  "Shortest Path Distance Approximation"
date:   2021-11-23 14:00:00 +0900
categories: research
tags: path approximation distance
author: Phuoc. Pham
comment: true
---


![Shortest Path Distance Approximation](https://miro.medium.com/max/2000/1*x_WiMjF0s6_gGRqPmDZ1Eg.png)

This is a brief sumarization of the paper named "Shortest Path Distance Approximation using Deep Learning Techniques (Fatemeh Salehi Rizi)"


#### **Introduction**
Q: Why do we need to use deep learning to approximate distance between nodes when we have traditional exact methods like Dijkstra’s and A* algorithms?

A: Dijkstra’s and A* are very slow on very large graphs. Approximation technique can offer constant inference time of $$O(1)$$.


#### **Research Scope**

Q: What are things they have tried in the published paper?

A: Evaluation the use of `node2vec` and `Poincare` embedding with different binary operators.

They have demonstrated that neural networks can predict the shortest path distances effectively and efficiently.

#### **Proposed Method**
There are several ways to approximate the shortest path distances, but among them, a family of scalable algorithms so-called "landmark-based approach". In this category, a fixed set of landmark nodes is slected and actual shortest path distances are pre-computed from each landmark to all other nodes. With the knowledge to the landmarks, together with the triangle inequality, typically allows one to compute approximate the shortest path distance between any two nodes in $$O(l)$$ time, where $$l$$ is the number of landmarks.


The following is a brief summarization of the method proposed in the original paper (*which is based on the landmark-based approach*):

1. Collect your graph data.
2. Use Node2Vec algorithm to find node embeddings for each node.
3. Use a certain number of nodes in the graph as, what they call, **“landmarks”** and compute their distances from all the rest of the nodes. Now you have samples of form $$<(landmark_i, node_x), distance>$$.
4. For each sample found above fetch the corresponding node embeddings of the landmark and the node, and combine them with any of the suitable binary operators (average, element-wise multiplication etc.). So now you should have samples of form $$(embedding, distance)$$.
5. Now you have input-output pairs, so you do what you do best. Find a good neural net configuration and train the hell out of the model. But we will see later, as with AI in general, it’s not that easy.

The step-by-step implementation of the proposed method are covered by the following Medium article: [Shortest Path Distance Approximation Using Deep Learning: Node2Vec](https://towardsdatascience.com/shortest-path-distance-with-deep-learning-311e19d97569)


**My note**: In the paper, the authors use `Node2Vec` or `Poincare` for getting the node embedding, but it is also possible to use `GCN` for the same purpose.

#### **References**
1. [Shortest Path Distance Approximation Using Deep Learning Paper](https://arxiv.org/pdf/2002.05257.pdf)
2. [Medium Article](https://towardsdatascience.com/shortest-path-distance-with-deep-learning-311e19d97569)