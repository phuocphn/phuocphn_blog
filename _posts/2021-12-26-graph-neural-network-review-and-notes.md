---
layout: post
title:  "Graph Neural Networks: Comprehensive Review & Personal Notes"
date:   2021-12-01 14:00:00 +0900
categories: ml
tags: graph neural network
author: Phuoc. Pham
comments: true
---

*This article is still in the process of writing. Last updated: Jan 05, 2022*


We are usually familiar with images, structured data and text when working with ML/AI. The image has a 2-D grid format, the strctured data usually takes the form of a table with rows and columns. Similarly, text can be represented as a 1-D sequence. In other words, these data can be represented in a particular coordinate system, meaning that they are in Euclidean space.

<p align="center">
  <img src="https://i.imgur.com/0bBI5DP.png">
</p>

However, social network data, molecular data, annd 3D mesh image data don’t exist in a Euclidean space, which means it can’t be represented by any coordinate systems with which we’re familiar. Therefore, we need a new model that can process this type of data differently from the existing CNN and RNN models. Such a model is called: *Graph Neural Network*. 

<p align="center" >
  <img src="https://imgur.com/wsEg0pl.png" width="400" height="400">
</p>


In this post, I will cover background and fudanmental concepts about Graph Neural Network, along with my understanding and insights so far with this topic. 
This post aims to serve as a future reference, provide a comprehensive information and adress popular questions related to Graph Neural Network. 




knowing how spectral convolution works in still helpful to understand and avoid potential problems with other methods.

### **What is a Graph ?** 
A graph $$G=(V, E)$$ is a set of nodes (vertices) $$\textbf{V}$$, connected by directed/undirected edges $$\textbf{E}$$ .Nodes generally contain data information, and edges contain relationship information between data.

![GCN](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkRrEC%2FbtqCB7CyV6S%2FuCtmLrJVKGokjWXDEtIUNK%2Fimg.png)

<center><i>Graph Representation by Matrices,  Source: [https://littlefoxdiary.tistory.com/17]</i></center>
<br/>

In order to work with graphs, these graphs must be represented in the form of matrices. Let's assume an undirected graph $$G$$ with $$N$$ nodes. Each node in this graph has a $$C$$-dimensional feature vector, and features of all nodes are represented as an $$N \times C$$ dimensional matrix $$\textbf{X}$$. For expressing connectivity information, edges of a graph are represented as an $$N \times N$$ matrix $$\textbf{A}$$, you can use adjacency matrix, degree matrix or laplacian matrix as described below. Among that, adjacency matrix is used most often.

<p align="center" >
  <img src="https://koenig-media.raywenderlich.com/uploads/2017/01/graph6.png">
</p>


**Adjacency Matrix**

The adjacency matrix is an N-by-N square matrix if the number of graph nodes is N. If $$i-$$node and $$j-$$node are connected then $$A_{ij} = 1$$, otherwise $$A_{ij} =0$$.


**Degree Matrix**

The adjacency matrix is an N-by-N square matrix if the number of graph nodes is N. The matrix contains information about the degree of each vertex, which is the number of edges connected to.


**Degree Matrix**

The laplacian matrix contains information about itself and neighboring nodes connected to it. It is defined as the subtraction of degree matrix and adjacency matrix. 

![EdgeConnectivity](https://imgur.com/bYiaa4S.png)


### **Graph Convolution Network.** 

GNNs (ConvGNNs) aim to mimic the simple and efficient solution provided by CNN to extract features through a weight-sharing strategy along the presented data. In images, a convolution relies on the
computation of a weighted sum of neighbor’s features and weight-sharing is possible thanks to the neighbor relative positions. With graph-structured data, designing such a convolution process is not straightforward. 
First, there is a variable and unbounded number of neighbors, avoiding the use of a fixed sized window to compute the convolution.
Second, no order exists on node neighborhood. As a consequence, one may first redefine the convolution operator to design a ConvGNN.





### **Graph Convolution in the Spatial vs Spectral Domain**

Graph Convolution Networks (GCN) draw on the idea of Convolution Neural networks re-defining them for the graph domain.  For images, convolution aims to capture the neighbourhood pixel information, while for graph, it aims to capture the surrounding node information.

But when it comes to GCNs, the idea of convolution can be classified into two categories: Spectral GCNs and Spatial GCNs.


#### **<ins>Spectral Methods</ins>**

**TL;DR:**: REQUIREs the use of eigen-stuff


**P1**: Spectral-based approaches define graph convolutions by introducing filters from the perspective of graph signal processing (GSP) which is based on graph spectral theory. It involves about Fourier transformation and Laplacian domain. 

>> GSP is the key to generalizing convolutions, allowing us to build functions that can take into account both the overall structure of the graph and the individual properties of the graph’s components.

**P2**: In a nutshell, a basis is defined by the eigendecomposition of the graph Laplacian matrix. This allows to define the graph Fourier transform, and thus the graph filtering operators. The original form of the spectral graph convolution (non-parametric) can be defined by a function of frequency (eigenvalues). Therefore, this method can theoretically extract information on any frequency. 

**P3**: In a spectral graph convolution, we perform an Eigen decomposition of the Laplacian Matrix of the graph. This Eigen decomposition helps us in understanding the underlying structure of the graph with which we can identify clusters/sub-groups of this graph. This is done in the Fourier space. [7]

**P4**: A general form of Spectral-based methods is defined as follows: [6]

$$H_j^{(l+1)} = \sigma (\sum_{i=1}^{f_i} U \text{diag} (F_{i,j,l}) U^T H_i^{(l)}) )$$

where $$U$$ is the eigenvectors, $$H_i^{(l)}$$ is the node features and $$F_{i,j,l}$$ is the trainable filter coeff.


Despite that spectral graph convolution is currently less commonly used compared to spatial graph convolution methods, knowing how spectral convolution works is still helpful to understand and avoid potential problems with other methods. 



**Disadvantages of using Spectral-based methods**

Despite the solid mathematical foundations + GSP literature, it suffers from following disadvantages: 

**D1**: It requires the entire graph to be processed simultaneously, which can be impractical for large graphs with billions of nodes and edges such as the social network graphs. Also it is difficult to take advantage of the parallel processing power.

**D2**: The filters are learned in the context of the spectrum of Graph Laplacian, therefore it need to assume a fixed graph and only the signal defined on the vertices may differ $$\to$$ generalizing poorly to new or different graphs ( not suitable for tasks where graph structure is changing from sample to sample  such as meshes, point clouds, or diverse biochemical datasets.. [3]) 

**D3**: It suffers from a large computational burden induced by the forward/inverse graph Fourier transform.


#### **<ins>Spatial Methods</ins>**

**TL;DR:** DON'T REQUIRE the use of eigen-stuff.

**P1** Spatial-based methods formulate graph convolutions as aggregating feature information from neighbours. The key to the spatial graph convolution method is to select a neighbor node with a fixed size and maintain "local invariant". Spatial graph convolution receives information only from fixed neighboring nodes and updates the node information.

The key of spatial-based methods is to learn a function $$f$$ for generating a node $$v_i$$'s representation by aggregating its own features $$\textbf{X}_i$$ and neighbours' features $$\textbf{X}_j$$. The aggregation function must be permulatation invariant of node orderings. (e.g: $$max, min, sum ...$$).  A non-linear transformation is applied after doing feature aggregation.

To explore the depth and breadth of a node's field of influence, a common way is to stack multiple graph convolution layers together. The first layer ò GCN facilitates information flow between first-order neighbours; the second layer gets information fron the second-order neighbours e.g: from the neighbour's neighbours. By continuing this way, the final hidden representation of each node receives messages from a further neighbourhood.

**P2**: Spatial-based methods consist of two functions: $$\text{AGGEGATOR}$$ collects the information from neighborhood nodes and summarizes them, $$\text{UPDATE}$$ fuses the concern node information with aggerator and find a new representation of the node.

$$H_{:v}^{(l+1)} = \text{UPDATE} ( f(H_{:v}^{(l)}), \text{AGGERATOR} ( g(H_{:u}^{(l)})) : u \in \mathcal{N} (v) ) $$

where: $$H_{:v}^{(l+1)}$$ is the new representation of the node, $$\mathcal{N} (v) $$ is a set of neighboring nodes of $$v$$.

In general, we can implement **all Spatial GNN** by a general framework: 

$${H}^{(l+1)} = \sigma (\sum_s C^{(s)} H^{(l)} W^{(l,s)})$$

where $$H$$ is the node features, $$C$$ is convolution support, and $$W$$ is trainable parameters.


**A1**: Spatial-based method do not require entire graph to work with, as they directly perform convolution in the graph domain by aggregating the neighbour node's information. Together with sampling strategies, the ocmputation can be performed in a batch of nodes instead of the whole graph, which has the potential to improve efficiency.

**A2**: It performs graph convolution locally on each node, thus it is easily share weights across different locations and structures. This is the reason why spatial-based models have attracted increasing attention in recent years.


#### **<ins>Hybird Approaches</ins>**

There are some papers that try to avoid using 
Kipf [CoRR](https://arxiv.org/abs/1609.02907) approximate the **spectral method** of  [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) and weaken the dependency on the Lapclacian, but ultimately arrive at center-surround weighting of neighborhoods.


### **Weisfeiler-Lehman (WL) Graph Isomorphism Test**

*(in progress)*

Similar to GNNs, the WL test iteratively updates a given node’s feature vector by aggregating feature vectors of its network neighbors. What makes the WL test so powerful is its injective aggregation update that maps
different node neighborhoods to different feature vectors. Our key insight is that a GNN can have as large discriminative power as the WL test if the GNN’s aggregation scheme is highly expressive and can model injective functions. 


#### **Graphs Isomorphism**

- **D1**: Same graph (different layouts) + Same graph (different labels). All that matters are the connections: Graphs with the same connections (# vetices, #edges) are ismorphic. [9]
- **D2**: Two graphs are isomorphic when there is an <ins>edge-preserving matching</ins> of their vertices.
- **D3**: (Formal definition) / $$\textbf{G}_1$$ is isomorphic to $$\textbf{G}_2$$  means edge-preserving vertex matching: 

$$\exists$$ bijection $$\text{f}: \textbf{V}_1 \to \textbf{V}_2 $$ with $$u - v $$ in $$\textbf{E}_1$$ IFF $$\text{f} (u) - \text{f} (v)$$ in $$\textbf{E}_2$$



<p align="center" >
  <img src="https://harryjo97.github.io/assets/post/Weisfeiler-Lehman-Algorithm/algorithm.PNG">
</p>

 The graph isomorphism problem asks whether two graphs are topologically identical. This is a challenging problem: no polynomial-time algorithm is known for it yet. The Weisfeiler-Lehman (WL) test of graph isomorphism (Weisfeiler & Lehman, 1968) is an effective and computationally efficient test that distinguishes a broad class of graphs. Its 1-dimensional form, “naïve vertex refinement”, is analogous to neighbor aggregation in GNNs. The
WL test iteratively (1) aggregates the labels of nodes and their neighborhoods, and (2) hashes the  aggregated labels into unique new labels. The algorithm decides that two graphs are non-isomorphic if at some iteration the labels of the nodes between the two graphs differ. [10]

<!-- Based on the WL test, Shervashidze et al. (2011) proposed the WL subtree kernel that measures the similarity between graphs. The kernel uses the counts of node labels at different iterations of the WL test as the feature vector of a graph. Intuitively, a node’s label at the k-th iteration of WL test represents a subtree structure of height k rooted at the node (Figure 1). Thus, the graph features considered by the WL subtree kernel are essentially counts of different rooted subtrees in the graph
 -->


#### **Injective Function**

A function $$f$$: $$\textbf{X} \to \textbf{Y}$$ is not injective if two distinct elements $$a,b \in \textbf{X} $$ exist with $$f(a)=f(b)$$. If that is not the case then the function is injective. If that is not the case then the function is injective. [11]

<p align="center" >
  <img src="https://i.stack.imgur.com/mg8MW.jpg">
</p>


### **Representive Papers.**

*(in progress)*

#### **Classic Papers**

- Graph Convolutional Networks (GCNs) — Kipf and Welling:  Among the most cited works in graph learning is a paper by Kipf and Welling. The paper introduced **spectral convolutions** to graph learning, and was dubbed simply as “graph convolutional networks”, which is a bit misleading since it is classified as a spectral method and is by no means the origin of all subsequent works in graph learning.


#### **Generalization & Feature Representation**

- [Generalization and Representational Limits of Graph Neural Networks](https://proceedings.mlr.press/v119/garg20c.html)
- [Towards Scale-Invariant Graph-related Problem Solving by Iterative Homogeneous Graph Neural Networks](https://arxiv.org/pdf/2010.13547.pdf)
- [On Size Generalization in Graph Neural Networks ](https://openreview.net/pdf?id=9p2CltauWEY)
- [Simple scalable graph neural networks](https://towardsdatascience.com/simple-scalable-graph-neural-networks-7eb04f366d07)

<br/>




---

<br/>


#### **References**
1. [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/pdf/1704.02901.pdf)
2. [Graph Convolutional Network에 대하여 - Spectral Graph Convolution](https://ralasun.github.io/deep%20learning/2021/02/15/gcn/)
3. [Enhance Information Propagation for Graph Neural Network by Heterogeneous Aggregations](https://arxiv.org/abs/2102.04064)
4. [How to Use Graph Neural Network (GNN) to Analyze Data](https://builtin.com/data-science/gnn)
5. [Beyond Graph Convolution Networks - Aishwarya Jadhav](https://towardsdatascience.com/beyond-graph-convolution-networks-8f22c403955a)
<!-- https://medium.com/swlh/data-structures-graphs-50a8a032db03There is a ton of information out there on the history of red-black trees, as well as the strategies for handling rotation and re coloring. However, it can be hard to distill which resources are the most useful when you’re first starting out. Here are some of my favorites, including a great lecture from one of the creators of red-black trees himself! Happy node painting! -->
6. [Bridging the Gap Between Spectral and Spatial Domains in Graph Neural Networks](https://arxiv.org/pdf/2003.11702.pdf)
7. [What is the difference between graph convolution in the spatial vs spectral domain?](https://ai.stackexchange.com/questions/14003/what-is-the-difference-between-graph-convolution-in-the-spatial-vs-spectral-doma)
8. [Graph Convolutional Networks for Geometric Deep Learning](https://towardsdatascience.com/graph-convolutional-networks-for-geometric-deep-learning-1faf17dee008) - *a brief description about popular GCN papers*
9. [2.8.3 Isomorphism: Video - MIT OpenCourseWare](https://www.youtube.com/watch?v=hVerxuP4cFg&ab_channel=MITOpenCourseWare)
10. [How Powerful are Graph Neural Networks? - arXiv](https://openreview.net/pdf?id=ryGs6iA5Km)
11. [What is the exact definition of an Injective Function](https://math.stackexchange.com/questions/1281906/what-is-the-exact-definition-of-an-injective-function)
12. [Expressive power of graph neural networks and the Weisfeiler-Lehman test](https://towardsdatascience.com/expressive-power-of-graph-neural-networks-and-the-weisefeiler-lehman-test-b883db3c7c49)
13. [Beyond Weisfeiler-Lehman: using substructures for provably expressive graph neural networks](https://towardsdatascience.com/beyond-weisfeiler-lehman-using-substructures-for-provably-expressive-graph-neural-networks-d476ad665fa3)

#### **Related Resources:**
1. [Gitta Kutyniok - Spectral Graph Convolutional Neural Networks Do Generalize](https://www.youtube.com/watch?v=Mo1A5AjzfC4)
2. [Week 13 – Lecture: Graph Convolutional Networks (GCNs)](https://www.youtube.com/watch?v=Iiv9R6BjxHM)
3. [When Spectral Domain Meets Spatial Domain in Graph Neural Networks](https://www.youtube.com/watch?v=8tLJ2beCv5w)
4. [Spectral Graph Convolution Explained and Implemented Step By Step](https://towardsdatascience.com/spectral-graph-convolution-explained-and-implemented-step-by-step-2e495b57f801)
5. [Anisotropic, Dynamic, Spectral and Multiscale Filters Defined on Graphs](https://towardsdatascience.com/tutorial-on-graph-neural-networks-for-computer-vision-and-beyond-part-2-be6d71d70f49)
