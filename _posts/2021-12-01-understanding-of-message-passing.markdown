---
layout: post
title:  "Understanding of Message Passing in Pytorch Geometric."
date:   2021-12-01 14:00:00 +0900
categories: research
tags: graph convolution
author: Phuoc. Pham
comments: true
---



Last week, I was asked to implement a new GCN layer to solve a new task in my research project. After spending days to figure out about the idea of `MessagePassing` in Pytorch Geometric library, finally, the new model is now working as expected. While waiting for the training process to finish, I decide to write this article to share with you what I have learned so far about Pytorch Geometric , MessagePassing and how to implement the new GCN layer, which is inherited from the `MessagePassing` base class. I hope this article will be helpful for the future readers.

If you have any questions, feel free to leave a comment below. I will try to try to answer with my best knowledge and as soon as possible.

The contents of this article is described as follows:

- Background about Message Passing
- The meaning of each function in the `MessagePassing` base class.
- An example of implementing GraphSAGE with the `MessagePassing` base class.




#### **What Is Message Passing ?**

In Graph Neural Network, the process of collecting/exchanging the characteristics of the node's neighborhood to improve the current node representation is called Message Passing. The main goal of Message Passing is to encode contextual graph information in node embeddings by iteratively combining neighboring nodes' features.

The Message Passing scheme can be defined in the mathematical terms as follows:

$$\mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right),$$

where $$\square$$ denotes a differentiable, permutation invariant function, e.g., sum, mean or max, and $$\gamma$$ and $$\phi$$ denote differentiable functions such as MLPs (Multi Layer Perceptrons).


#### **`MessagePassing` Class In Pytorch Geometric**

In PyTorch Geometric, you can implement Message Passing through the base class called `torch_geometric.nn.MessagePassing`. It helps in creating such kinds of message passing graph neural networks by automatically taking care of message propagation. The user only has to define the functions  $$\phi$$ , i.e.  message(), and $$\gamma$$ , i.e. update(), as well as the aggregation function $$\square$$


##### **MessagePassing - `__init__` function**

`torch_geometric.nn.MessagePassing(aggr="add", flow="source_to_target",  node_dim=-2)`

- `aggr`: Decide how to aggregate messages between each node. Default value is `add`. Possible values include: `add`, `mean`, `max`, `None`
- `flow` : Decide in which direction to flow the message (whether to receive or forward it from the neighboring node). Default value is “source_to_target”. Possible values:  `source_to_target`, `target_to_source`
- `node_dim`: indicates along which axis to propagate.


##### **MessagePassing - `propagate()` function**

`propagate(edge_index, size=None, **kwargs)`

By calling  `propagate()`, which internally calls the `message()` $$\to$$ `aggregate()` $$\to$$ `update()` functions. 
This is the skeleton of the `propagate()` function.

```python
def propagate():
	if mp_type == 'adj_t' and self.fuse and not self.__explain__:
		out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

	# Otherwise, run both functions in separation.
	elif mp_type == 'edge_index' or not self.fuse or self.__explain__:
		msg_kwargs = self.__distribute__(self.inspector.params['message'],coll_dict)
		out = self.message(**msg_kwargs)
		out = self.aggregate(out, **aggr_kwargs)

	out = self.update(out, **update_kwargs)
```
- `message` and `aggregate` functions are used separately or combined via `message_and_aggregate()` function. If applicable, using this function will save both time and memory since messages do not explicitly need to be materialized.


##### **MessagePassing - `message()` function**
```python
def message(self, x_j: torch.Tensor) -> torch.Tensor:
    # need to be implemented
    return x_j
```
- You specify how you construct “message” ( that occurs at each edge) for each of the node pair (`x_i`, `x_j`). Since it follows the calls of propagate, it can take any argument passing to propagate. One thing to note is that you can define the mapping from arguments to the specific nodes with "\_i" and "\_j". Therefore, you must be very careful when naming the argument of this function.

Node features get automatically mapped to source (`_j`) and target (`_i`) nodes

- **Source nodes** (denoted by `_j`): surrounding (neighborhood) nodes that we want to do an aggregation over their's embeddings.
- **Target nodes** (denoted by `_i`): the current node that we want to update.

![Different between source and target](//i.imgur.com/UBkC71k.png)


##### **MessagePassing - `forward()` function**

If there is any transformation needs to be applied to the <ins>current node features</ins>, then it should be implemented in this function. For example, passing the node features to a FC layer to get a another embedding (as illustrated in the image below) should be done in 

```python
def forward(self, x, edge_index):
    # x has shape [num_nodes, in_channels]
    # edge_index has shape [2, num_edges]

    return self.propagate(edge_index, x=x)
```

##### **MessagePassing - `update()` function**
Updates node embeddings for each node in the graph. It takes in the output of aggregation as first argument and any argument which was initially passed to `propagate()`.

```python
def update(self, inputs: torch.Tensor):
    # need to be implemented
    return inputs
```

#### **An Overview**
The following images give the an overview about where to implement different part of your custom message passing logic.

- Everything related to the node transformation should be implemented in `forward()`. Expected output shape: $$[\text{# vertices, #features}]$$

![forward](https://i.imgur.com/2gWlacE.png)
- Whenever you need two or more nodes to do a transformation, you should implement your custom logic in `message()`. Expected input shape: $$x_i, x_j: [\text{# edges, #features}]$$. Expected output shape is also: $$[\text{# edges, #features}]$$
<p align="center"><img src="https://i.imgur.com/2mP3Xpb.png"></p>

- The logic of how ot update the current node embedding with the calculated embedding should be implemented in `update()` function.  Expected input shape: $$[\text{1, #features}]$$ (this shape is obtained after passing through the  `aggregate()` function). Expected output shape: $$[\text{# vertices, #features}]$$.
![update](https://i.imgur.com/M7AVXgG.png)






### **Re-Implement SAGEConv layer with pytorch_geometric**

GraphSAGE is a convolutional graph neural network algorithm. The key idea behind the algorithm is that we learn a function that generates node embeddings by sampling (to reduce the computation overload) and aggregating feature information from a node’s local neighborhood.

![GraphSAGE](https://miro.medium.com/max/2000/1*t3ODGTJC5bcRFDKVaih3pA.png)

In the above image, `K` corresponds to the number of hops, which is equivalent to the number of SAGEConv layers. If you have two SAGEConv layers, then `K=2`. [[ref]](https://github.com/pyg-team/pytorch_geometric/issues/1398#issuecomment-653756482)


The main GraphSAGE algorithm is descripted as follows: 


![Algorithms](http://i.imgur.com/zTkMXqi.png)

The most important lines is 4 and 5, where the message passing formula is explicitly defined as:  (with the aggregation function is $$\mathbf{max}$$)


$$h^k_{\mathcal{N}(v)} \leftarrow max(\{ \sigma(\symbf{W}_{pool} \symbf{h}^k_{u_i} + \symbf{b}), \forall u_i \in \mathcal{N}(v) \})  $$ 

$$h^k_{\mathcal{N}} \leftarrow \sigma (\symbf{W}^k  \cdot \textup{CONCAT} (\symbf{h}^{k-1}_v, \symbf{h}^k_{\mathcal{N}(v)}))$$ 


Let's take a look about how to implement this algorithm with pytorch_geometric. I want to show two different approaches that I have found,  and after that I will list some notes about the differences between them. 


#### **Kung-Hsiang, Huang (Steeve) Implementation**
The following code is the completed SAGEConv implementation, which is taken from the great article: [Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric](https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8). Compared to the built-in GraphSAGE implementation of pytorch_geometric (which will be introduced in the second part), this implementation is strictly followed the procedure described in the original paper. But in terms of performance, the built-in GraphSAGE implementation of pytorch_geometric is much more efficient.


```python
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, 
        	in_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        return new_embedding
```

###### **Little Notes/Comments**: 
In the first equation, each neighboring node embedding is passed through linear layer, and then followed by a nonlinear ReLU function. Remember that because we are only working with the **source nodes**, therefore the `message` function contains only 1 single argument: `x_j`

```python
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        
    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        return x_j
```

After the `message` function is executed, the `aggregate()` with the default value `max` will be called and return aggregated value:  $$\textup{aggr_out} = h^k_{\mathcal{N}(v)}$$.




#### **Buit-in Pytorch-Geometric SAGEConv Implementation**

And the code below is another implementation of GraphSAGE from pytorch_geometric library. Note that the message message passing formula is little differences with the previous one. The aggregation function is $$\mathbf{mean}$$, and the summarization is used instead of concatenation [[ref]](https://github.com/pyg-team/pytorch_geometric/issues/945#issuecomment-582928411). The authors of pytorch_geometric library mentioned that summation/concatenation is the same, but summation is better computation-wise though since it does not require to put node features into a new memory layout. [[ref #1]](https://github.com/pyg-team/pytorch_geometric/issues/1252#issuecomment-633810675), [[ref #2]](https://github.com/pyg-team/pytorch_geometric/issues/945)

$$\mathbf{x}^{\prime}_{i} =\mathbf{W}_{1} \mathbf{x}_i + \mathbf{W_2} \cdot\mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j$$


```python
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
```

Yes, there is a difference between message passing formula as result in the implementation. And also, you may wonder that there is no node sampling mechanism available in both implementations so far. The reseason is this kind of sampling technique is (usually) implemented outside the `SAGEConv` layer. If you use `SAGEConv` from pytorch_geometric, you have to use NeighborSampler to do node sampling. You have read two examples to understand more about how to use it.
- [Sampling Large Graphs in PyTorch Geometric, Mike Chaykowsky](https://towardsdatascience.com/sampling-large-graphs-in-pytorch-geometric-97a6119c41f9)
- [Reddit Example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py)


It is also worth to mention that ff we do not use NeighborSampler when passing data to `SAGEConv` Layers then the `SAGEConv` will not sample subset of neighbors itself and use all the neighbors using edge indices.[[ref]](https://github.com/pyg-team/pytorch_geometric/issues/1649#issue-703930416)





#### **References**
1. [A Comprehensive Case-Study of GraphSage using PyTorchGeometric and Open-Graph-Benchmark](https://www.arangodb.com/2021/08/a-comprehensive-case-study-of-graphsage-using-pytorchgeometric/)
2. [Pytorch Geometric Message Passing 설명](https://greeksharifa.github.io/pytorch/2021/09/04/MP/)
3. [CREATING MESSAGE PASSING NETWORKS](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)
4. [Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric](https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8)
5. [PyTorch Geometric 탐구 일기 - Message Passing Scheme (1)](https://baeseongsu.github.io/posts/pytorch-geometric-message-passing1/)
6. [ADL4CV - Graph Neural Networks and Attention](https://www.youtube.com/watch?v=FbkE7FsHDkc)
