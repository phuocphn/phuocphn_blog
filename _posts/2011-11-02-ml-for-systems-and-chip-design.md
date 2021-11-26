---
layout: post
title:  "Machine Learning / Reinforcement Learning for System-on-Chip Design"
date:   2021-11-02 16:00:00 +0900
categories: research
tags: soc chip design rl reinforcement_learning
author: Phuoc. Pham
---
I have a hard time remembering what I read. This blog post is a summarization of what I have been working for the last 2 months about applying deep reinforcement leanring for RTL synthesis. It will be served for the purpose of future referencing.


In the past decade, systems and hardware have transformed ML. (Great systems enables the development of AI technologies). Now, it's time for ML to transform systems and hardware.  The AI boom is happening all over the world. We need significantly better systems and chips to keep up with the computational demands of AI. Look at the following picture, between 1959 to 2012, compute usage roughly doubled every two years
Since 2012, the amount of compute used in the largest AI training runs double every 3.4 months. By comparison, Moore's Law had an 18-month doubling period!

![AI and compute all](https://i.ibb.co/1zZzV0F/ai-and-compute-all.png)

Source: [AlexNet to AlphaGo Zero: A 300,000x Increase in Compute](https://openai.com/blog/ai-and-compute/)

But turns out these problems is very hard, Let's look the complexity of chip placement problem.

![Chip Placement Complexity](https://i.ibb.co/ZMzmdD0/chip-placement-complexity.png)

### **Combinatorial Optimization on Graph Data.**
Many problems in systems and chips are ***combinatorial optimization problems on graph data.***, for example:

* Compiler Optimization
	- Input: XLA/HLO graph
	- Objective: Scheduling/fusion of ops
* Chip Placement
	- Input: A chip netlist graph
	- Objective: Placement on 2D or ND girds
* Datacenter Resource Allocation
	- Input: A jobs workload graph
	- Objective: Placement on datacenter cells and racks

Combinatorial Optimization Problems or in this example, resource allocation scheduling on graph data is something that very interesting and appears over and over in systems and chip problems. And the approach the authors try to do is applying learning-based optimization.


 **Advantages of Learning Based Approaches**

ML models, unlike traditional approaches (such as branch and bound, hill climbing methods, or ILP solvers) can:
- Learn the underlying relationship between the context and target optimization metrics and leverage it to explore various optimization trade-offs
- "Gain experience" as they solve more instances of the problem and become experts over time.
- Scale on distributed platforms and train billions of parameters



### **Learning to Optimize Device Placement**

**What is Device Placement and Why Is it Important?**

![Device Placement Problem](https://i.ibb.co/48qQL1M/device-placement2.png)

Models become larger and larger, in order to run it effectively, it needs to distribute over the devices. The current trend towards many-device training, bigger models and larger batch sizes
Common example is taking the computation tensor ops to different GPUs.


**Posing Device Placement as an RL Problem.**

We train *a policy* that it will output assignments of ops to different available devices. We can run the model according to this assignments and use the runtime as feebacks to update the policy. So the policy next generates better assignments and the optimal runtime.
![Device Placement as an RL Problem](https://i.ibb.co/wWCYm5f/device-placement-as-rl.png)

<!-- **An End-to-End Hierarchical Placement Model.** -->

<!-- 
Learned Placement on NMT
Here you can see some of our results from using this policy to to place a neural machine translation with a pair of encoder decoder + attention architecture. Each of the colors  show a different GPU and  there are 4 different GPUs and white corresponds to a CPU placement.

The policy was able to clear out that the embedding  would benefit from being put on CPUs, and there are some other non-intuitive days of placing these are the LSTM cells on two different GPUs, and but it turns out that this model works better than the baseline, which place every layer of every layer of the encoder/decoder on a different GPU.

In order to understand why the place in my the policy was better, we will look into  the placement profiling and the ops.

Learned Placement on Inception-V3
Learn to place each embedding output of intermediate layers on different GPUs
 -->


**Generalized Polices for Device Placement**
The authors have been doing was train a policy from scatch for every new placement problem. But of course, the main goal is to go beyond this, which train a policy that can ***apply device placement to new unseen graphs at inference.***

Objective: Minimize expected runtime for predicted placement d across graphs.

$$J(\theta) = \mathop{\mathbb{E}}_{G\sim 	\mathbb{G}, D \sim \pi_{\theta} (G)}[r_G, D] \approx \frac{1}{N}\sum_G \mathop{\mathbb{E}}_{D \sim \pi_{\theta} (G)} [r_G, D] $$



**Generalized Device Placement Architecture**

To capture the graph structure and obtain generalization for unseen data, it requires new architectures and embeddings that transfer knowledge across graphs with different size and operation types. We need to use Graph Convolution Neural Network (GCN) for such purpose. And GraphSAGE is a typical (or representative) method for using GCN to capture the graph information as it is quite popular and powerful.

![Generalized Device Placement Architecture](https://i.ibb.co/qj828rp/generalize-device-placement-architecture.png)

<sup>Source: Yanqi Zhou, Sudip Roy, Amirali Abdolrashidi, Daniel Wong, Peter C. Ma, Qiumin Xu Ming Zhong, Hanxiao Liu, Anna Goldie, Azalia Mirhoseini, James Laudon, arxiv 2019 “GDP: generalized device placement for dataflow graphs<sup>




### **Learning to Partition Graphs**

**What is Graph Partitioning & Why We Need It ?**

The main objective of graph partitioning is to reduce complexity by breaking down problems into smaller subproblems, and we can solve subproblems easier. This technique has a lot of applications in many fields, such as: VLSI Design,  Device Placement, Distributed Social Network, Clustering, etc.

![Graph Partitioning](https://i.ibb.co/LtvpX8D/graph-partitioning.jpg)

The representive problem for graph partitioning: [The Normalized Cut Objective.](https://arxiv.org/abs/1910.07623)


### **Learning to Optimize Chip Placement**

**The (High) Motivation For Applying ML in Chip Design**

* Reducing the design cycle from 1.5 - 2 years to weeks
	- Today, we design chips for the NN architectures for 2-5 years from now
	- Shortening the chip design cycle would enable us to be far more adaptive to the rapidly advancing field of ML
* New possibilities emerge if we evolve NN architectures and chips together
	- Discovering the next generation of NN architectures (which would not be computationally feasible with today's chips)
* Enabling cheaper, faster, and more environmentally friendly chips


**Chip Placement Problem**

This is just another example of graph resource optimization. Basically, the logic of the given circuit is encoded or passed to us in the form of a **netlist** which is a graph of chip components (such as macros, which are memory components SRAM and standard cells which are logic gates.) are connected by wires. And the objective is place the chip components onto 2D chip canvas to minimize the latency of computation, power consumption, chip area and cost, while adhering to *constraints, such congestion, cell utilization, heat profile, etc*.

![Chip Placement Problem](https://i.ibb.co/mS8bpXB/netlist.png)


There are several prior approaches to chip placement: Partitioning-Based Methods (e.g. MinCut), Stochastic/Hill-Climbing Methods (e.g. Simulated Annealing) or Analytic Solvers (e.g. RePIAce)

The 4th category have proposed by researcher at Google, which is: **Learning-Based Methods.**


**Chip Placement with Reinforcement Learning**

The researchers have proposed a deep reinforcement learning approach to the chip placement problem, where an agent is trained to place the nodes of chip netlist one at a time onto the canvas. Once all of them have bene placed, we get a reward signal which we will use to update the parameters of our policy better and better at this task of placing chip netlist.

![Chip Placement with Reinforcement Learning](https://i.ibb.co/wzytBc1/chip-placement-as-rl.png)

* **State**: Graph embedding of chip netlist, embedding of the current node, and the canvas
* **Action**: Placing the current node onto a gird cell
* **Reward**: A weighted average of total wirelength, density, and congestion.

The following images are kind of a teaser of the final results. Sadly, the images are blurred due to condidentiality, but these are placements of actual TPU-v4 Block and the left image is the human placement. The white area are macros and the green area is composed of standard cell clusters. Overall, the proposed method finds smoother, rounder macro placements to reduce the wirelength.

![TPU-v4](https://i.ibb.co/yQTqh2H/tpu-v4.png)

You can see that the human expert took 6-8 weeks to generate this placement in conjuction with (or in the loop with) very expensive EDA tools. The proposed method only took 24 hours to generate the super-human placement that has lower wire lengths.

**What they are optimizing for (Objective Function)**

The objective function is basically to minimize the bellow cost function or to maximize the expected reward given the placement $$P$$ of a netlist $$G$$ over the average expected reward of all the graphs in our training set.

![Objective function](https://i.ibb.co/wdJB0Kc/objective-fn.png)

A hybrid approach to placement optimization is applied, the basic idea is that the RL policy is trained to play some macros of netlist one at a time and then *Force-Directed Method* (this method models this entire system as a set Springs, so it can pull connected nodes tightly towards each other). The reason for this is that the macro placement problem is much more complicated in a certain sense and it was kind of unsolved problem because of the fact that macros have non negligible area, therefore, standard cells are so small that you can basically model them as having no area, and so as many approaches are unlocked by this, so you can take various analytic approaches and almost prove that you have an optimal placement, whereas macros you cannot make these assumptions. Therefore, you want to tackle this hard part of placement and reduce the complexity of the RL problem  by focusing just on the macros.


### **Terminology**
*(just for reference purposes)*

**What is a macro ?**

Macros are intellectual properties that you can  use in your design. You do not need to design it. For example, memories, processor core, serdes, PLL etc.  A macro can be hard or soft macro. Soft macro and Hard macro are categorized as IP.

<p align="center">
  <img src="https://1.bp.blogspot.com/-v7L70PlQ1cc/XjhnpS7aUaI/AAAAAAAAiX0/cvIEuZyDWjg957TkDY1hodxkWWJroq6RgCLcBGAsYHQ/s640/placement2.PNG">
</p>



**Soft Macros**

Soft macros are used in SOC implementations. Soft macros are synthesizble RTL form, are more flexible than hard macros in terms of reconfigurability. Soft macros are not specific to any manufacturing process and have the disadvantage of being unpredictable in terms of timing, area, performance, or power.  Soft macros carry greater IP protection risks because RTL source code is more portable and therefore, less easily protected than either a netlist or physical layout data. Soft macros are editable and can contain standard cells, hard macros, or other soft macros.


**Hard Macros**

Hard macros are targeted for specific IC manufacturing technology. They are block level designs which are optimized for power or area or timing and silicon tested. While accomplishing physical design it is possible to only access pins of  hard macros unlike soft macros which allow us to manipulate the RTL.  Hard macro is a block that is generated in a methodology other than place and route and is imported into physical design database as a GDS2 file.


### **Knowledge Transfer Across Chips**

**Moving Towards Generalized Placements**

How to train policies that generalize across this general problem of placing netlist ?
At first, (*Google*) researchers focus on optimizing for each specific placement of a netlist onto a canvas and training a policy to a specific instance problem. But they felt that they should be able to move towards training policy that are able to solve this general problem of placing any netlist or any ASIC netlist or maybe even analog circuits or other types of related problems.

Other motivation is that, the designers would like to make some changes to the chip logic or add more memory or partition, and therefore it is necessary to be able to respond quickly, for example, able to leverage what we've learned on previous instances of the problem to give them a solution much faster. In the original formulation, the policy need to be trained with 10K of samples, and what we want to move towards a solution where we can even use zero shot learning (or at least with little fine-tunning) to reduce the training time.

![Moving Towards to Generalization](https://i.ibb.co/Km975xz/generalized-placements.png)


**First Attempts at Generalization Have Done by Google Researchers** 

- Using the previous RL policy architecture, we trained it on multiple chips and tested it on new unseen chips *-> Didn't work !*
- Freezing different layers of the RL policy and then testing it on new unseen chips *-> Didn't work either !*

What did work? ***Leveraging supervised learning to find the right architecture !***

**Achieving Generalization by training accurate reward predictors.**
We observed that the main reason is that ***the value network is not generalize***, a value network trained only on placements generated by a single policy is unable to accurately predict the quality of placements generated by another policy, limiting the ability of the policy network to generalize.

The Google researchers decided to decompose the problem, and trained models capable of accurately predicting reward from off-policy data. It based on the belief that if the network is unable to predict reward across variety of placements, then it would be unable to solve the general problem of placing chip.


### **Train A Supervised Learning Model**

**Compiling a Dataset of Chip Placements:**

To train a more accurate predictor, a new dataset of 10K placements is generated. Each placement is labeled with their wirelength and congestion, which is drawn from the vanilla RL policies at different stage of maturity or different stages of the training process. This data is valuable because it gives us a variety of quality of placements. 

![Generated Placements](https://i.ibb.co/KmvQkPm/generated-data.png)

**Reward Model Architecture and Features.**

The dataset after compiling is then used to find the right architecture that would be able to perform the task of predicting the quality of these placements.

![Reward Model Architecture and Features](https://i.ibb.co/3s9bK4m/reward-model-architecture.png)

**Note:** The netlist metadata such as: #wires and macros, name of netlist are also provided, which maybe gives the model some idea of the functionality or the goal of this chip block.


What Google researchers found is that ***some graph neural network architectures are much more focused on node features***. Whereas in the chip placement problem it's actually more a function of the edges, ( wire lengths timing it's more related to the critical paths within this network it's not really about the node features themselves)  and so we took a kind of *edge based approach*.


#### **How we are generating representations of this graph and of the nodes ?**

In this sub-section, I will show you how to create a representation of the graph, and use it as the input feature for downstream tasks.


**Edge-based Graph Convolution #1: Node Embeddings:**
![Edge-based Graph Convolution: Node Embeddings](https://i.ibb.co/zWMC5XR/node-embedding.png)
Each node is is represented as a tupe of $$(X, Y, W, H)$$. And we feed node feature into a fully connected work that produces an embedding

**Edge-based Graph Convolution #2: Edge Embedding:**

We then concatenate these neighbor embeddings. Basically, we generate a some type of representation, and then we pass that into another fully connected layer to generate an edge embedding.

<p align="center">
  <img src="https://i.ibb.co/6Whxbfy/edge-embedding-step1.png">
</p>


**Edge-based Graph Convolution #3: Propagate**: Once again, we will represent the features of the nodes as *averages of their edge embeddings*. 

![Edge-based Graph Convolution: Propagate](https://i.ibb.co/DRs2bz4/propage.png)

**Edge-based Graph Convolution #4: Repeat**:  We iterate multiple times until we converge. 

![Edge-based Graph Convolution: Repeat](https://i.ibb.co/PNwM1DG/repeat.png)

**Edge-based Graph Convolution #5: Final Step**:  At the very end we can get a representation of the entire graph by just taking the mean of all of the edge embeddings.

![Edge-based Graph Convolution: Final Step](https://i.ibb.co/5kFcrqL/final-step.png)


#### **Label Prediction Results on Test Chips**
The (Google) researchers found that they were actually able to do quite well at this task of predicting the rewards for placements (which are generated from a variety of different policies). The prediction results are good for the `wirelength` metric, although there are little noises for the `congestion` metric. But away, you can see there is a clear correlation between the ground truth and predictions.

![Label Prediction Results on Test Chips](https://i.ibb.co/4dZgNsp/supervised-learning-results.png)


#### **Policy/Value Model Architecture.**
The architecture (found by using the supervised learning) is used as **the encoder** for the policy-value model architecture. And the policy net will generate the prediction about locations, where the next nodes should be placed.

![Policy/Value Model Architecture](https://makinarocks.github.io/assets/images/2021-02-15-chip_placement_with_reinforcement_learning/chip_placement_with_deep_reinforcement_learning_model.png)

It is worth to mention that the mask component in the above picture is also play the important role, as it will enforce the hard constraint on density of the placement. This information is valuable because not only does it help us you know obviously to prevent generating invaluable placements but it also helps us to reduce the search space for faster learning.

Other sections/ information are beyong the scope of this note. For the experiment results and detail analysis, I refer you to check this blog post: [Chip Design with Deep Reinforcement Learning](https://ai.googleblog.com/2020/04/chip-design-with-deep-reinforcement.html)


### **References**

1. [Lecture by Azalia Mirhoseini & Anna Goldie (CS 159 Spring 2020)](https://www.youtube.com/watch?v=lBzh9WY5hpU&t=2418s)
2. [Chip Design with Deep Reinforcement Learning](http://ai.googleblog.com/2020/04/chip-design-with-deep-reinforcement.html)
3. [What is macro?](http://88physicaldesign.blogspot.com/2015/10/what-is-macro.html)
