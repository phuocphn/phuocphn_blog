---
layout: post
title:  "Understanding the limitations of K-means algorithm."
date:   2021-11-08 12:24:00 +0900
categories: ml
tags: k-means
author: Phuoc. Pham
comment: true
---

Does K-means requires any assumptions before applying it the your dataset ?

I recently came across this question on [Stack Exchange](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means/133694#133694), and I thought it offered a great opportunity to explore deeply about this, about the assumptions underlying the K-means algorithm and what would happen if one of these assumptions is broken.


When people are first exposed to machine learning K-means clustering is one of the techniques that creates immediate excitement. However, the effectiveness of K-means rests on a number of (usually implicit) assumptions about your dataset. These assumptions match our intuition about what a cluster is—which makes them all the more dangerous. There are traps for the unwar [[ref]](https://blog.learningtree.com/assumptions-ruin-k-means-clusters/)
 
This is the assumptions you need to check before applying K-means in your problem.

* K-means assumes the variance of the distribution of each attribute (variable) is spherical;
* All variables have the same variance;
* The prior probability for all $$K$$ clusters is the same, i.e., each cluster has roughly equal number of observations;


Understanding the assumptions underlying a method is essential: it doesn't just tell you when a method has drawbacks, it tells you how to fix them. [[ref]](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means/133694#133694)


Let's go to each assumption. And We'll stick to 2-dimensional data since it's easy to visualize.

### **Be able to optimize $$\ne$$ we're accomplishing the goal !.**


**Diversion: Anscombe's Quartet**

Anscombe's quartet comprises four data sets that have nearly identical simple descriptive statistics, yet have very different distributions and appear very different when graphed. Each dataset consists of eleven (x,y) points. They were constructed in 1973 by the statistician Francis Anscombe to demonstrate both the importance of graphing data before analyzing it, and the effect of outliers and other influential observations on statistical properties.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/425px-Anscombe%27s_quartet_3.svg.png">
</p>


<sup> *(left-to-right, top-to-bottom) (1), is appropriate for linear regression. (2) it suggests the wrong shape, in (3) it is skewed by a single outlier- and in (4) there is clearly no trend at all!*</sup>

One could say "Linear regression is still working in those cases, because it's minimizing the sum of squares of the residuals. ". Linear regression will always draw a line, **but if it's a meaningless line, who cares?**

*So now we see that just because an optimization can be performed doesn't mean we're accomplishing our goal. And we see that making up data, and visualizing it, is a good way to inspect the assumptions of a model. Hang on to that intuition, we're going to need it in a minute.*

### **Broken Assumption: Non-Spherical Data**

You argue that the k-means algorithm will work fine on non-spherical clusters. Non-spherical clusters like… these?

<p align="center">
  <img src="https://i.stack.imgur.com/g5Jb8.png">
</p>

Ref: [What is the meaning of spherical dataset?](https://datascience.stackexchange.com/questions/22021/what-is-the-meaning-of-spherical-dataset)

Maybe this isn't what you were expecting- but it's a perfectly reasonable way to construct clusters. Looking at this image, we humans *immediately* recognize two natural groups of points- there's no mistaking them. So let's see how k-means does: assignments are shown in color, imputed centers are shown as X's.
<p align="center">
  <img src="https://i.stack.imgur.com/SlpL1.png">
</p>

You might say “That’s not a fair example… no clustering method could correctly find clusters that are that weird.” Not true! Try [single linkage hierachical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering):

<p align="center">
  <img src="https://i.stack.imgur.com/vBuTf.png">
</p>


Lastly, I should note that K-means is still salvagable! If you start by transforming your data into polar coordinates, the clustering now works:

<p align="center">
  <img src="https://i.stack.imgur.com/0sUph.png">
</p>

### **Broken Assumption: Unevenly Sized Clusters**
What if the clusters have an uneven number of points- does that also break k-means clustering? Well, consider this set of clusters, of sizes 20, 100, 500. Let's generate each from a multivariate Gaussian:\

<p align="center">
  <img src="https://i.stack.imgur.com/WiH4T.png">
</p>

This looks like K-means could probably find those clusters, right? Everything seems to be generated into neat and tidy groups. So let's try K-means:


<p align="center">
  <img src="https://i.stack.imgur.com/zAI1g.png">
</p>
Ouch. What happened here is a bit subtler. **In its quest to minimize the within-cluster sum of squares, the k-means algorithm gives more "weight" to larger clusters**. In practice, that means it's happy to let that small cluster end up far away from any center, while it uses those centers to "split up" a much larger cluster.




### **Conclusion: No Free Lunch**

Sound counterintuitive? Consider that for every case where an algorithm works, we could construct a situation where it fails terribly. *Linear regression assumes your data falls along a line- but what if it follows a sinusoidal wave?* A t-test assumes each sample comes from a normal distribution: what if you throw in an outlier? *Any gradient ascent algorithm can get trapped in local maxima*, and *any supervised classification can be tricked into overfitting*.

What does this mean? It means that ***assumptions are where your power comes from!*** When Netflix recommends movies to you, it's assuming that <ins>if you like one movie, you'll like similar ones (and vice versa)</ins>. Imagine a world where that wasn't true, and your tastes are perfectly random- scattered haphazardly across genres, actors and directors. Their recommendation algorithm would fail terribly. Would it make sense to say "Well, it's still minimizing some expected squared error, so the algorithm is still working"? ***You can't make a recommendation algorithm without making some assumptions about users' tastes- just like you can't make a clustering algorithm without making some assumptions about the nature of those clusters.***

So don't just accept these drawbacks. Know them, so they can inform your choice of algorithms. Understand them, so you can tweak your algorithm and transform your data to solve them. And love them, because if your model could never be wrong, that means it will never be right.

### **References:**
1. [Assumptions Can Ruin Your K-Means Clusters](https://blog.learningtree.com/assumptions-ruin-k-means-clusters/)
2. [How to understand the drawbacks of K-means](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)