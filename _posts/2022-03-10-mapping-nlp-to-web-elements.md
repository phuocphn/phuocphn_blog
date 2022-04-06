---
layout: post
title:  "Mapping natural language commands to web elements"
date:   2021-12-01 14:00:00 +0900
categories: paper_summaries
tags: nlp testing web
author: Phuoc. Pham
comments: true
---


### **Introduction**

The authors consider the task of mapping natural language commands to web page elements (e.g., links, buttons, and form inputs). 
While some commands refer to an element’s text directly, many others require more complex reasoning with the various aspects of web pages: the text, attributes, styles, structural data from the document object model (DOM), and spatial data from the rendered web page.

**Application**

Identifying elements via natural language has several real-world applications. 

1. Providing a voice interface for interacting with web pages, which is especially useful as an assistive technology for the visually impaired
2. Browser Automation: natural language commands are less brittle than CSS or XPath selectors and could generalize across different websites.

### **Problem Definition**

Given a web page $$w$$ with elements $$e_1,...,e_k$$ and a command $$c$$, **the task is to select the element $$e\in {e_1, ... , e_k}$$ described by the command $$c$$**. The training and test data contain $$(w,c,e)$$ triples.

### **Dataset**
Dataset of 51,663 commands on 1,835 web pages is collected by crawling and then labeled by crowdworkers, brainstorming different actions for each web page.
The collected web pages have an average of 1,051 elements, while the commands are 4.1 tokens long on average.

### **Models**

The authors trained 3 different models to solve the above-mentioned task.

#### **Retrival-based**

- It uses the command as a search query to retrieve the most relevant element based on its TF-IDF score.
- Each element is represented as a bag-of-tokens computed by `tokenizing and stemming its text content` + `tokenizing the attributes`
- When computing term frequencies, we down-weight the attribute tokens by afactor of $$\alpha=3$$.
- The document frequencies are computed over the web pages in the training dataset.

#### **Embedding-based**

Follow a common method for matching two pieces of text is to embed them separately and then compute a score from the two embeddings.

For a command $$c%%$$ and elements $$e1 , ... , e_k $$ , we define the following conditional distribution over the elements:

$$p (e_i | c) ∝ \textit{exp} [s(f(c), g(e_i))]$$

where:


- $$f(c)$$ is the embedding of $$c$$,  each token of $$c$$ is embeded into a fixed-dimensional vector and take an average over the token embeddings. (The token embeddings are initialized with GloVe vectors.) 
- $$g(e_i)$$ is the embedding of $$e_i$$, by embedding the properties of $$e_i$$, concatenate the results, and then apply a linear layer to obtain a vector of the same length as $$f(c)$$. 
- $$s$$ is a scoring function. For calculating $$s(f(c),g(e))$$, we first let $$f(c)$$ and $$g(e)$$ be the results of normalizing the two embeddings to
unit norm. Then a linear layer is applied on the concatenated vector $$[f(c);g(e);f(c) ◦ g(e)]$$ (where $$◦$$ denotes the element-wise product).


The model is trained to maximize the log-likelihood of the correct element in the training data.

#### **Alignment-based**

Let $$t(e)$$ be the concatenation of $$e$$’s text content and text attributes of $$e$$, trimmed to 10 tokens. 

1. We construct a matrix $$A(c, e)$$ where each entry $$A_{ij}(c,e)$$ is the dot product between the embeddings of the $$i$$-th token of $$c$$ and the $$j$$-th token of $$t(e)$$. 
2. Then we apply two convolutional layers of size 3×3 on the matrix, apply a max-pooling layer of size 2×2, concatenate a tag embedding, and then apply a linear layer on the result to get a 10- dimensional vector $$h(c, e)$$.
3. We apply a final linear layer on $$h(c, e)$$ to **compute a scalar score**, and then train on the same objective function as the encoding-based model. To incorporate context, we simply concatenate the four vectors $$h(c, n_d(e))$$ of the neighbors $$n_d(e)$$ to the final linear layer input.

In a simple words, this method tries to align the query command $$c$$ and each element $$e_i$$ to form a 2D matrix (image-like format) and use CNN as a feature extractor to calculate the score similarity between the query $$c$$ and the selected element.

### **Experiment**

#### **Main Result**
We train the neural models using Adam (Kingma and Ba, 2014) with initial learning rate $$1e−3$$, and apply early stopping based on the development set. The models can choose any element that is visible on the page at rendering time.

The experimental results are shown in Table 1. Both neural models significantly outperform the retrieval model.

![https://i.imgur.com/lmQsLmB.png](https://i.imgur.com/lmQsLmB.png)

#### **Error Analysis**

To get a better picture of how the models han- dle different phenomena, we analyze the pre- dictions of the embedding-based and alignment- based models on 100 development examples where at least one model made an error. The errors, summarized in the following table:

![https://i.imgur.com/RnkEi7y.png](https://i.imgur.com/RnkEi7y.png)

### **Conclusion**
We presented a new task of grounding natural language commands on openended and semistructured web pages. With different methods of referencing elements, mixtures of textual and non- textual element attributes, and the need to properly incorporate context, our task offers a chal- lenging environment for language understanding with great potential for real-world applications.

