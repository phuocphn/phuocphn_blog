---
layout: post
title:  "Style-Guided Web Application Exploration"
date:   2022-04-06 12:48:00 +0700
categories: paper_summaries
tags: style exploration testing web
author: Phuoc. Pham
comments: true
---

Authors: *Davood Mazinanian, Mohammad Bajammal, Ali Mesbah*

**Last update**: 12:48 PM April 06, 2022 

Finding out which elements are actionable in web apps is not a trivial task. To improve the exploring efficiency, we propose a **browser-independent**, **instrumentation-free** approach based on *structural and visual stylistic cues*. 

Our approach, implemented in a tool called **StyleX**, employs machine learning models to predict actionable elements on a webpage. 

## **Introduction**

Automated exploration (usually) uses a web crawler to mimic user behaviours. 

A crawler needs to effectively explore the *event space*. To that end, in each new state of the web app, the crawler needs to (1) ***identify web elements that can potentially trigger an event (***e.g., elements with click or mouseover events), and (2) ***determine the order or sequence of which web element to exercise next.***

#### **Problem #1**

The first step of identifying web elements that have event listeners (actionables) is not straightforward.

- Only considering “clickable” elements by default such as `<a>, <button>`,  undermine the crawler’s ability of discovering new states as other elements as `<div>` and `<span>` may also have event listeners.
- Nowadays, event listeners are attached to web elements via DOM APIs `addEventListener()`. Hence, only using HTML attribute for identifying “actionable” elements is much more challenging.

Current workarounds: use an instrumented browser engine.

#### **Problem #2**

Once the set of actionables is identified, **the order in which the actionables are explored by the crawler can also impact the state exploration of the web app.** 

Note that **this is different from choosing the next web app state to explore**, from a given list of already explored states (i.e., the state exploration strategy). In contrast, the
issue at hand here is concerned with **ranking the execution order of actionables on a given state.** Existing crawlers often fire events in a random or top-down order [31].

#### **Solution**

This paper shows that these challenges can be tackled by using the stylistic information of
web elements. 

The authors propose a technique that identifies actionables based on the insight that a web element’s structural and visual styles (i.e., their DOM location and the way they look) can
potentially indicate whether they have events attached to them. 

These structural and visual stylistic features **can provide an effective event ranking strategy during crawling to achieve a higher code coverage**. Our ranking approach essentially exploits the Consistent Identification usability guideline [1]: elements with similar functionality should have a consistent presentation across the web app. **By postponing the execution of actionables that look similar, we aim at diversifying the covered functionality.**


💡 [1] World Wide Web Consortium. 2016. Web Content Accessibility
Guidelines 2.0: Consistent Identification. [https://www.w3.org/TR/](https://www.w3.org/TR/) UNDERSTANDING-WCAG20/consistent-behavior-consistent-functionality. html/ Accessed: 15 Feb 2019.




## **Background and Motivation**

### **The internals of modern crawlers**

![internal_working](https://i.imgur.com/LjJ0mNN.png)

The crawler navigates a web browser to a particular URL, and then extracts actionables in the loaded page. These elements are **candidate actionables** for a crawler, since it does not know whether interacting with them will lead to a new state a priori. 

Subsequently, the crawler chooses the next event corresponding to a candidate actionable, fires it, and monitors the web browser to see whether there is a change in the current state of the web app.

Crawlers often allow custom state abstraction functions; The crawler then chooses the next state to expand based on a **state exploration strategy.** While crawling, the crawler can construct a model of the web app (state-flow graph) under analysis using the states and transitions between them. This can be a graph is then used for different purposes, e.g., automated test case generation.

### **Motivation Example**

If we were to use an automated crawler to analyze a following Google Calendar app we would observe the following:

![Untitled](https://i.imgur.com/K3xTpEP.png)

**Actionables**. Most web crawlers focus on hyperlinks and **miss a large portion of actionables** to be fired  as most of actionable elements in Google Calendar are <div>s and not hyperlinks
(i.e., <a> tags).

**Equivalent classes of actionables**. There are usually equivalent classes of actionables that appear within or across different states, and share similar functionality. (e.g D~B group). 
In such cases where the corresponding functionalities are similar (or identical), *firing events on these similar actionables is unlikely to result in new states or more JavaScript code coverage.* 

## **Approach**

### **Predicting actionables**

**Intuition**:  stylistic features can be used to train a machine learning model to predict which elements on the page are actionable.

**Data collection**:  To collect data for training a model, we need a large set of pre-identified (1) actionable elements to be used as ***positive examples***, and (2) elements without any event listeners to be used as ***negative examples***.

1,000 websites is randomly selected from Alexa’s top ranking.  For each website, a scripted is used to travers the DOM loaded in the web browser, and collects all the HTML elements present in the web page. For each element, **the script collects any attached event listeners** using Chrome DevTools.

If an element has an attached event listener, it is considered actionable and is stored for later analysis along with the type of the event(s) handled. In addition, we con HTML elements which are actionable by default, such as hyperlinks and buttons, **as positive examples**, regardless if whether they have a JavaScript event listener ( as it can change the state even without an explicitly attached JavaScript event

**Incorporating event propagation.** Firing certain events on DOM elements causes the same event to be triggered on the element’s ancestors. e.g: *when a user clicks on a button, all the button’s ancestor elements are also clicked too through event propagation.* 

The order in which the event listeners of the same type attached to the ancestor elements are executed can be different: i.e., 

- first execute the events attached to the ancestors, going down in the DOM hierarchy—**the capture phase.**
- —or first execute the events attached to the element itself, going up in the hierarchy—**the bubble phase**.

To incorporate this behavior in the training model, for the event handlers that bubble, **we mark the descendant elements of an actionable to be actionable too.** This can improve the model performance as most of structural and visual styles are also inherited by the descendant elements.

**Training features.**  There are 68 features of structural and visual styles  (extracted via the DOM API) for each element, which will be used as learning features in the trained models.  

![Untitled](https://i.imgur.com/eTefI9Q.png)

**Training and testing sets.**  80%-20% split ratio for cross validation.

**Balancing positive/negative examples**: under-sampling, i.e., randomly removing elements from negative examples until the two sets have the same number of elements.

**Choosing event types.** The five most frequent event types are considered: `click, mouseover, mouseout, and mousedown, and touchstart`. For each of these event types, we train a separate binary model that predicts whether a certain element has a listener for this given event type.
**Machine learning algorithms**. The authors trained with multiple classifiers as CART, C4.5, and C5.0 decision trees, random forests, and NN, eventually selected the model with the highest accuracy and deployed it to a general-purpose crawler.

### **Prioritizing actionables using styles**

**Intuition**: actionables with dissimilar appearance might be better candidates to be examined earlier, as similarly-looking elements might represent the same functionality in the web app.

**Representing actionables.** The main goal is to identify elements with similar appearance across different states of the web app, so that when the crawler comes across a new actionable, it can be **ranked** w.r.t the ones that have already been explored. *To do so, we represent actionables in such a way that they can be compared across different states.*

For purposes of ranking, each actionable is represented using a vector of features $f$ consisting of stylistic properties in the above table. (all CSS properties + without DOM-related properties).

## **Experiments + Results**

*(still in the process of writing...)*