---
layout: post
title:  "AI/Machine Learning Interview Preparation."
date:   2021-11-04 19:24:00 +0900
categories: ml 
tags: interview
author: Phuoc. Pham
---

I recently watched some videos on Youtube about ML interview preparation and the most frequently asked questions during the interview. This post will serve as a reference resource for the future interviews and other viewers, who is also looking for a new ML job position.

The content of this post is mostly inspired from the talk of Huyen Chip (Nvidia) and EMMA DING (Data Interview Pro).

Before going the interview preparation section, it is important to understand the differences between ML positions. Finding the correct position may help you eliminate inrelevant information and shorten the preparation time.


### **The Differences between Job Titles.**

#### **Research vs Applied Research** 
**Research**: This job involves *finding the answers* for fundamental questions and expand the body of theoretical knowledge. (ex: develop a new learning method for unsupervised transfer learning). It usually focus on long term outcome.

**Applied Research**: This job involves *finding the solutions* to practical problems, ( ex: develop techniques to make that new learning method work on a real world dataset). Focus on immediate commercial outcome.

*Caveats:* 
- Cutting-edge research is spearheaded by big corporations
- Lacking theories to explain methods that work well empirically.

#### **Research Scientist and Research Engineer**

**Research Scientist**: Develop original ideas. It might requires PhDs, and might act as an advisor to research engineers

**Research Engineer**: Use engineering to actualize these ideas. Don't require PhDs, Springboard to become research scientist.

*Caveats:*
- Depends on organizations/teams. In some teams, there are virtually no difference.
- Scientists & engineers can be equal first authors (e.g GPT-2, Transformer paper)

#### **Data Scientist and ML engineer**

**Data Scientist:** Extract knowledge and insights from structured and unstructured data. Use data to help company make decisions. And to be clear, Data scientist is a SCIENTIST, engineering isn't a top priority.

**ML Engineers:** ML models learn from data, (ML is part of data science). Develop models to turn data into products. And ML engineers is an ENGINEER , it is a top priority.

*Caveats:*

- Machien Learning Engineers at startups might spend most of their time wrangling data, understanding data, setting up infrastructure, and deploying models instead of training ML models.

### **ML at Big Companies $$\ne$$ ML at Startups.**
**Big companies:** Can afford research, can afford specialists, standardized process.

**Startups:** CAN'T (It is quite opposite to big companies), and it usually needs *generalists*, make up process as they go. (more flexible)

### **Six common PATHs to become ML Engineers**.

1. BS/MS in ML $$\to$$ ML Engineer (Tech Ivies  $$\to$$ FAANG/startups)
2. PhD in ML  $$\to$$ ML research (Published at top-tier conferences  $$\to$$ FAANG/ML - top startups)
3. Data scientist  $$\to$$ On-job training  $$\to$$ ML engineer/researcher ( Companies want to start using ML)
4. Software Engineer  $$\to$$ Courses  $$\to$$ ML Engineer (Software engineers want to transition into ML)
5. Adjacent fields  $$\to$$ On-job training  $$\to$$ ML research (Not enough talents from AI/ML PhDs, ex: stats, physis, math)
6. Unrelated fields  $$\to$$ residency/fellowship  $$\to$$ ML researcher



```Avoid anyone that promises you ML expertise in days or weeks!```


### **Understanding the Interviewers' Mindset.**

For hiring <ins>senior role</ins>: The interviewers usually want to hire for **skills**, while for the <ins>junior role</ins>, the interviewers usually want to hire for **attitude.** (wheather you are fit to the company environment or not, and also about budget constraints.)

1. **Companies hate hiring**
    - Expensive for companies
    - Stressful for hiring managers
    - Boring for interviewers
2. **Companies don't want the best people**: They want the best people who can do a reasonable job within the time and monetary constraints
3. **Companies don't know what they're hiring for**: They don't even know for sure if they'll need that person -> job descriptions for reference purpose only.
4. **Most recruiters can't evaluate technical skills**
They rely on weak signals:
    - previous employers
    - degrees
    - awards/ papers
    - Github/Kaggle
    - referrals.
PSA: Past projects aren't meritocratic
- Not everyone can afford to contribute to OSS or do Kaggle competitions.
- Placing too much importance on voluntary activities punishes candidates from less privileged background.
5. **Most interviewers are bad**: Little or no training for interviewers, even at big companies.
6. **Interview outcome depends on many random variables.**
It is, in no way, a reflection of your ability or your self-worth.


### **Interview Process Overview**
Actually, it is evolved out of the traditional software engineering interview process.

- Resume screen 
- Phone screen 
- Coding challenges / take-home assignments
- Technical offsite interviews 
- Onsite interviews


During the interview, you will probably encounter some bad questions as follow, you should prepare and try to avoid / redirect to another question instead of trying to give the un-completed answers. If you got many bad questions, just ask again what they are looking for ?

**Bad interview questions**
1. Questions that ask for the retention of knowledge that can be easily looked up "*write down the equation for the Adam optimizer*"

2. Questions that evaluate irrelevant skills : "*write a linked list*"
3. Questions whose solutions rely on one single insight "*take derivative of x^x*"
4. Questions that try to evaluate multiple skills at once "*explain PCA to your grandma*"
5. Questions that use specific hard-to-remember names:    "*Moore-Penrose inverse*" or "*Frobenius norm*"
6. Open-ended questions with one expected answer.
7. Easy questions during later interview rounds "*find the longest common subsequence*"




### **Different Types of ML Problems in the Interview.**
  - **ML Basic, ML background** (easy to prepare & converd in basic ML course)
  - **Questions from your resume** (walk through previous ML projects, italso requires you to have the ability to summarize the experience)
    It usually requires you to describle project in a high level, and dive into technical details later.
    How you can prepare ?

    You have to think through your projects before the interview.
      - What are the models and how you use them ?
      - Pros and cons ?
      - Why you use them vs other methods ? 
      
    
  The important thing is: You have to convince the interviewer that <ins>you understand the models + you gained experience with this project. </ins>

  - **ML coding**
    Understanding of the theory and ability to code up from scratch in short of time.
    It usually requires to do it on online IDE or whiteboard.
    Sounds daunting ? No, it is not, due to the time limitations, only a few are asked in interviews.

    <ins>Most commonly asked algorithms: </ins>
    Supervised learning: Decision tree & Linear and logistic regression, K nearest neighbors.
    Unsupervised learning: K-means

    How to prepare:
      - Try to implement yourself first
      - Search online when getting stuck 
      - Practice before interview

Things to notice: The efficiency of your implementation + Time and space complexity in Big-O notation.

  - **Applied ML**: This type of question requires you to be familiar with:
      - Entire workflow
      - Getting and cleaning the data
      - Building and evaluating models.
      - Shipping the model to production.

    During the interview: The interviewer will ask about about open-ended problems + your solution + follow-up questions.

    Example: How to detect spam emails ? How to apprach the problem ?.  This a step-by-step example approach to answer the problem.

    1. You should ask the interview to clarify the questions:
      - What data is available ?
      - The data format ? 
    2. Describle the high level workflow
      - Data collection
      - Data processing
      - Model selection
      - Model evaluation
    3. Dive deep into each component.
      - Discuss the design with the interviewer.

  - **Alternative interview formats**: multiple choice quiz, quiz, code debugging, pair programming, good cop, bad cop.


### **Tips for ML Interviewee**
  1. You should give examples (It is the best way to demonstrate your knowledge.)
    E,g: What is precision (you can mention this definition: true positive / detected positive,  and then give an example: A COVID test detects 100 positives 99 are true positives, therefore, the precision is 99%.)
  2. Don't mention things you aren't familiar with
    Everything you say may lead to follow-up questions.
  3. Job search and interview preparation are lifelong processes, The best time to interview is when you don't need a job. 
  4. Start looking for jobs 3-6 months before.
  5. Build up your portfolio and publish them.
  6. Get people to refer you.
  7. Look up your interviewers. Review their work
  8. Have your friends to give you mock interviews.
  9. Don't pretend that you know something when you don't.
  10. Don't criticize your previous or current employers
  11. Don't talk about your age, marital status, religion, political affiliation.
  12. Have competing offers
  13. Don't sweat it. If you tank an interview, move on.









### **Most common asked questions**:

**Precison & Recall**

[Precision vs. Recall – An Intuitive Guide for Every Machine Learning Person](https://www.analyticsvidhya.com/blog/2020/09/precision-recall-machine-learning/)

[Classification: Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)


**Eigenvectors and Eigenvalues**

**Google Interview Questions**

- Given the following dataset, can you predict how K-means clustering works on it ? Explain.
- Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or not ?


---



### **References**
1. [Chip Huyen on Machine Learning Interviews (Full Stack Deep Learning - November 2019)](https://www.youtube.com/watch?v=pli1K75PSa8)
2. [Machine Learning Interview Questions and Answers, Edureka](https://www.youtube.com/watch?v=t6gOpFLt-Ks&ab_channel=edureka%21)
3. [Google Machine Learning System Design Mock Interview](https://www.youtube.com/watch?v=uF1V2MqX2U0&ab_channel=DataScienceJay)
4. [Cracking Machine Learning Problems, Data Science Interviews](https://www.youtube.com/watch?v=21E-bUnGQQ4&ab_channel=DataInterviewPro)