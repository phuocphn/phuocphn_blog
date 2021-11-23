#### **Inductive Bias**

Every machine learning algorithm with any ability to generalize beyond the training data that it sees has some type of inductive bias, which are the assumptions made by the model to learn the target function and to generalize beyond training data.

For example, in linear regression, the model assumes that the output or dependent variable is related to independent variable linearly (in the weights). This is an inductive bias of the model.


---

Inductive bias is the set of assumptions a learner uses to predict results given inputs it has not yet encountered.

---

Inductive bias can be thought of as the set of assumptions we make about a domain which we are trying to learn about.

Technically, when we are trying to learn Y from X and, initially, the hypothesis space (different functions for learning X->Y) for Y is infinite. To learn anything at all, we need to reduce the scope. This is done in the form of our beliefs/assumptions about the hypothesis space, also called inductive bias.

Through the introduction of these assumptions, we constrain our hypothesis space and also get the capability to incrementally test and improve on the data in the form of hyper-parameters.

Examples of inductive bias -

Linear Regression: Y varies linearly in X (in parameters of X).
Logistic Regression: There exists a hyperplane which separates negative / positive examples
Neural Networks: crudely speaking, Y is some non-linear function of X (the non linearity depends on the activation functions, topology etc.)


https://stackoverflow.com/questions/35655267/what-is-inductive-bias-in-machine-learning