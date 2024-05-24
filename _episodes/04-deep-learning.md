---
title: Deep Learing Fundamentals
teaching: 1
exercises: 0
questions:
- "What are the basic timeseries I can use in pandas ?"
- "How do I write documentation for my Python code?"
- "How do I install and manage packages?"
objectives:
- "Brief overview of basic datatypes like lists, tuples, & dictionaries."
- "Recommendations for proper code documentation."
- "Installing, updating, and importing packages."
- "Verify that everyone's Python environment is ready."
keypoints:
- "Are you flying yet?"
---
<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>



 Advances in computational power, experimental techniques, and simulations are producing vast quantities of data across fields like particle physics, cosmology, atomospherica science, materials science, and quantum computing. However, traditional analysis methods struggle to keep up with the scale and complexity of this "big data". Machine learning algorithms excel at finding patterns and making predictions from large, high-dimensional datasets. By training on this wealth of data, ML models can accelerate scientific discovery, optimize experiments, and uncover hidden insights that would be difficult for humans to discern alone. As the pace of data generation continues to outstrip human processing capabilities, machine learning will only grow in importance for driving progress in the physical sciences.


Ultimately, machine learning represents a valuable addition to the climate scientist's toolbox, but should be applied judiciously in conjunction with established physical and statistical methods to gain the most robust insights about the Earth's complex climate system



## ANN


ANNs consists of multiple nodes (the circles) and layers that are all connected and using basic math gives out a result. These are called feed forward networks. 

![](../fig/ANN_forward.png)

 In each individual node the values coming in are weighted and summed together and bias term is added and activation. Hence, the linear regression mapping by an activation function to produce non-linear model as shown below:

 \[ Z = \sigma( W^T \cdot X) \]


 ![](../fig/ANN_activation.png)


 ### Activation functions

Activation function determines, if information is moving forward from that specific node.
This is the step that allows for nonlinearity in these algorithms, without activation all we would be doing is linear algebra. Some of the common activation functions are indicated in figure below:

 ![](../fig/ANN_activation2.png)


So training of the network is merely determining the weights “w” and bias/offset “b"  with the addition of nonlinear activation function. Goal is to determine the best function so that the output is as  correct as possible; typically involves choosing “weights”. 


### Loss Function

You know the data and the goal you’re working towards, so you know the best, which loss function to use. Basic MSE or MAE works well for regression tasks. The basic MSE and MAE works well for regression task is given by:

\[ loss = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_{i} - y_{i})^2 \]



The quantinty you want ot determine("loss") help to determine the best weights and bias terms in the model. Gradient descent is a technique to find the weight that minimizes the loss function.  This is done by starting with a random point, the gradient (the black lines) is calculated at that point. Then the negative of that gradient is followed to the next point and so on. This is repeated until the minimum is reached.

![](../fig/loss_function.png)



The gradeint descent formula tells us that the next location depends on the negative gradient of J multiplied by the learning rate \(\lambda\).

\[ J_{i+1} = J_{i} - \lambda \nabla J_{t} \]


As the loss function depends on the linear function and its weights \(w_0\) and \(w_1\), the gradient is calculated as parital derviatives with relation to the weights.

![](../fig/loss_function2.png)


The only other thing one must pay attention to is the learning rate \(lambda\) (how big of a step to take). Too small and finding the right weights takes forever, too big and you might miss the minimum.

\[ w_{i+1} = w_i - \lambda \frac{\partial J}{\partial w_i}\]


Backpropagation is a technique used to compute the gradient of the loss function when its functional form is unknown. This method calculates the gradient with respect to the neural network's weights, allowing for the optimization of these weights to minimize the loss. A critical requirement for the activation functions in this process is that they must be differentiable, as this property is essential for the gradient computation necessary in backpropagation.

\[ \frac{\partial J}{\partial w_k} = \frac{\partial}{\partial w_k}\begin{pmatrix} \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2\end{pmatrix} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)\frac{\partial \hat{y}}{\partial w_{k}}\]


