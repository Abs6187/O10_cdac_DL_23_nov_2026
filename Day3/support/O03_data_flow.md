## Journey of Data: The Feed-forward Mechanism

Data moves forward through interconnected artificial neuron layers, transforming step by step from raw input to prediction.

* **The Initial Point - Input Layer:** Ingests data (often as a vector) matching input dimensionality—for example, 784 neurons for a 28×28 grayscale image.
* **Hidden Layers - The Transformation Hubs:** Perform most data transformations. Each neuron computes a weighted sum, applies an activation function, and passes the result forward using **weights** to modulate signals between layers.
* **The Termination - Output Layer:** Produces the final prediction or classification. Size depends on the task—for example, 1 neuron for binary classification or 10 neurons for 10-class classification.

## Activation Functions: Giving Neurons their Non-linearity

Activation functions let neurons model complex, non-linear relationships rather than pure linear transformations.

* **Nature of Activation Functions:** Mathematical functions on neuron outputs that create non-linear decision boundaries.
* **Common Activation Functions:**
  * **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$. Computationally efficient and widely used to train deep networks.
  * **Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$. S-shaped curve mapping inputs to $(0, 1)$.
  * **Tanh (Hyperbolic Tangent):** $\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$. Zero-centered alternative to sigmoid outputting values between $-1$ and $1$.
  * **Softmax:** Commonly used in multi-class output layers to produce a probability distribution over classes.
* **Importance:** Without activation functions, stacked layers collapse into the expressive power of a single-layer perceptron and cannot approximate complex non-linear functions.


## Learning and Adaptation: Backpropagation

Backpropagation ("backward propagation of errors") trains deep neural networks by computing the gradient of the loss function with respect to each weight using the chain rule.

* **Feedforward Step:** Passes an input forward through the network to generate a prediction.
* **Compute Loss:** Measures the error between the predicted output and the actual target value.
* **Backward Pass:** Propagates the error backward through the network by applying the chain rule to compute gradients for every weight.

## Understanding Errors: The Cost Function

The cost (or loss) function quantifies how closely the network's predictions match actual target values.

* **Mean Squared Error (MSE):** Used for regression; computes the average squared difference between predicted ($\hat{y}_i$) and actual ($y_i$) values:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Cross-Entropy Loss:** Used for classification; measures the difference between the true probability distribution and the model's predicted distribution.
* **Choosing a Loss Function:** Match the loss function to the task—Cross-Entropy for classification, MSE for regression.

## Gradient Descent: Searching for the Optimal Weights

Gradient descent iteratively minimizes error by updating weights in the opposite direction of the cost function's gradient.

* **Learning Rate ($\alpha$):** Hyperparameter controlling step size per update; too small slows convergence, while too large risks overshooting or divergence.
* **Variants:**
  * **Batch Gradient Descent:** Computes the gradient using the entire dataset.
  * **Stochastic Gradient Descent (SGD):** Updates weights using a single sample per iteration.
  * **Mini-Batch Gradient Descent:** Balances speed and stability using a small subset (mini-batch) of samples.

## Backpropagation in Action: Adjusting Weights to Minimise Error

Once gradients are computed, weights and biases are updated to reduce network error.

* **Chain Rule Application:** Enables efficient gradient calculation across every layer in deep architectures.
* **Weight Update:** Adjusts each weight using the learning rate $\alpha$ and the cost gradient:
  $$W_{\text{new}} = W_{\text{old}} - \alpha \times \frac{\partial \text{Cost}}{\partial W_{\text{old}}}$$
* **Bias Update:** Adjusts network biases using the same gradient descent rule.


## Enhancing Learning: Techniques and Tricks

Deep neural networks learn complex non-linear patterns well but face **slow convergence** and **overfitting**. Momentum, learning rate adjustments, and regularization address these issues.

### 1. Momentum and Learning Rate: Speeding up Convergence

**Convergence** is reducing training loss to an optimal or near-optimal level. Standard gradient descent can converge slowly, oscillate, or stall in local minima.

* **Momentum:**
  * **Principle:** Like a ball rolling downhill, it accelerates updates along consistent gradients and dampens oscillations across noisy directions.
  * **Mathematical Representation:** Incorporates a fraction $\gamma$ of the prior update with learning rate $\eta$ and loss gradient $\nabla L$:
    $$\Delta w^{(t)} = \gamma \Delta w^{(t-1)} + \eta \nabla L$$
  * **Benefits:** Reduces oscillations and helps escape shallow local minima.
* **Learning Rate Adjustments:**
  * **Principle:** Controls step size per update—too large causes divergence, too small causes slow convergence.
  * **Adaptive Learning Rates:** Methods like **Adagrad**, **RMSprop**, and **Adam** scale learning rates using gradient history for faster, more stable training.
  * **Benefits:** Speeds up convergence and reduces manual learning rate tuning.

### 2. Regularisation: Preventing Overfitting in Neural Networks

**Overfitting** occurs when a model memorizes training data and generalizes poorly to unseen data. Regularization adds complexity penalties or constraints to improve generalization.

* **L1 and L2 Regularization:**
  * **Principle:** Penalizes parameter magnitude—**L1** (Lasso) adds an absolute-value penalty; **L2** (Ridge) adds a squared penalty.
  * **Benefits:** L1 aids feature selection; L2 keeps weights small.
* **Dropout:**
  * **Principle:** Randomly drops neurons during forward and backward passes in training.
  * **Benefits:** Prevents neuron co-adaptation and acts like training an ensemble, improving generalization.
* **Early Stopping:**
  * **Principle:** Stops training once validation performance starts degrading.
  * **Benefits:** Prevents fitting training noise and yields better generalization.