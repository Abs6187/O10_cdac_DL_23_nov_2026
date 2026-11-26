'''
This file covers
1. Neural Network and its applications, single layer neural network
2. Activation functions: Sigmoid, ReLU, Tanh
3. Overview of Backpropagation of error

Scripts: implementation of single layer neural network with different activation functions
'''

'''
You Already know:
1. Classic ML (Linear Regression, Logistic Regression, Decision Trees, SVM, KNN, gradient boosting)
2. Data Analytics (Exploratory Data Analysis, Feature Engineering, Data Preprocessing)

Now the conceptual shift to DL is
1. Deep Learning models are capable of learning complex patterns from large amounts of data.
    - representational learning: automatically learning features from raw data
    - composable differentiable computation graph
    - end-to-end learning: learning directly from input to output without manual feature engineering
'''
'''
ML vs DL 
1. ML is domain engineered, shallow decision-making, while DL is data-driven.
Neural Networks (NN) learn hierarchical representations of data, where each layer captures increasingly abstract features.


Where DNNs win and whare tabular/classical ML win?
DNNs excel in tasks involving unstructured data such as images, audio, and text, where hierarchical feature learning is crucial.
Classical ML methods often perform better on tabular data with well-defined features and smaller datasets.
'''

'''
Single Layer Neural Network (SLNN) consists of an input layer and an output layer with no hidden layers.
Also called a perceptron.
It can be used for binary classification tasks where the output is either 0 or 1.
The output is typically obtained by applying an activation function such as the sigmoid function to the weighted sum of inputs.
Mathematically, it can be represented as:
    y = f(w1*x1 + w2*x2 + ... + wn*xn + b)
where f is the activation function, wi are the weights, xi are the input features, and b is the bias term.

'''

'''
Example:
'''
# Example: Single Layer Neural Network for binary classification
import numpy as np

np.random.seed(42)  # for reproducibility

# Single-layer unit demo: z = w . x + b
# What this function does:
# It computes the forward pass of a single-layer neural network.
# X: input features, w: weights, b: bias, activation_fn: activation function
# Returns: z (weighted sum), a (activation output)
def single_layer_forward(X, w, b, activation_fn):
    z = X @ w + b
    a = activation_fn(z) # a is the activation output : f(z)
    return z, a


def main1():
    X_sample = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    ])
    w_demo = np.array([1.5, -2.0])
    b_demo = 0.25

    z_out, a_out = single_layer_forward(X_sample, w_demo, b_demo, lambda z: z)
    print("Pre-activations z:", np.round(z_out, 4))
    print("Activation output a:", np.round(a_out, 4))

 # ----------------------------

def main2():
    # Delivery ETA[Estimated Time of Arrival] Risk (Distance vs. Traffic Congestion) 
    # Rows: [normalized_distance, traffic_index]
    X_delivery = np.array([
    [0.2, 0.1],  # Short hop, light traffic
    [0.8, 0.2],  # Long route, light traffic
    [0.3, 0.9],  # Short hop, heavy traffic
    [0.9, 0.85], # Long route, heavy traffic
    ])

    w_delivery = np.array([1.8, 3.4])
    b_delivery = -1.5 # bias term for the delivery model : baseline offset for the weighted sum
    # bias : is determined based on expert knowlegde or prior experience to set a baseline for the model
    '''
    with b_delivery = 0 the model would have no baseline offset and the weighted sum would be purely determined by the input features and their corresponding weights.
    with b_delivery = -1.5 the model has a negative baseline offset, which can help to adjust the weighted sum downwards before applying the activation function.
    with b_delivery = - 3.0 the model has an even stronger negative baseline offset, further adjusting the weighted sum downwards before applying the activation function.

    bias = 0 -> crosses 0.5  when z(raw) > 0 (very sensitive/ triggers easily)
    bias = -1.5 -> crosses 0.5  when z(raw) > 1.5 (moderate sensitivity)
    bias = -3.0 -> crosses 0.5  when z(raw) > 3.0 (low sensitivity / triggers less easily / high hurdle)
    '''
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z)) # formula : sigmoid(z) = 1 / (1 + e^(-z)) : 

    z_del, a_del = single_layer_forward(X_delivery, w_delivery, b_delivery, sigmoid)
    print("Pre-activations z (delivery):", np.round(z_del, 4))
    print("Activation output a (delivery):", np.round(a_del, 4))


def main3():
    # Rows: [cpu_utilization_ratio, memory_utilization_ratio]
    X_infra = np.array([
    [0.25, 0.30], # Healthy node
    [0.75, 0.40], # CPU-heavy
    [0.35, 0.85], # Memory-heavy
    [0.92, 0.90], # Saturated node
    ])

    w_infra = np.array([2.0, 2.2])
    b_infra = -2.1 # bias term for the infrastructure model : baseline offset for the weighted sum

    relu = lambda z: np.maximum(0, z)
    z_inf, a_inf = single_layer_forward(X_infra, w_infra, b_infra, relu)
    print("Pre-activations z (infrastructure):", np.round(z_inf, 4))
    print("Activation output a (infrastructure):", np.round(a_inf, 4))
    
def main4():
    # Rows: [required_run_rate_norm, wickets_remaining_norm]
    X_cricket = np.array([
    [0.40, 0.80], # Cruising chase (low RRR, 8 wickets left)
    [0.85, 0.70], # High chase rate, plenty of wickets
    [0.85, 0.20], # High chase rate, few wickets left
    [0.55, 0.10], # Moderate RRR, tailenders at crease
    ])

    w_cricket = np.array([3.0, -1.8])
    b_cricket = -0.5

    tanh_fn = np.tanh
    z_cri, a_cri = single_layer_forward(X_cricket, w_cricket, b_cricket, tanh_fn)
    print("Pre-activations z (cricket):", np.round(z_cri, 4))
    print("Activation output a (cricket):", np.round(a_cri, 4))
    
'''
how do you decide the weights and bias for a single-layer neural network?
Bias : ?
Weights : ?

how to decide a baseline bias?

'''

if __name__ == "__main__":
    # main1()
    # main2()  # Example with activation function (sigmoid)
    # main3()  # Placeholder for another example if needed
    main4()  # Placeholder for future example