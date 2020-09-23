# The Back-propagation Algorithm
Implementation of the back-propagation algorithm using only the linear algebra and other mathematics tool available in numpy and scipy.

We will restrict ourselves to fully-connected feed forward neural networks with one hidden layer (plus an input and an output layer).

Credits to Jón Guðnason and Eyjólfur Ingi Ásgeirsson on their Data Mining and Machine Learning course at RU.

### The Sigmoid function
We will use the following nonlinear activation function:

$$\sigma(a)=\frac{1}{1+e^{-a}}$$

We will also need the derivative of this function:

$$\frac{d}{da}\sigma(a) = \frac{e^{-a}}{(1+e^{-a})^2} = \sigma(a) (1-\sigma(a))$$

**Note**: To avoid overflows inside `sigmoid(x)` we check if `x<=100` and return `0.0` in that case.

Example inputs and outputs:
* `sigmoid(0.5)` -> `0.6224593312018546`
* `d_sigmoid(0.2)` -> `0.24751657271185995`

### The Perceptron Function
A perceptron takes in an array of inputs $X$ and an array of the corresponding weights $W$ and returns the weighted sum of $X$ and $W$, as well as the result from the activation function (i.e. the sigmoid) of this weighted sum.

The function `perceptron(x, w)` returns the weighted sum and the output of the activation function.

Example inputs and outputs:
* `perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1]))` -> `(1.0799999999999998, 0.7464939833376621)`
* `perceptron(np.array([0.2,0.4]),np.array([0.1,0.4]))` -> `(0.18000000000000005, 0.5448788923735801)`

### Forward Propagation
When we have the sigmoid and the perceptron function, we can start to implement the neural network.

The function `ffnn` computes the output and hidden layer variables for a single hidden layer feed-forward neural network. If the number of inputs is $D$, the number of hidden layer neurons is $M$ and the number of output neurons is $K$, the matrices $W_1$ of size $[(D+1)\times M]$ and $W_2$ of size $[(M+1)\times K]$ represent the linear transform from the input layer and the hidden layer and from the hidden layer to the output layer respectively.

`y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)`:

* `x` is the input pattern of $(1\times D)$ dimensions (a line vector)
* `W1` is a $((D+1)\times M)$ matrix and `W2` is a $(M+1)\times K$ matrix. (the `+1` are for the bias weights)
* `a1` is the input vector of the hidden layer of size $(1\times M)$ (needed for backprop).
* `a2` is the input vector of the output layer of size $(1\times K)$ (needed for backprop).
* `z0` is the input pattern of size $(1\times (D+1))$, (this is just `x` with `1.0` inserted at the beginning to match the bias weight).
* `z1` is the output vector of the hidden layer of size $(1\times (M+1))$ (needed for backprop).
* `y` is the output of the neural network of size $(1\times K)$.

Example inputs and outputs:

*First load the iris data:*
```
features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
```

*Then call the function*
```
# Take one point:
x = train_features[0, :]
K = 3 # number of classes
M = 10
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
```
*Outputs*:
* `y` : `[0.47439547 0.73158513 0.95244346]`
* `z0`: `[1.  5.  3.4 1.6 0.4]`
* `z1`: `[1.         0.94226108 0.97838733 0.21444943 0.91094513 0.14620877 0.94492407 0.57909676 0.88187859 0.99826648 0.04362534]`
* `a1`: `[ 2.79235088  3.81262569 -1.29831091  2.32522998 -1.76465113  2.84239174 0.31906659  2.01034143  6.3558639  -3.08751164]`
* `a2`: `[-0.1025078   1.00267978  2.99711137]`

### Backward Propagation

The back-propagation algorithm evaluates the gradient of the error function $\Delta E_n(x)$.

`y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)`:
* `x, M, K, W1` and `W2` are the same as for the `ffnn` function
* `target_y` is the target vector. In our case (i.e. for the classification of Iris) this will be a vector with 3 elements, with one element equal to 1.0 and the others equal to 0.0. (*).
* `y` is the output of the output layer (vector with 3 elements)
* `dE1` and `dE2` are the gradient error matrices that contain $\frac{\partial E_n}{\partial w_{ji}}$ for the first and second layers.

Assume sigmoid hidden and output activation functions and assume cross-entropy error function (for classification). Notice that $E_n(\mathbf{w})$ is defined as the error function for a single pattern $\mathbf{x}_n$. 

The inner working of the `backprop` function follows this order of actions:
1. runs `ffnn` on the input.
2. calculates $\delta_k = y_k - target\_y_k$
3. calculates $\delta_j = (\frac{d}{da}\sigma(a1_j)) \sum_{k} w_{k,j+1}\delta_k$ (the `+1` is because of the bias weights)
4. initializes `dE1` and `dE1` as zero-matrices with the same shape as `W1` and `W2`
5. calculates `dE1_{i,j} = \delta_j z0_i` and `dE2_{j,k} = \delta_k z1_j`

Example inputs and outputs:

*Call the function*
```
K = 3  # number of classes
M = 6
D = train_features.shape[1]

x = features[0, :]

# create one-hot target for the feature
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

np.random.seed(42)
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
```
*Output*
* `y`: `[0.42629045 0.1218163  0.56840796]`
* `dE1`:
```
[[-3.17372897e-03  3.13040504e-02 -6.72419861e-03  7.39219402e-02
  -1.16539047e-04  9.29566482e-03]
 [-1.61860177e-02  1.59650657e-01 -3.42934129e-02  3.77001895e-01
  -5.94349138e-04  4.74078906e-02]
 [-1.11080514e-02  1.09564176e-01 -2.35346951e-02  2.58726791e-01
  -4.07886663e-04  3.25348269e-02]
 [-4.44322055e-03  4.38256706e-02 -9.41387805e-03  1.03490716e-01
  -1.63154665e-04  1.30139307e-02]
 [-6.34745793e-04  6.26081008e-03 -1.34483972e-03  1.47843880e-02
  -2.33078093e-05  1.85913296e-03]]
```
* `dE2`:
```
[[-5.73709549e-01  1.21816299e-01  5.68407958e-01]
 [-3.82317044e-02  8.11777445e-03  3.78784091e-02]
 [-5.13977514e-01  1.09133338e-01  5.09227901e-01]
 [-2.11392026e-01  4.48850716e-02  2.09438574e-01]
 [-1.65803375e-01  3.52051896e-02  1.64271203e-01]
 [-3.19254175e-04  6.77875452e-05  3.16303980e-04]
 [-5.60171752e-01  1.18941805e-01  5.54995262e-01]]
```

> (*): *This is referred to as [one-hot encoding](https://en.wikipedia.org/wiki/One-hot). If we have e.g. 3 possible classes: `yellow`, `green` and `blue`, we could assign the following label mapping: `yellow: 0`, `green: 1` and `blue: 2`. Then we could encode these labels as:*
$$
\text{yellow} = \begin{bmatrix}1\\0\\0\end{bmatrix}, \text{green} = \begin{bmatrix}0\\1\\0\end{bmatrix}, \text{blue} = \begin{bmatrix}0\\0\\1\end{bmatrix},
$$
>*But why would we choose to do this instead of just using $0, 1, 2$ ? The reason is simple, using ordinal categorical label injects assumptions into the network that we want to avoid. The network might assume that `yellow: 0` is more different from `blue: 2` than `green: 1` because the difference in the labels is greater. We want our neural networks to output* **probability distributions over classes** *meaning that the output of the network might look something like:*
$$
\text{NN}(x) = \begin{bmatrix}0.32\\0.03\\0.65\end{bmatrix}
$$
> *From this we can directly make a prediction, `0.65` is highest so the model is most confident in that the input feature corresponds to the `blue` label*


## Training the Network
Training consists of:
1. forward propagating an input feature through the network
2. Calculate the error between the prediction the network made and the actual target
3. Back-propagating the error through the network to adjust the weights.


### Train
`W1tr, W2tr, E_total, misclassification_rate, guesses = train_nn(X_train, t_train, M, K, W1, W2, iterations, eta)`:

Inputs:
* `X_train` and `t_train` are the training data and the target values
* `M, K, W1, W2` are defined as above
* `iterations` is the number of iterations the training should take, i.e. how often we should update the weights
* `eta` is the learning rate.

Outputs:
* `W1tr`, `W2tr` are the updated weight matrices
* `E_total` is an array that contains the error after each iteration.
* `misclassification_rate` is an array that contains the misclassification rate after each iteration
* `guesses` is the result from the last iteration, i.e. what the network is guessing for the input dataset `X_train`.

The inner working of the `train_nn` function follows this order of actions:

1. Initializes necessary variables
2. Runs a loop for `iterations` iterations.
3. In each iteration we collect the gradient error matrices for each data point. Start by initializing `dE1_total` and `dE2_total` as zero matrices with the same shape as `W1` and `W2` respectively.
4. Runs a loop over all the data points in `X_train`. In each iteration we call backprop to get the gradient error matrices and the output values.
5. Once we have collected the error gradient matrices for all the data points, we adjust the weights in `W1` and `W2`, using `W1 = W1 + eta* dE1_total / N` where `N` is the number of data points in `X_train` (and similarly for `W2`).
6. For the error estimation we'll use the cross-entropy error function.
7. When the outer loop finishes, we return from the function

Example inputs and outputs:

*Call the function*:
```
K = 3  # number of classes
M = 6
D = train_features.shape[1]
np.random.seed(42)
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.3)
```
*Output*
* `W1tr`:
```
[[ 0.38483561 -0.76090605  0.65776134 -0.11972904 -0.68585273 -0.6963981 ]
...
[-1.93490158  5.40360714 -1.49729944  1.59882456  0.18374103 -0.70066333]]
```
* `W2tr`:
```
[[-0.27112102 -1.2313577  -1.94162633]
...
[-0.25543656 -1.25570954 -0.19962407]]
```
* `Etotal`:
```
[2.54201424 2.23668767 2.00342554 1.90063241 1.88144908 1.87174812
...
0.34689913 0.44249507 0.34517052 0.44083018 0.34346007 0.43918565
0.34176746 0.43756094]
```
* `misclassification_rate`:
```
[0.7  0.7  0.7  0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55
...
0.   0.1  0.   0.1  0.   0.1  0.   0.1  0.   0.1 ]
```
* `last_guesses`:
```
[0. 2. 1. 1. 1. 2. 0. 2. 2. 1. 2. 2. 0. 1. 1. 2. 0. 2. 1. 0.]
```

### Guess
`guesses = test_nn(X_test, M, K, W1, W2)`:

* `X_test`: the dataset that we want to test.
* `M`: size of the hidden layer.
* `K`: size of the output layer.
* `W1, W2`: fully trained weight matrices, i.e. the results from using the `train_nn` function.
* `guesses`: the classification for all the data points in the test set `Xtest`. This should be a $(1\times N)$ vector where $N$ is the number of data points in `X_test`.

The function runs through all the data points in `X_test` and uses the `ffnn` function to guess the classification of each point.
