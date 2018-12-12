# LM
A Python module to create simple multi-layer perceptron neural networks using Levenberg-Marquardt training

## Prerequisites
This package uses Python 3.x
This package requires numpy which can be downloaded using the following commands
```
pip install numpy
```

## Installing
To install and use this package simply run:
```
pip install --index-url https://test.pypi.org/simple/ LM
```
Then you can simply import it using:
```
import LM
```

## Example
- Creating a new neural network
```
structure = [1, 5, 1]
nn = LM.NN(structure)
```
The list `structure` describes how many layers there are in the network and how many neurons are in each layer sequentially
- Training the neural network
```
nn.train_lm(train_X, train_Y)
```
- Predicting new input
```
nn.predict(test_X)
```