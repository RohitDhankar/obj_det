import numpy as np 

class Neuron(object):
    """A simple feed-forward artificial neuron.
    Args:
    num_inputs (int): The input vector size / number of input values.
    activation_function (callable): The activation function.
    Attributes:
    W (ndarray): The weight values for each input.
    b (float): The bias value, added to the weighted sum.
    activation_function (callable): The activation function.
    """

    def __init__(self, num_inputs, activation_function):
        super().__init__()
        # Randomly initializing the weight vector and bias value:
        print("----self.weights------",np.random.rand(num_inputs))
        ## Own Output ----self.weights------ [0.59865848 0.15601864 0.15599452]
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function

    def forward(self, x_input):
        """Forward the input signal through the neuron."""
        z = np.dot(x_input, self.weights) + self.bias
        print("-----forward----z----",z)
        return self.activation_function(z)


# Fixing the random number generator's seed, for reproducible results:
np.random.seed(42) # because of SEED - 3 Random Nubers below -x_input-- are Randomly always the 3 same values 
# Random input column array of 3 values (shape = `(1, 3)`)
x_input = np.random.rand(3).reshape(1, 3)
print("--x_input--\n",x_input) # [[0.37454012 0.95071431 0.73199394]]
print("--x_input.size----\n",x_input.size) # 
# BOOK VALUES > [[0.37454012 0.95071431 0.73199394]]

# Instantiating a Perceptron (simple neuron with step function):
step_fn = lambda y: 0 if y <= 0 else 1

perceptron = Neuron(num_inputs=x_input.size, activation_function=step_fn) 
# perceptron --- an Object of the Class Neuron
print("--perceptron.weights--\n",perceptron.weights) # [0.05808361 0.86617615 0.60111501]
print("--perceptron.bias--\n",perceptron.bias) # [0.70807258]
outPut = perceptron.forward(x_input)
print("----perceptron.forward(x_input)--->>-outPut-\n",outPut)

### BOOK VALUES 
# # > perceptron.weights
# = [0.59865848 0.15601864 0.15599452]
# # > perceptron.bias
# = [0.05808361]
# out = perceptron.forward(x)
# # > 1