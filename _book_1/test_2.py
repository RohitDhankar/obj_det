
import numpy as np

class FullyConnectedLayer(object):
    """A simple fully-connected NN layer.
        Args:
        num_inputs (int): The input vector size/number of input values. --- 
        layer_size (int): The output vector size/number of neurons.
        activation_function (callable): The activation function for this layer.
        derivated_activation_function
        
        Attributes:
        W (ndarray): The weight values for each input.
        b (ndarray): The bias value, added to the weighted sum.
        size (int): The layer size/number of neurons.
        activation_function (callable): The neurons' activation function.
    """
    def __init__(self, num_inputs, layer_size, activation_function):  #, derivated_activation_function=None):
        
        super().__init__()
        # Randomly initializing the parameters (using a normal distribution this time):
        weights_std_nrm = np.random.standard_normal((num_inputs, layer_size))
        print("------weights_std_nrm.shape---",weights_std_nrm.shape) #(2, 3)
        print("------weights_std_nrm---",weights_std_nrm) #[[-0.23415337 -0.23413696  1.57921282]]

        self.weights = np.random.standard_normal((num_inputs, layer_size)) ##Draw samples from a standard Normal distribution (mean=0, stdev=1).
        self.bias = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_function = activation_function
        
        # self.derivated_activation_function = derivated_activation_function
        # self.x_input, self.y = None, None
        # self.dL_dW, self.dL_db = None, None

    def forward(self, x_input):
        """
        Forward the input vector through the layer, returning its activation vector.
        Args:
            x_input (ndarray): The input vector, of shape `(batch_size, num_inputs)`
        Returns:
            activation (ndarray): The activation value, of shape `(batch_size, layer_size)`.
        """
        z = np.dot(x_input, self.weights) + self.bias
        print("-----forward----z----",z)
        #print("-----forward---activation_function(z)--==self.y--",self.activation_function(z))
        self.y = self.activation_function(z)
        self.x_input = x_input  # (we store the input and output values for back-propagation)
        return self.activation_function(z)

    # def backward(self, dL_dy):
    #     """
    #     Back-propagate the loss, computing all the derivatives, storing those w.r.t. the layer parameters,
    #     and returning the loss w.r.t. its inputs for further propagation.
    #     Args:
    #         dL_dy (ndarray): The loss derivative w.r.t. the layer's output (dL/dy = l'_{k+1}).
    #     Returns:
    #         dL_dx (ndarray): The loss derivative w.r.t. the layer's input (dL/dx).
    #     """
    #     dy_dz = self.derivated_activation_function(self.y)  # = f'
    #     dL_dz = (dL_dy * dy_dz) # dL/dz = dL/dy * dy/dz = l'_{k+1} * f'
    #     dz_dw = self.x.T
    #     dz_dx = self.W.T
    #     dz_db = np.ones(dL_dy.shape[0]) # dz/db = d(W.x + b)/db = 0 + db/db = "ones"-vector

    #     # Computing the derivatives with respect to the layer's parameters, and storing them for opt. optimization:
    #     self.dL_dW = np.dot(dz_dw, dL_dz)
    #     self.dL_db = np.dot(dz_db, dL_dz)

    #     # Computing the derivative with respect to the input, to be passed to the previous layers (their `dL_dy`):
    #     dL_dx = np.dot(dL_dz, dz_dx)
    #     return dL_dx

    # def optimize(self, epsilon):
    #     """
    #     Optimize the layer's parameters, using the stored derivative values.
    #     Args:
    #         epsilon (float): The learning rate.
    #     """
    #     self.W -= epsilon * self.dL_dW
    #     self.b -= epsilon * self.dL_db


#==============================================================================
# Main Call
#==============================================================================

if __name__ == "__main__":
    np.random.seed(42) # Fixing the seed for the random number generation, to get reproducable results.

    ## Draw samples from a uniform distribution.
    x1 = np.random.uniform(-1, 1, 2).reshape(1, 2)  # BOOK VALUES > [[-0.25091976  0.90142861]]
    print("----x1------",x1)  ## [[-0.25091976  0.90142861]]   ###  random.uniform(low=0.0, high=1.0, size=None)
    x2 = np.random.uniform(-1, 1, 2).reshape(1, 2)  ## BOOK VALUES  > [[0.46398788 0.19731697]] # Random input column-vectors of 2 values (shape = `(1, 2)`)
    print("----x2------",x2) ##[[0.46398788 0.19731697]]

    #RELU>>Activation-Func #
    relu_function = lambda y: np.maximum(y, 0)  ## np.maximum >> The arrays holding the elements to be compared. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).
    layer = FullyConnectedLayer(num_inputs=2, layer_size=3,activation_function=relu_function, derivated_activation_function=None)
    
    # Our layer can process x1 and x2 separately...
    out1 = layer.forward(x1) ## BOOK VALUES  > > [[0.28712364 0.         0.33478571]]
    print("----out1------",out1)
    out2 = layer.forward(x2) ## BOOK VALUES  > > [[0.         0.         1.08175419]]
    print("----out2------",out2)

    # ... or together:
    x12 = np.concatenate((x1, x2))  # stack of input vectors, of shape `(2, 2)`
    print("--np.concatenate->>-x12------",x12)
    out12 = layer.forward(x12) ## BOOK VALUES  > > [[0.28712364 0.         0.33478571] ----AND--->>  [0.         0.         1.08175419]]
    print("----out12------",out12)
    
    



