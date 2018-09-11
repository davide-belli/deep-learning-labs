"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()
        
        
        self.n_inputs = n_inputs
        self.n_outputs = n_classes
        self.n_layers = len(n_hidden)
        self.size_layers = [n_inputs] + n_hidden
        
        self.layers = nn.Sequential()
        for i in range(self.n_layers):
            linear = nn.Linear(self.size_layers[i], self.size_layers[i + 1])
            # torch.nn.init.normal_(linear.weight, 0, 0.01)
            # linear.bias.data.fill_(0.)
            
            self.layers.add_module("Linear_" + str(i), linear)
            # self.layers.add_module("BatchNorm_" + str(i), nn.BatchNorm1d(self.size_layers[i+1]))
            self.layers.add_module("ReLU_" + str(i), nn.ReLU())
            self.layers.add_module("Dropout_" + str(i), nn.Dropout(p=0.05))
        self.layers.add_module("Linear_" + str(i+1), nn.Linear(self.size_layers[-1], self.n_outputs))
        self.layers.add_module("Softmax", nn.Softmax(dim=1))
        
        self.sequential = nn.Sequential(*self.layers)
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # for layer in self.layers:
        #     x = layer(x)
        # out = x
        out = self.sequential(x)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
