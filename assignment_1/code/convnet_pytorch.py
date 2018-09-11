"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        batchnorm = nn.BatchNorm1d()
        relu = nn.ReLU()
        
        self.layers = nn.Sequential()
        
        self.layers.add_module("conv_1", nn.Conv2d(n_channels, 64, 3))
        self.layers.add_module("batchnorm_1", batchnorm)
        self.layers.add_module("relu_1", relu)

        self.layers.add_module("maxpool_1", maxpool)
        
        self.layers.add_module("conv_2", nn.Conv2d(64, 128, 3))
        self.layers.add_module("batchnorm_2", batchnorm)
        self.layers.add_module("relu_2", relu)

        self.layers.add_module("maxpool_2", maxpool)
        
        self.layers.add_module("conv_3_a", nn.Conv2d(128, 256, 3))
        self.layers.add_module("batchnorm_3_a", batchnorm)
        self.layers.add_module("relu_3_a", relu)
        self.layers.add_module("conv_3_b", nn.Conv2d(256, 256, 3))
        self.layers.add_module("batchnorm_3_b", batchnorm)
        self.layers.add_module("relu_3_b", relu)

        self.layers.add_module("maxpool_3", maxpool)
        
        self.layers.add_module("conv_4_a", nn.Conv2d(256, 512, 3))
        self.layers.add_module("batchnorm_4_a", batchnorm)
        self.layers.add_module("relu_4_a", relu)
        self.layers.add_module("conv_4_b", nn.Conv2d(512, 512, 3))
        self.layers.add_module("batchnorm_4_b", batchnorm)
        self.layers.add_module("relu_4_b", relu)

        self.layers.add_module("maxpool_4", maxpool)
        
        self.layers.add_module("conv_5_a", nn.Conv2d(512, 512, 3))
        self.layers.add_module("batchnorm_5_a", batchnorm)
        self.layers.add_module("relu_5_a", relu)
        self.layers.add_module("conv_5_b", nn.Conv2d(512, 512, 3))
        self.layers.add_module("batchnorm_5_b", batchnorm)
        self.layers.add_module("relu_5_b", relu)
        
        self.layers.add_module("maxpool_5", maxpool)
        
        self.layers.add_module("avgpool", nn.AvgPool2d())
        
        self.layers.add_module("linear", nn.Linear(512, n_classes))
        self.layers.add_module("softmax", nn.Softmax(dim=1))

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
        out = self.sequential(x)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
