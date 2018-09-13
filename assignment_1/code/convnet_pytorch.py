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
    It handles the different conv and parameters of the model.
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
        super(ConvNet, self).__init__()
        
        maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        relu = nn.ReLU()
        self.debug = True
        
        self.conv1 = nn.Sequential()
        self.conv2 = nn.Sequential()
        self.conv3a = nn.Sequential()
        self.conv3b = nn.Sequential()
        self.conv4a = nn.Sequential()
        self.conv4b = nn.Sequential()
        self.conv5a = nn.Sequential()
        self.conv5b = nn.Sequential()
        self.linear = nn.Sequential()
        
        self.conv1.add_module("conv_1", nn.Conv2d(n_channels, 64, 3, padding=1))
        self.conv1.add_module("batchnorm_1", nn.BatchNorm2d(64))
        self.conv1.add_module("relu_1", relu)

        self.conv1.add_module("maxpool_1", maxpool)
        
        self.conv2.add_module("conv_2", nn.Conv2d(64, 128, 3, padding=1))
        self.conv2.add_module("batchnorm_2", nn.BatchNorm2d(128))
        self.conv2.add_module("relu_2", relu)

        self.conv2.add_module("maxpool_2", maxpool)
        
        self.conv3a.add_module("conv_3_a", nn.Conv2d(128, 256, 3, padding=1))
        self.conv3a.add_module("batchnorm_3_a", nn.BatchNorm2d(256))
        self.conv3a.add_module("relu_3_a", relu)
        self.conv3b.add_module("conv_3_b", nn.Conv2d(256, 256, 3, padding=1))
        self.conv3b.add_module("batchnorm_3_b", nn.BatchNorm2d(256))
        self.conv3b.add_module("relu_3_b", relu)

        self.conv3b.add_module("maxpool_3", maxpool)
        
        self.conv4a.add_module("conv_4_a", nn.Conv2d(256, 512, 3, padding=1))
        self.conv4a.add_module("batchnorm_4_a", nn.BatchNorm2d(512))
        self.conv4a.add_module("relu_4_a", relu)
        self.conv4b.add_module("conv_4_b", nn.Conv2d(512, 512, 3, padding=1))
        self.conv4b.add_module("batchnorm_4_b", nn.BatchNorm2d(512))
        self.conv4b.add_module("relu_4_b", relu)

        self.conv4b.add_module("maxpool_4", maxpool)
        
        self.conv5a.add_module("conv_5_a", nn.Conv2d(512, 512, 3, padding=1))
        self.conv5a.add_module("batchnorm_5_a", nn.BatchNorm2d(512))
        self.conv5a.add_module("relu_5_a", relu)
        self.conv5b.add_module("conv_5_b", nn.Conv2d(512, 512, 3, padding=1))
        self.conv5b.add_module("batchnorm_5_b", nn.BatchNorm2d(512))
        self.conv5b.add_module("relu_5_b", relu)
        
        self.conv5b.add_module("maxpool_5", maxpool)
        
        self.conv5b.add_module("avgpool", nn.AvgPool2d(1))
        
        self.linear.add_module("linear", nn.Linear(512, n_classes))
        # self.linear.add_module("softmax", nn.Softmax(dim=1))

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
        if self.debug:
            print("Got", x.data.shape)
            out = self.conv1(x)
            print("Got", out.data.shape)
            out = self.conv2(out)
            print("Got", out.data.shape)
            out = self.conv3a(out)
            print("Got", out.data.shape)
            out = self.conv3b(out)
            print("Got", out.data.shape)
            out = self.conv4a(out)
            print("Got", out.data.shape)
            out = self.conv4b(out)
            print("Got", out.data.shape)
            out = self.conv5a(out)
            print("Got", out.data.shape)
            out = self.conv5b(out)
            print("Got", out.data.shape)
            print("myshape", out.data.shape)
            out = self.linear(out.view(out.data.shape[0], -1))
            print("myshape", out.data.shape)
            self.debug = False
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3a(out)
            out = self.conv3b(out)
            out = self.conv4a(out)
            out = self.conv4b(out)
            out = self.conv5a(out)
            out = self.conv5b(out)
            out = self.linear(out.view(out.data.shape[0], -1))
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
