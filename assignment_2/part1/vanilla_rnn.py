################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
    
        self.seq_len = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        
        self.b_h = torch.nn.Parameter(torch.Tensor(self.num_hidden, 1).zero_().to(self.device))
        self.W_hh = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=0, std=0.0001)).to(self.device))
        self.W_hx = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.input_dim).normal_(mean=0, std=0.0001).to(self.device))
        
        self.b_p = torch.nn.Parameter(torch.Tensor(self.num_classes, 1).zero_().to(self.device))
        self.W_ph = torch.nn.Parameter(torch.Tensor(self.num_classes, self.num_hidden).normal_(mean=0, std=0.0001).to(self.device))
        
        self.tanh = torch.nn.Tanh()
        # self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        # Implementation here ...
        
        p = None
            
        h = torch.zeros(self.num_hidden, self.batch_size).to(self.device)

        for i in range(self.seq_len):
            a1 = self.W_hx
            a2 = x[:, i].reshape(self.input_dim, -1)  # Deals with input of any dimensionality
            a = a1 @ a2
            b1 = self.W_hh
            b2 = h
            b = b1 @ b2
            c = self.b_h
            h_hat = a + b + c
            h = self.tanh(h_hat)
            
        r = self.W_ph @ h
        s = self.b_p
        # p = self.softmax(r + s)
        p = r + s
        
        return p.reshape(self.batch_size, -1)
        
        pass
