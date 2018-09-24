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
        
        self.b_h = torch.nn.Parameter(torch.Tensor(1, self.num_hidden).zero_())
        self.W_hh = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=0, std=0.0001))
        self.W_hx = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.input_dim).normal_(mean=0, std=0.0001))
        
        self.b_p = torch.nn.Parameter(torch.Tensor(1, self.num_classes).zero_())
        self.W_ph = torch.nn.Parameter(torch.Tensor(self.num_classes, self.num_hidden).normal_(mean=0, std=0.0001))
        
        self.tanh = torch.nn.Tanh()
        # self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        # Implementation here ...
        
        p = None
            
        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device)

        # for i in range(self.seq_len):
        #     a1 = self.W_hx
        #     a2 = x[:, i].transpose(0, 1)  # Deals with input of any dimensionality
        #     a = a1 @ a2
        #     b1 = self.W_hh
        #     b2 = h
        #     b = b1 @ b2
        #     c = self.b_h
        #     h_hat = a + b + c
        #     h = self.tanh(h_hat)
        #
        # r = self.W_ph @ h
        # s = self.b_p
        # # p = self.softmax(r + s)
        # p = r + s

        for i in range(self.seq_len):
            h_hat = x[:, i].view(self.batch_size, -1) @ self.W_hx.t() + h @ self.W_hh.t() + self.b_h
            h = self.tanh(h_hat)
            p = h @ self.W_ph.t() + self.b_p
        
        return p
        
        pass
