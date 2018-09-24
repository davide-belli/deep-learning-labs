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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        
        self.seq_len = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.b_g = torch.nn.Parameter(torch.Tensor(1, self.num_hidden).zero_())
        self.W_gh = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=0, std=0.0001))
        self.W_gx = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.input_dim).normal_(mean=0, std=0.0001))

        self.b_i = torch.nn.Parameter(torch.Tensor(1, self.num_hidden).zero_())
        self.W_ih = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=0, std=0.0001))
        self.W_ix = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.input_dim).normal_(mean=0, std=0.0001))

        self.b_f = torch.nn.Parameter(torch.Tensor(1, self.num_hidden).zero_())
        self.W_fh = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=0, std=0.0001))
        self.W_fx = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.input_dim).normal_(mean=0, std=0.0001))

        self.b_o = torch.nn.Parameter(torch.Tensor(1, self.num_hidden).zero_())
        self.W_oh = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=0, std=0.0001))
        self.W_ox = torch.nn.Parameter(torch.Tensor(self.num_hidden, self.input_dim).normal_(mean=0, std=0.0001))
        
        self.b_p = torch.nn.Parameter(torch.Tensor(1, self.num_classes).zero_())
        self.W_ph = torch.nn.Parameter(torch.Tensor(self.num_classes, self.num_hidden).normal_(mean=0, std=0.0001))
        
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Implementation here ...

        p = None

        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        c = torch.zeros(self.batch_size, self.num_hidden).to(self.device)


        for j in range(self.seq_len):
            g = self.tanh(x[:, j].view(self.batch_size, -1) @ self.W_gx.t() + h @ self.W_gh.t() + self.b_g)
            i = self.sigmoid(x[:, j].view(self.batch_size, -1) @ self.W_ix.t() + h @ self.W_ih.t() + self.b_i)
            f = self.sigmoid(x[:, j].view(self.batch_size, -1) @ self.W_fx.t() + h @ self.W_fh.t() + self.b_f)
            o = self.sigmoid(x[:, j].view(self.batch_size, -1) @ self.W_ox.t() + h @ self.W_oh.t() + self.b_o)
            
            c = g * i + c * f
            h = self.tanh(c) * o
            
            p = h @ self.W_ph.t() + self.b_p

        return p
        
        pass