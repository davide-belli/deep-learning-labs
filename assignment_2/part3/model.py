# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.lstm = nn.LSTM(1, lstm_num_hidden, num_layers=lstm_num_layers, dropout=0.3)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.h_0 = torch.zeros(lstm_num_layers, seq_length, lstm_num_hidden).to(device)
        self.c_0 = torch.zeros(lstm_num_layers, seq_length, lstm_num_hidden).to(device)
        self.eval_h_0 = torch.zeros(lstm_num_layers, 1, lstm_num_hidden).to(device)
        self.eval_c_0 = torch.zeros(lstm_num_layers, 1, lstm_num_hidden).to(device)
        self.eval_h = None
        self.eval_c = None

    def forward(self, x, train=True):
        
        # Handle hidden layer for evaluation
        if train:
            # print("train")
            h, c = self.h_0, self.c_0
            self.eval_c = None
        else:
            # print("eval")
            if self.eval_c is None:
                h, c = self.eval_h_0, self.eval_c_0
            else:
                h, c = self.eval_h, self.eval_c
                
        # Implementation here...
        out, (h, c) = self.lstm(x, (h, c))
        out = self.linear(out)
        
        if not train:
            self.eval_h, self.eval_c = h, c
        
        
        return out
