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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

# from part1.dataset import PalindromeDataset
# from part1.vanilla_rnn import VanillaRNN
# from part1.lstm import LSTM

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

from tensorboardX import SummaryWriter


################################################################################

def train(config):
    assert config.model_type in ('RNN', 'LSTM')
    
    exp_name = 'runs/{}_batch{}_dim{}_len{}_{}'.format(config.model_type, config.batch_size,
                                             config.input_dim, config.input_length, datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(exp_name)
    print(config)
    
    writer = SummaryWriter(exp_name)
    
    def to_label(tensor):
        _, tensor = tensor.max(-1)
        return tensor
    
    def to_one_hot(values):
        values = values.view(values.shape[0], -1)
        tensor = torch.zeros(values.shape[0], values.shape[1], config.num_classes)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                tensor.data[i, j, values[i, j]] = 1

        # tensor = torch.zeros(values.shape[0], values.shape[1], config.num_classes, dtype=torch.float)
        # tensor[values.type(torch.LongTensor)] = 1
        return tensor
    
    def get_accuracy(predictions, targets):
        idx_p = to_label(predictions)
        idx_t = targets
        correct = [(1 if idx_p[i] == idx_t[i] else 0) for i in range(len(idx_p))]
        accuracy = sum(correct) / len(correct)
        return accuracy
    
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)
    
    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden,
                           config.num_classes, config.batch_size, device)
        model.to(device)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim, config.num_hidden,
                           config.num_classes, config.batch_size, device)
        model.to(device)
    else:
        raise ValueError
        
    
    # print(list(model.named_parameters()))
    
    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        
        # Only for time measurement of step through network
        t1 = time.time()
        
        # Add more code here ...
        model.train()
        optimizer.zero_grad()
        
        # Add more code here ...
        batch_targets = batch_targets.to(device)
        
        if config.input_dim == 1:
            batch_inputs = batch_inputs.unsqueeze(-1).to(device)
            batch_outputs = model(batch_inputs)
        elif config.input_dim == 10:
            batch_inputs_long = batch_inputs.type(torch.LongTensor)
            batch_inputs_onehot = to_one_hot(batch_inputs_long).type(torch.FloatTensor).to(device)
            batch_outputs = model(batch_inputs_onehot)
        
        loss = criterion(batch_outputs, batch_targets)
        accuracy = get_accuracy(batch_outputs, batch_targets)
        
        loss.backward()
        
        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        # The gradients are clipped to avoid exploding gradient problem
        
        optimizer.step()
        
        # with torch.no_grad():
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)
        
        if step % 10 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))
            writer.add_scalar('Accuracy', accuracy, step)
            writer.add_scalar('Loss', loss.item(), step)
        
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            with open("logs.txt", "a") as f:
                f.write("{}   Accuracy: {}\n".format(exp_name, accuracy))
                
            break
    
    print('Done training.')


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()
    
    # Model params
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    
    config = parser.parse_args()
    
    # Train the model
    train(config)
