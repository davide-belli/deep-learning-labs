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

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# from dataset import PalindromeDataset
# from vanilla_rnn import VanillaRNN
# from lstm import LSTM

# You may want to look into tensorboardX for logging
from tensorboardX import SummaryWriter


################################################################################

def train(config):
    assert config.model_type in ('RNN', 'LSTM')
    
    exp_name = 'runs/{}_batch{}_dim{}_len{}_{}'.format(config.model_type, config.batch_size,
                                             config.input_dim, config.input_length, datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(exp_name)
    print(config)
    
    writer = SummaryWriter(exp_name)
    
    # Convert to labels
    def to_label(tensor):
        _, tensor = tensor.max(-1)
        return tensor
    
    # Output accuracy given predictions and targets
    def get_accuracy(predictions, targets):
        idx_p = to_label(predictions)
        idx_t = targets
        correct = (idx_p == idx_t).type(torch.FloatTensor) # TODO test accuracy
        accuracy = (correct.sum() / correct.shape[0]).item()
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
        
        batch_targets = batch_targets.to(device)
        
        if config.input_dim == 1:
            batch_inputs = batch_inputs.unsqueeze(-1).to(device)
        elif config.input_dim == 10:
            temp = batch_inputs.type(torch.LongTensor).to(device)
            batch_inputs = torch.zeros(config.batch_size, config.input_length, config.input_dim).to(device)
            batch_inputs.scatter_(2, temp.unsqueeze(-1), 1)

        batch_outputs = model(batch_inputs)
        
        loss = criterion(batch_outputs, batch_targets)
        accuracy = get_accuracy(batch_outputs, batch_targets)
        
        loss.backward()
        
        ############################################################################
        # QUESTION: what happens here and why?
        # ANSWER: The gradients are clipped up to a certain threshold value (of the vector norm) to avoid exploding gradient problem
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        
        optimizer.step()
        
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)
        
        # Print and save data to Tensorboard
        if step % 10 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))
            writer.add_scalar('Accuracy', accuracy, step)
            writer.add_scalar('Loss', loss.item(), step)
        
        # Output final scores
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
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
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
