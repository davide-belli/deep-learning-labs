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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# from part3.dataset import TextDataset
from dataset import TextDataset
# from part3.model import TextGenerationModel
from model import TextGenerationModel
import pickle

################################################################################

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    
    config.train_steps = int(config.train_steps)
    
    softmax = torch.nn.Softmax(dim=0)
    
    def to_label(tensor):
        _, tensor = tensor.max(-1)
        return tensor

    def get_accuracy(predictions, targets):
        idx_p = to_label(predictions)
        idx_t = targets
        correct = [(1 if idx_p[i, j] == idx_t[i, j] else 0) for i in range(idx_p.shape[0]) for j in range(idx_p.shape[1])]
        accuracy = sum(correct) / len(correct)
        return accuracy
    
    def to_one_hot(values):
        values = values.view(values.shape[0], -1)
        tensor = torch.zeros(values.shape[0], values.shape[1], dataset.vocab_size)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                tensor.data[i, j, values[i, j]] = 1
                
        return tensor
    
    def apply_temperature(tensor):
        tensor = softmax(tensor)
        tensor = tensor**(1./config.temp)
        tot = tensor.sum().item()
        return tensor/tot
    
    def sample(tensor):
        tot = tensor.sum().item()
        x = torch.rand(1).item() * tot
        tot = 0
        for i in range(tensor.shape[0]):
            tot += tensor[i]
            if x < tot:
                return i-1
        return tensor.shape[0]

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    print("Vocab size", dataset.vocab_size)
    
    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                config.lstm_num_hidden, config.lstm_num_layers, config.device)  # fixme
    model.to(device)
    
    # Setup the loss and optimizer

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)
    

    def iterate():
        
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
    
            # Only for time measurement of step through network
            t1 = time.time()
            #######################################################
            # Add more code here ...
            #######################################################
    
            model.train()
            optimizer.zero_grad()
    
            batch_targets = torch.stack(batch_targets).t().unsqueeze(-1).to(device)
            # batch_onehot = to_one_hot(torch.stack(batch_inputs).t()).type(torch.FloatTensor).to(device)
            batch_inputs = torch.stack(batch_inputs).t().unsqueeze(-1).type(torch.FloatTensor).to(device)
            
            batch_outputs = model(batch_inputs)
    
            loss = criterion(batch_outputs.transpose(1, 2), batch_targets.squeeze(-1))
            accuracy = get_accuracy(batch_outputs, batch_targets)
    
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
    
            optimizer.step()
    
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
    
            if step % config.print_every == 0:
                
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))
    
            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                model.eval()
                with torch.no_grad():
                    c = torch.randint(0, dataset.vocab_size, (1, 1, 1)).type(torch.FloatTensor).to(device)
                    sentence = [int(c[0, 0, 0])]
                    for i in range(config.seq_length):
                        out = model(c).reshape(-1)
                        if config.temp == 0:
                            c = int(to_label(out))
                        else:
                            prob = apply_temperature(out)
                            c = sample(prob)
                        sentence.append(c)
                        c = torch.FloatTensor([c]).reshape(1, 1, 1).to(device)
                print(dataset.convert_to_string(sentence))
    
            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655

                print('Done training.')
                break
                
            if step and step % 10000 == 0:
                with open("model.pickle", 'wb') as pickle_file:
                    pickle.dump(model, pickle_file)

        
    epoch = 0
    try:
        while True:
            epoch += 1
            iterate()
            print("Epoch {} completed".format(epoch))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early during epoch {}'.format(epoch))
        



 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='./datasets/shakespeare.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-1, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temp', type=float, default="0", help="Temperature for sampling")
    
    config = parser.parse_args()

    # Train the model
    train(config)
