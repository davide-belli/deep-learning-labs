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

from part3.dataset import TextDataset
from part3.model import TextGenerationModel

# from dataset import TextDataset
# from model import TextGenerationModel

import os
import pickle

################################################################################

def train(config):
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("dreams"):
        os.makedirs("dreams")

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)
    
    config.train_steps = int(config.train_steps)

    if config.file_name is not None:
        exp_name = '{}_{}'.format(config.file_name, datetime.now().strftime("%Y-%m-%d %H:%M"))
    else:
        exp_name = '{}_{}'.format(config.txt_file.split("/")[-1], datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(exp_name)
    print(config)
    
    def to_label(tensor):
        _, tensor = tensor.max(-1)
        return tensor

    def get_accuracy(predictions, targets):
        idx_p = to_label(predictions)
        idx_t = targets
        correct = [(1 if idx_p[i, j] == idx_t[i, j] else 0) for i in range(idx_p.shape[0]) for j in range(idx_p.shape[1])]
        accuracy = sum(correct) / len(correct)
        return accuracy
    
    softmax = torch.nn.Softmax(dim=0)
    
    def apply_temperature(tensor, t=config.temp):
        tensor = softmax(tensor/t)
        return tensor
    
    def sample(tensor):
        result = torch.multinomial(tensor, 1).item()
        return result
        # Manual sampling
        # tot = tensor.sum().item()
        # x = torch.rand(1).item() * tot
        # tot = 0
        # for i in range(tensor.shape[0]):
        #     tot += tensor[i]
        #     if x < tot:
        #         return i-1
        # return tensor.shape[0]
    
    def generate_text(c, length, sentence, temp=0):
        for i in range(length):
            if config.input_dim > 1:
                c = torch.zeros(1, 1, dataset.vocab_size).to(device).scatter_(2, c, 1)
            out = model(c).reshape(-1)
            if temp == 0:
                c = int(to_label(out))
            else:
                prob = apply_temperature(out, temp)
                c = sample(prob)
            sentence.append(c)
            c = torch.LongTensor([c]).reshape(1, 1, 1).to(device)
        generated_string = dataset.convert_to_string(sentence)
        
        return generated_string

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    print("Vocab size", dataset.vocab_size)
    
    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                config.lstm_num_hidden, config.lstm_num_layers, config.device, input_size=dataset.vocab_size)  # fixme
    model.to(device)
    
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)
    

    def run_epoch(total_steps):
        
        step = total_steps
        
        for batch_inputs, batch_targets in data_loader:
            step += 1
    
            # Only for time measurement of step through network
            t1 = time.time()
            #######################################################
            # Add more code here ...
            #######################################################
    
            model.train()
            optimizer.zero_grad()
    
            batch_targets = torch.stack(batch_targets).t().unsqueeze(-1).to(device)
            
            # Shouldn't use dimension = 1 to represent the sequence input!
            if config.input_dim == 1:
                batch_inputs = torch.stack(batch_inputs).t().unsqueeze(-1).type(torch.FloatTensor).to(device)
            else:
                temp = torch.stack(batch_inputs).t().unsqueeze(-1).type(torch.LongTensor).to(device)
                batch_inputs = torch.zeros(temp.shape[0], temp.shape[1], dataset.vocab_size).to(device)
                batch_inputs.scatter_(2, temp, 1)
            
            batch_outputs = model(batch_inputs)
    
            loss = criterion(batch_outputs.transpose(0, 1).transpose(1, 2), batch_targets.squeeze(-1))
            accuracy = get_accuracy(batch_outputs.transpose(0, 1), batch_targets)
    
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
                for t in [0, 0.5, 1, 2]:
                    with torch.no_grad():
                        
                        # "Dream" new text from a starting sentence
                        if config.dream:
                            start_string = "smth"
                            if config.dream_string is not None:
                                start_string = config.dream_string
                            sentence = dataset.convert_to_idx(start_string)
                            temp = torch.tensor(sentence[:-1]).reshape(1, -1, 1).type(torch.LongTensor).to(device)
                            batch_inputs = torch.zeros(temp.shape[0], temp.shape[1], dataset.vocab_size).to(device).scatter_(2, temp, 1)
                            batch_inputs.scatter_(2, temp, 1)
                            _ = model(batch_inputs)
                            
                            c = torch.tensor(sentence[-1]).view(1, 1, 1).type(torch.LongTensor).to(device)
                            generated_string = generate_text(c, config.dream_length, sentence, t)
                            # print("############# DREAM #############\n" + generated_string)
            
                            with open("dreams/dream_t{}_{}".format(t, exp_name), "a") as f:
                                f.write("{} | {}\n".format(step, generated_string))
                            model.reset_hidden()  # Reset hidden layer after finishing to generate a sentence
                            
                        # Generate 30 chars long sequence
                        c = torch.randint(0, dataset.vocab_size, (1, 1, 1)).type(torch.LongTensor).to(device)
                        sentence = [int(c[0, 0, 0])]
                        generated_string =  generate_text(c, config.seq_length, sentence, t)
                    print(generated_string)

                    with open("logs/logs_t{}_{}".format(t, exp_name), "a") as f:
                        f.write("{} | {}\n".format(step, generated_string))
                    model.reset_hidden()  # Reset hidden layer after finishing to generate a sentence
    
            # Save Model every 10k iterations
            if step and step % 10000 == 0:
                with open("models/model_{}.pickle".format(exp_name), 'wb') as pickle_file:
                    pickle.dump(model, pickle_file)
                    
            # End of the training
            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                print('Done training.')
                break
                
                    
        return step

    epoch = 0
    try:
        total_steps = 0
        while True:
            epoch += 1
            total_steps += run_epoch(total_steps)
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
    parser.add_argument('--txt_file', type=str, default='./assets/shakespeare.txt',
                        help="Path to a .txt file to train on")
    parser.add_argument('--file_name', type=str, default=None,
                        help="name of the training corpus")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

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
    parser.add_argument('--dream', type=bool, default="True", help="Dream new text from sentence")
    parser.add_argument('--dream_length', type=int, default="200", help="Length of dreamt sentence")
    parser.add_argument('--dream_string', type=str, default="He did not really like ", help="Start for the dreamt sentence")
    
    config = parser.parse_args()
    if config.file_name is not None:
        config.txt_file = "./assets/" + config.file_name

    # Train the model
    train(config)
