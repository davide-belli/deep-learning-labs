"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Best model, accuracy: 0.5833
# DNN_HIDDEN_UNITS_DEFAULT = '1000, 800, 600, 400'
# LEARNING_RATE_DEFAULT = 1e-4
# MAX_STEPS_DEFAULT = 20000
# BATCH_SIZE_DEFAULT = 3000
# EVAL_FREQ_DEFAULT = 200


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    _, idx_p = predictions.max(1)
    _, idx_t = targets.max(1)
    correct = [(1 if idx_p[i] == idx_t[i] else 0) for i in range(len(idx_p))]
    accuracy = sum(correct) / len(correct)
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():
    """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    
    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    ############################## VARIABLES ##############################
    
    SAVE_PLOTS = False
    SAVE_LOGS = False
    
    img_size = 32
    n_classes = 10
    input_size = img_size * img_size * 3
    batch_size = FLAGS.batch_size
    eval_freq = FLAGS.eval_freq
    n_iterations = FLAGS.max_steps
    lr_rate = FLAGS.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ############################## METHODS ##############################
    
    # fp = open('memory_profiler_basic_mean.log', 'w+')
    # @profile(stream=fp)
    def test():
        net.eval()
        
        output_t = net(x_t)
        loss_t = criterion(output_t, y_t).detach()
        acc_t = accuracy(output_t.detach(), y_t_onehot)
        
        return acc_t, loss_t
    
    def plot(iteration):
        idx_test = list(range(0, iteration + 1, eval_freq))
        idx = list(range(0, iteration + 1))
        
        plt.clf()
        plt.cla()
        plt.subplot(1, 2, 1)
        plt.plot(idx_test, test_accuracies, "k-", linewidth=1, label="test")
        plt.plot(idx, accuracies, "r-", linewidth=0.5, alpha=0.5, label="train")
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(idx_test, test_losses, "k-", linewidth=1, label="test")
        plt.plot(idx, losses, "r-", linewidth=0.5, alpha=0.5, label="train")
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./out/plot/plot_pytorch_" + str(batch_size) + "_" + str(lr_rate) + ".png", bbox_inches='tight')
        return
    
    def to_label(tensor):
        _, tensor = tensor.max(1)
        return tensor

    ############################## MAIN ##############################
    
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    
    net = MLP(input_size, dnn_hidden_units, n_classes)
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # optimizer = optim.SGD(net.parameters(), lr=lr_rate, momentum=0.8, nesterov=False)
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)
    
    losses = []
    accuracies = []
    test_accuracies = []
    test_losses = []
    alpha = 0.0001

    x_t = cifar10['test'].images
    y_t = cifar10['test'].labels
    x_t = torch.from_numpy(x_t.reshape(-1, input_size))
    y_t_onehot = torch.from_numpy(y_t).type(torch.LongTensor)
    y_t = to_label(y_t_onehot)
    x_t, y_t = x_t.to(device), y_t.to(device)
    y_t_onehot = y_t_onehot.to(device)

    plt.figure(figsize=(10, 4))
    
    for i in range(n_iterations):
        
        x, y = cifar10['train'].next_batch(batch_size)
        x = torch.from_numpy(x.reshape(-1, input_size))
        y_onehot = torch.from_numpy(y).type(torch.LongTensor)
        y = to_label(y_onehot)
        x, y = x.to(device), y.to(device)
        y_onehot = y_onehot.to(device)
        
        optimizer.zero_grad()
        output = net(x)
        train_loss = criterion(output, y)

        reg_loss = 0
        for param in net.parameters():
            reg_loss += param.norm(2)
            
        loss = train_loss + alpha * reg_loss
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        accuracies.append(accuracy(output.detach().data, y_onehot.detach()))
        
        del x, y
        
        if i % eval_freq == 0:
            acc_t, loss_t = test()
            test_accuracies.append(acc_t)
            test_losses.append(loss_t)
            
            log_string = "[{:5d}/{:5d}] Test Accuracy: {:.4f} | Batch Accuracy: {:.4f} | Batch Loss: {:.6f} | Train/Reg: {:.6f}/{:.6f}\n".format(
                i, n_iterations, test_accuracies[-1], accuracies[-1], loss, train_loss, reg_loss * alpha
            )
            print(log_string)
            
            if SAVE_LOGS:
                with open("./out/log/pytorch_log_" + str(batch_size) + "_" + str(lr_rate) + ".txt", "a") as myfile:
                    myfile.write(log_string)

            if SAVE_PLOTS:
                plot(i)

            net.train()
            
            
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
  Main function
  """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
