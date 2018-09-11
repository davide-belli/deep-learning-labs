"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# TODO remove
SAVE = True

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
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    img_size = 32
    input_size = img_size * img_size * 3
    batch_size = FLAGS.batch_size
    eval_freq = FLAGS.eval_freq
    n_iterations = FLAGS.max_steps
    lr_rate = FLAGS.learning_rate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    def test():
        net.eval()
    
        output_t = net(x_t)
        loss_t = criterion(output_t, y_t)
        acc_t = accuracy(output_t.detach(), y_t_onehot.detach())
    
        return acc_t, loss_t.item()

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
        plt.savefig("plot_convnet.png", bbox_inches='tight')
        return

    def to_label(tensor):
        _, tensor = tensor.max(1)
        return tensor

    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    net = ConvNet(3, 10)
    net.to(device)
    params = list(net.named_parameters())
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
        x = torch.from_numpy(x)
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
        accuracies.append(accuracy(output.detach(), y_onehot.detach()))
    
        del x, y
    
        if i % eval_freq == 0:
            acc_t, loss_t = test()
            test_accuracies.append(acc_t)
            test_losses.append(loss_t)
            log_string = "[{}/{}] Test Accuracy: {:.4f} | Batch Accuracy: {:.4f} | Batch Loss: {:.6f} | Train/Reg: {:.6f}/{:.6f}\n".format(
                i, n_iterations, test_accuracies[-1], accuracies[-1], loss, train_loss, reg_loss * alpha
            )
            print(log_string)
            plot(i)

            if SAVE:
                with open("pytorch_log.txt", "a") as myfile:
                    myfile.write(log_string)
        
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
