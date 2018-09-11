"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)),
                   'bias': np.zeros(out_features)}
    self.grads = {'weight': np.zeros((out_features, in_features)),
                  'bias': np.zeros(out_features)}
    
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    out = x @ self.params['weight'].T + self.params['bias'].reshape(1, -1)
    
    self.x = x
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dim_out, dim_in = self.params['weight'].shape
    dim_batches = dout.shape[0]
    
    self.grads['bias'] = np.sum(dout, axis=0)
    batch_gradients = dout.reshape(dim_batches, dim_out, 1) @ self.x.reshape(dim_batches, 1, dim_in)
    self.grads['weight'] = np.sum(batch_gradients, axis=0)
    dx = dout @ self.params['weight']
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x.clip(min=0)
    
    self.x = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout * (self.x > 0)
    
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    b = x.max(axis=1).reshape(-1, 1)  # normalization term
    x = np.exp(x - b)
    sums = np.sum(x, axis=1).reshape(-1, 1)
    out = x / sums
    self.softmaxes = out
    
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    d1, d2 = self.softmaxes.shape
    
    diag3d = np.stack([np.diag(x) for x in self.softmaxes[:]])
    product3d = self.softmaxes.reshape(d1, d2, 1) @ self.softmaxes.reshape(d1, 1, d2)
    dsoftmax = diag3d - product3d
    dx = (dsoftmax @ dout.reshape(d1, d2, 1)).reshape(d1, d2)
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = x + 1e-15  # Avoid x == 0
    out = np.mean(- np.sum(np.log(x) * y, axis=1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = x + 1e-15  # Avoid x == 0
    dx = - (y / x) / x.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
