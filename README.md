# Assignments and solutions for Deep Learning course, MSc AI @ UvA 2018/2019.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  
Solutions and implementation from [Davide Belli](https://github.com/davide-belli)
Total Grade: 96 %

## Assignment 1: MLPs, CNNs and Backpropagation
In this assignment you will learn how to implement and train basic neural architectures
like MLPs and CNNs for classification tasks. Modern deep learning libraries come
with sophisticated functionalities like abstracted layer classes, automatic differentiation,
optimizers, etc.

#### Tasks:
- Derive equations for MLP backpropagation
- Implement forward and backward MLP in NumPy
- Implement MLP in PyTorch and improve it with additional layers
- Derive backpropagation for Batch Normalization and implement it manually in PyTorch
- Implement a simple CNN in PyTorch
- Improve CNN with additional layers 
---

## Assignment 2: Recurrent Neural Networks
In this assignment you will study and implement recurrent neural networks (RNNs). This
type of neural network is best suited for sequential processing of data, such as a sequence
of characters, words or video frames. Its applications are mostly in neural machine
translation, speech analysis and video understanding. These networks are very powerful
and have found their way into many production environments. For example Google’s
neural machine translation system relies on Long-Short Term Networks (LSTMs).

#### Tasks:
- Derive equations for RNN backpropagation
- Implement vanilla RNN in PyTorch
- Implement LSTM in PyTorch
- Compare performance and results of RNN and LSTM on palyndrome sequences
- Compare optimizers: RMSProp, Adam based on momentum and adaptive learning rate concepts
- Study a modified LSTM cell to capture input patterns with a frequence
- Implement a Generative Model learning from books
- Explare greedy-sampling and random-sampling with temperature

#### Samples of generated text:
- Anna Karenina:  
_He did not really like the first to the peasants were standing at the door. He saw the club, and his
book was a girl who had always been at the same time and had so much as the carriage with one of
the conversation with_
- Shakespeare works:  
_He did not really like a friend, I will have my  
About my heart with the saltiness meel, but weived thou deeds.  
BROTUD:  
With days,  
Mifte ours, you will make the marriage, sir.  
BOYET:  
Ay, Sir John! and this was so many as t_
- The Republic:  
_He did not really like to fall under the performer of all the objects, and the souls of the soul, and are
all objects against any one who is not the best of all, but there is no difficulty in the same thing is
concerned wit_
- The Odissey:
_He did not really like to speak of the house. Telemachus spoken of for its bed, for I have never seen
the great stringing of the court, and something for myself to the same survain it will be reaching this
way, but they may_
- LATEX:  
\begin{DPalign*}  
&nbsp;&nbsp;\lintertext{\rlap{\indent Then show that we know that it is a maximum.}  
\end{DPalign*}  
The last term is
$\dfrac{dy}{dx} = \dfrac{1}{\sqrt{\theta^5}} - \dfrac{1}{\sqrt{\theta^5}} - \dfrac{1}
- HTML:
<pre>
<p><font size="2">The structure of a response is signon information aggregate</font></td>
</tr>
<tr>
<td width="180"><b><font size="1">&lt;STATE&gt;</font></b></td>
<td width="366"><font size="2">Custom
</pre>

---
## Assignment 3: Deep Generative Models

In this assignment you will study and implement Deep Generative Models. Deep generative
models come in many flavors, but all represent the probability distribution (of the data)
in some way or another. Some generative models allow for explicit evaluation of the log
likelihood, whereas other may only support some function that relies on the probability
distribution—such as sampling. In this assignment, we focus on two—currently most
popular—deep generative models, namely Variational Auto Encoders (VAE) and
Generative Adversarial Networks (GAN). You will implement both in PyTorch as part
of this assignment. Note that, although this assignment does contain some explanation
on the model, we do not aim to give a complete introduction in this assignment. The
best source for understanding the model are the papers that introduced them, the
hundreds of blogpost that have been written on them ever since, or the lecture.

#### Tasks:
- Theoretical questions about ancestral sampling, latent models, MC integration, VAE prior-posterior divergence, lower bound, reparameterization trick
- Derive VAE reconstruction and regularization losses
- Implement a VAE in PyTorch
- Theoretical questions about mini-max games, convergence, GAN loss
- Implement a GAN in PyTorch
- Compare VAE and GAN results on MNIST dataset
- Visualize and discuss manifolds and latent space interpolations

---

## Copyright

Copyright © 2019 Davide Belli.

<p align=“justify”>
This project is distributed under the <a href="LICENSE">MIT license</a>.  
Please follow the <a href="http://student.uva.nl/en/content/az/plagiarism-and-fraud/plagiarism-and-fraud.html">UvA regulations governing Fraud and Plagiarism</a> in case you are a student.
</p>
