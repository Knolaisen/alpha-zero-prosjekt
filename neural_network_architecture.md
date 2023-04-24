# We wanted to create something similar to AlphaZero neural network.
The neural network used in AlphaZero is a type of convolutional neural network (CNN) that is specifically designed to process board game states. It has multiple convolutional layers followed by fully connected layers, and uses residual connections to improve gradient flow during training.




## What and why residual blocks

A residual block is designed to address the problem of vanishing gradients that can occur when training very deep neural networks. 
The idea behind residual blocks is to add a shortcut connection that allows information to flow directly from the input of a layer to its output, 
bypassing one or more layers in between.
    
To learn more one can read the the paper "Deep Residual Learning for Image Recognition" by Kaiming He et al (2015).

We chose to imitate AlphaZero and we have 20 residual blocks in our net




