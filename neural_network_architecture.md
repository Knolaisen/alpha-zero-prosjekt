# We wanted to create something similar to AlphaZero neural network.
The neural network used in AlphaZero is a type of convolutional neural network (CNN) that is specifically designed to process board game states. It has multiple convolutional layers followed by fully connected layers, and uses residual connections to improve gradient flow during training.




## What and why residual blocks

A residual block is designed to address the problem of vanishing gradients that can occur when training very deep neural networks. 
The idea behind residual blocks is to add a shortcut connection that allows information to flow directly from the input of a layer to its output, 
bypassing one or more layers in between.
    
To learn more one can read the the paper "Deep Residual Learning for Image Recognition" by Kaiming He et al (2015).

We chose to imitate AlphaZero and we have 20 residual blocks in our net




# The architecture of our Neural network
It has three main parts: the input convolutional layer, the residual blocks, and two output heads (policy and value). Where the policy head tries to find out what move the current player should play and consists of about 4000 nodes. The Value head tries to predict whom will win given the input layer and is one node.

## In-depht explainetion of value head:
The value_head is designed to estimate the value of a given state, which is a measure of how favorable a certain game state is for the player. In the context of board games or reinforcement learning, this value is used to help the AI make better decisions on which actions to take.

1. nn.Conv2d(num_filters, 1, kernel_size=1): This convolutional layer is used to reduce the number of channels to 1. It applies a 1x1 convolution kernel, which is essentially a linear combination of the input channels for each spatial location. This operation is used to integrate the information from the input channels and create a single channel output. It is a common technique for reducing the dimensionality of the data while retaining important features.

2. nn.BatchNorm2d(1): Batch normalization is used to normalize the outputs of the previous layer, improving the stability and convergence speed of the network during training.

3. nn.ReLU(inplace=True): The ReLU activation function introduces non-linearity into the network and helps capture complex patterns in the input data.

4. nn.Flatten(): The flattening layer reshapes the output tensor into a 1D tensor to prepare it for the following fully connected layers.

5. nn.Linear(config.INPUT_SIZE , 256): This linear (fully connected) layer maps the flattened input to a 256-dimensional hidden space. The number 256 is chosen as an arbitrary size for the hidden layer and can be adjusted based on the problem's complexity and the network's capacity.

6. nn.ReLU(inplace=True): Another ReLU activation function is used to introduce non-linearity in the fully connected part of the network.

7. nn.Linear(256, 1): The final linear layer maps the 256-dimensional hidden space to a single output value. This value represents the estimated value of the given state.

8. nn.Tanh(): The Tanh activation function scales the output value to the range of -1 to 1. This is useful in the context of board games or other situations where the value function's output should be bounded. A value close to 1 indicates a highly favorable state for the player, while a value close to -1 indicates a highly unfavorable state.

The value_head is designed this way to process the input tensor, extract meaningful features, and produce a single scalar value that represents the favorability of the input state. The architecture is chosen to be relatively simple yet capable of capturing the necessary information to make accurate value predictions.

## In-depht explainetion of policy head:

The policy_head is designed to estimate the probabilities of different actions or moves for a given state. In the context of board games or reinforcement learning, these probabilities help the AI make decisions on which actions to take. The policy_head is responsible for outputting a probability distribution over the possible actions.


1. nn.Conv2d(num_filters, 2, kernel_size=1): This convolutional layer applies a 1x1 convolution kernel to reduce the number of channels to 2. This operation is used to integrate the information from the input channels and create a 2-channel output. The choice of 2 channels is arbitrary and may depend on the specific problem or desired level of feature abstraction.

2. nn.BatchNorm2d(2): Batch normalization is used to normalize the outputs of the previous layer, improving the stability and convergence speed of the network during training.

3. nn.ReLU(inplace=True): The ReLU activation function introduces non-linearity into the network and helps capture complex patterns in the input data.

4. nn.Flatten(): The flattening layer reshapes the output tensor into a 1D tensor to prepare it for the following fully connected layer.

5. nn.Linear(2 * 65 * 1, config.OUTPUT_SIZE): This linear (fully connected) layer maps the flattened input to the desired output size, which corresponds to the number of possible actions. The output size is determined by the config.OUTPUT_SIZE parameter.

6. nn.Softmax(dim=1): The softmax activation function converts the output of the linear layer into a probability distribution over the possible actions. The softmax function ensures that the probabilities sum to 1 and emphasizes the most likely actions while suppressing the least likely ones.

The policy_head is designed this way to process the input tensor, extract meaningful features, and produce a probability distribution over the possible actions. The architecture is chosen to be relatively simple but effective in capturing the necessary information to make accurate policy predictions. 

## In-depht explainetion of residual blocks:

The residual_blocks is a sequence of residual blocks that are designed to improve the performance and learning capability of the neural network, especially when it becomes deeper. These residual blocks allow the network to learn complex features and patterns in the input data while mitigating the vanishing gradient problem commonly encountered in deep networks.

1. nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding): The first convolutional layer in the residual block applies a 2D convolution with a specified kernel size and padding. This layer is responsible for extracting features from the input data.

2. nn.BatchNorm2d(out_channels): The first batch normalization layer normalizes the outputs of the previous convolutional layer. This improves the stability and convergence speed of the network during training.

3. nn.ReLU(inplace=True): The ReLU activation function introduces non-linearity into the network, which helps capture complex patterns in the input data.

4. nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding): The second convolutional layer in the residual block applies another 2D convolution with the same kernel size and padding as the first layer. This layer refines the features extracted in the previous layers.

5. nn.BatchNorm2d(out_channels): The second batch normalization layer normalizes the outputs of the previous convolutional layer, further improving the stability and convergence speed of the network.

6. Residual Connection: The input tensor (residual) is added to the output tensor of the second batch normalization layer. This skip connection allows the gradients to flow more easily through the network during training, which helps mitigate the vanishing gradient problem in deep networks. The residual connection also enables the network to learn identity mappings when no additional complexity is needed, allowing the network to adapt its depth based on the problem's complexity.

7. nn.ReLU(inplace=True): The final ReLU activation function introduces non-linearity after the residual connection.

The residual_blocks are designed this way to enable the neural network to learn complex features while maintaining the stability and performance of the network as it becomes deeper. The combination of convolutional layers, batch normalization, and ReLU activations helps to capture important patterns in the input data, and the residual connections allow the network to learn effectively even when it is very deep.

## Picture illustrate architecture
<img width="356" alt="alphaZero_architecture" src="https://user-images.githubusercontent.com/89105607/234270822-95b8d67e-149a-46c9-a3e5-abdff6eda93e.png">
TODO We should create our own ilustration
