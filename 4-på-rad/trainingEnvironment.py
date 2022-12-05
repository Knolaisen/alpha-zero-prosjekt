from neuralNetwork import NeuralNetwork, device
import torch
from torch import nn
import numpy as np
import environment

def load_network(name='./connect_4_brain.pth', device=device, debug=False) -> object:
    PATH = name
    network = NeuralNetwork().to(device=device)
    try:
        network.load_state_dict(torch.load(PATH))
        if debug:
            print(f'Loading network from path:{PATH}')
        return network
    except (OSError, IOError) as e:
        print('Could not find pre-existing network. Returning randomly initialized network')
        return network

#torch.autograd.set_detect_anomaly(True)

def train(network, training_data, epochs=1, debug=False) -> None:

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    #learning_rate = 8e-7
    learning_rate = 1e-2
    weight_decay = False
    optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    number_of_possible_moves = 7

    for epoch in range(epochs):

        running_loss_policy = 0.0
        running_loss_value = 0.0
        training_data_shape = np.shape(training_data)
        roof = training_data_shape[0]-1

        for i, data in enumerate(training_data, 0):
            # getting inputs
            board_state, policy_target, value_target = data
            game = environment.ConnectFour(board= board_state)
            if game.is_game_over():
                exception_message = f'WARNING: attempted training at closed game: \n{data}\nEpoch: {epoch+1}, iteration: {i}.'
                raise Exception(exception_message)
            if np.count_nonzero(abs(policy_target)>0) > len(game.get_available_moves()):
                exception_message = f'WARNING: attempted training with flawed policy target: \n{data}\nEpoch: {epoch+1}, iteration: {i}.'
                raise Exception(exception_message)

            # converting to tensor
            policy_target = torch.from_numpy(policy_target).to(device=device, dtype=float)
            value_target = torch.from_numpy(value_target).to(device=device, dtype=torch.float32)
      
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            policy_output, value_output = network.forward(board_state, tensor=True)  
            loss_policy = criterion_policy(policy_output, policy_target)
            loss_policy.backward(retain_graph=True)
            loss_value = criterion_value(value_output, value_target)
            loss_value.backward()
            optimizer.step()
            
            # print statistics for policy
            running_loss_policy += loss_policy.item()
            running_loss_value += loss_value.item()
            #stat_batch = 5
            #if i % stat_batch == stat_batch-1: # print every 2000 mini-batches
            
            if i == roof:
                #print(f'[{epoch + 1}, {i + 1:5d}] loss_policy: {running_loss_policy / stat_batch:.3f}, loss_value: {running_loss_value / stat_batch:.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] loss_policy: {running_loss_policy / (roof+1):.3f}, loss_value: {running_loss_value / (roof+1):.3f}')
                running_loss_policy = 0.0
                running_loss_value = 0.0

    #print('Finished Training')

def save_network(network, name='./connect_4_brain.pth', debug=False) -> None:
    PATH = name
    torch.save(network.state_dict(), PATH)
    if debug:
        print(f'Saving neural network to path: {PATH}')

# --- OPTIMIZATION TRICKS --- #
# Disable bias for convolutions directly followed by a batch norm
# This works since the biases are canceled through batch norm anyways, hence it changes nothing and speeds it up
# use: nn.Conv2d(..., bias=False, ....)
# note: batchnorm needs to normalize on the same dimension as the conv-biases are in