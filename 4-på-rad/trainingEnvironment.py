from neuralNetwork_bigger import ResNet, device
import torch
from torch import nn
import numpy as np
#import environment

def load_network(name='./connect_4_brain.pth', device=device, debug=False) -> object:
    PATH = name
    network = ResNet().to(device=device)
    try:
        network.load_state_dict(torch.load(PATH))
        if debug:
            print(f'Loading network from path:{PATH}')
        return network
    except (OSError, IOError) as e:
        print('Could not find pre-existing network. Returning randomly initialized network')
        return network

#torch.autograd.set_detect_anomaly(True)

def train(network, training_data, epochs=1, debug=False, overfit_test=False):
    network.train()
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    weight_decay = 1e-4
    momentum = 0.9
    
    #print(f'Using Adam optimizer')
    #learning_rate = 6e-6
    #optimizer = torch.optim.Adam(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print(f'Using SGD optimizer')
    learning_rate = 1e-4
    optimizer = torch.optim.SGD(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    
    train_losses_value = []
    train_losses_policy = []

    for epoch in range(epochs):

        np.random.shuffle(training_data)
        running_loss_policy = 0.0
        running_loss_value = 0.0
        training_data_shape = np.shape(training_data)
        roof = training_data_shape[0]-1

        for i, data in enumerate(training_data, 0):
            # getting inputs
            board_state, policy_target, value_target = data
            
            # converting to tensor
            policy_target = torch.from_numpy(np.asarray(np.argmax(policy_target))).to(device=device)
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
            
            if debug:
                print(f'\npolicy output: {policy_output}\npolicy target: {policy_target}\nvalue output: {value_output}\nvalue target: {value_target}')
            
            # print statistics for policy
            running_loss_policy += loss_policy.item()
            running_loss_value += loss_value.item()
            #stat_batch = 5
            #if i % stat_batch == stat_batch-1: # print every 2000 mini-batches
            
            if i == roof:
                #print(f'[{epoch + 1}, {i + 1:5d}] loss_policy: {running_loss_policy / stat_batch:.3f}, loss_value: {running_loss_value / stat_batch:.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] loss_policy: {running_loss_policy / (roof+1):.3f}, loss_value: {running_loss_value / (roof+1):.3f}')
                train_losses_policy.append(running_loss_policy/(roof+1))
                train_losses_value.append(running_loss_value/(roof+1))
                running_loss_policy = 0.0
                running_loss_value = 0.0
    #print('Finished Training')
    return train_losses_policy, train_losses_value

def cross_validate(network, validation_data, debug=False):
    network.eval()
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    crossValidation_loss_value = None
    crossValidation_loss_policy = None

    np.random.shuffle(validation_data)
    running_loss_policy = 0.0
    running_loss_value = 0.0
    validation_data_shape = np.shape(validation_data)
    roof = validation_data_shape[0]-1

    for i, data in enumerate(validation_data, 0):
        # getting inputs
        board_state, policy_target, value_target = data
        
        # converting to tensor
        policy_target = torch.from_numpy(np.asarray(np.argmax(policy_target))).to(device=device)
        value_target = torch.from_numpy(value_target).to(device=device, dtype=torch.float32)
        
        # forward + calculate loss
        policy_output, value_output = network.forward(board_state, tensor=True)  
        loss_policy = criterion_policy(policy_output, policy_target)
        loss_value = criterion_value(value_output, value_target)
        
        if debug:
            print(f'\npolicy output: {policy_output}\npolicy target: {policy_target}\nvalue output: {value_output}\nvalue target: {value_target}')
        
        # print statistics for policy
        running_loss_policy += loss_policy.item()
        running_loss_value += loss_value.item()
        if i == roof:
            #print(f'[{i + 1:5d}] loss_policy: {running_loss_policy / (roof+1):.3f}, loss_value: {running_loss_value / (roof+1):.3f}')
            crossValidation_loss_policy = running_loss_policy/(roof+1)
            crossValidation_loss_value = running_loss_value/(roof+1)
            running_loss_policy = 0.0
            running_loss_value = 0.0

    #print(f'Finished cross validating')
    return crossValidation_loss_policy, crossValidation_loss_value

def evaluate(network, eval_data, debug=False):
    network.eval()
    correct_policy = 0
    correct_value = 0
    total_policy = 0
    total_value = 0
    with torch.no_grad():
        for i, data in enumerate(eval_data, 0):
            # getting inputs
            board_state, policy_target, value_target = data
            
            # converting to tensor
            policy_target = torch.from_numpy(np.asarray(np.argmax(policy_target))).to(device=device)
            value_target = torch.from_numpy(value_target).to(device=device, dtype=torch.float32)
            
            # forward + apply softmax, nll and take argmax of policy
            policy_output, value_output = network.forward(board_state, tensor=True)  
            best_move = torch.argmax(nn.functional.log_softmax(policy_output, dim=0))

            if debug:
                print(f'\npolicy output (best move): {best_move}\npolicy target: {policy_target}\nvalue output: {value_output}\nvalue target: {value_target}')
     
            # +1 polcy prediction point for correctly predicted best_move, +1 value prediction point for correct side [-1, 0, 1]
            total_policy += 1
            if best_move == policy_target:
                correct_policy += 1
            
            total_value += 1
            if torch.sign(value_output) == torch.sign(value_target):
                correct_value += 1
    #print(f'Finished evaluating')
    return (correct_policy, total_policy), (correct_value, total_value)
        
def save_network(network, name='./connect_4_brain.pth', debug=False) -> None:
    PATH = name
    torch.save(network.state_dict(), PATH)
    if debug:
        print(f'Saving neural network to path: {PATH}')
