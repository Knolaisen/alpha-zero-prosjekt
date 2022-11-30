from neural_network import NeuralNetwork, device
from torch import nn
import torch
import pickle

#TODO Importere modellen som allerede har blitt trent så man slipper å starte fra scratch hver gang

# Array storing all training data: board state, policy target and value target respectively 
filename_training = 'training_data'
try:
	infile_training = open(filename_training, 'rb')
	training_data = pickle.load(infile_training)
	infile_training.close()
except (OSError, IOError) as e:
	print('No training data available')

connect_4_brain = NeuralNetwork().to(device=device)
torch.autograd.set_detect_anomaly(True)

criterion_policy = nn.CrossEntropyLoss()
criterion_value = nn.MSELoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(params=connect_4_brain.parameters(), lr=learning_rate)

number_of_possible_moves = 7

epochs = 1
for epoch in range(epochs):

    running_loss_policy = 0.0
    running_loss_value = 0.0
    for i, data in enumerate(training_data, 0):
        # getting inputs
        board_state, policy_target, value_target = data

        # converting to tensor
        policy_target = torch.from_numpy(policy_target).to(device=device, dtype=float)
        value_target = torch.from_numpy(value_target).to(device=device, dtype=torch.float32)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        policy_output, value_output = connect_4_brain.forward(board_state, tensor=True)
        loss_policy = criterion_policy(policy_output, policy_target)
        loss_policy.backward(retain_graph=True)
        loss_value = criterion_value(value_output, value_target)
        loss_value.backward()
        optimizer.step()

        # print statistics for policy
        running_loss_policy += loss_policy.item()
        running_loss_value += loss_value.item()
        stat_batch = 50
        if i % stat_batch == stat_batch-1: # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss_policy: {running_loss_policy / stat_batch:.3f}')
            print(f'[{epoch + 1}, {i + 1:5d}] loss_value: {running_loss_value / stat_batch:.3f}')
            running_loss_policy = 0.0
            running_loss_value = 0.0

print('Finished Training')

PATH = './connect_4_brain.pth'
torch.save(connect_4_brain.state_dict(), PATH)