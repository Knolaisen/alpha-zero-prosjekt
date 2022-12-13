from matplotlib import pyplot as plt
import trainingEnvironment
import numpy as np
import monteCarlo
import environment
import dataHandler
import hashlib
import time

def play_supervised_move(game): 
    print(f'\nYou are now playing against the AI!')
    game.switch_sides()
    print(f'\nCurrent board state:\n {game.get_board()}')
    available_moves = game.get_available_moves()
    supervised_move = int(input(f'\nChoose your column! Available moves: {available_moves}\n>'))
    while supervised_move not in available_moves:
        print(f'You attempted to play on column {supervised_move}, however this move is not valid')
        supervised_move = int(input(f'\nChoose your column! Available moves: {available_moves}\n> '))
    game.move(supervised_move)
    game.switch_sides()
    return None

def play(number_of_games = 1):
    #start_time = time.time()
    
    network_name = './ResNet_connect4.pth'
    neural_network = trainingEnvironment.load_network(network_name, debug=False)
    neural_network.eval()
    node_dictionary = dataHandler.load_nodes(filename='visited_nodes', debug=True)
    
    method = monteCarlo.MonteCarlo(
        network= neural_network,
        node_dictionary= node_dictionary,
        debug= False
    )
    game = environment.ConnectFour(debug=False)   
    number_of_games_left = number_of_games 
    while number_of_games_left > 0:
        game.reset()
        #print(f'{number_of_games_left} number of games left ...')
        
        if number_of_games_left % 2 == 1: # Alternate between opening the game, and answering to opponents opening.
            policy = neural_network.forward(game.board, only_policy= True) 
            opponent_opening_move = np.random.choice(np.where(policy == policy.max())[0])
            game.move(opponent_opening_move)
            game.counter = 0 # Reset counter
            game.switch_sides()

        #print(f'\nStarting game at counter {game.counter} and board state: \n{game.board}')
        while not game.is_game_over():
            #print(f'After {game.counter} moves, the following board state is: \n{game.board}')

            if game.counter % 2 == 1:
                game.switch_sides()
                best_move = method.search(
                    board= game.get_board(),
                    available_moves= game.get_available_moves(),
                    illegal_moves= game.get_illegal_moves(),
                    save_training_data= (True, 'training_data_for_player_-1_backup')
                )
                game.move(best_move)
                #print(f'Player -1 choses column {best_move}.')
                win_state = game.is_win_state()
                if win_state[0]:
                    reward = win_state[1]
                game.switch_sides()
                continue
            if game.counter % 2 == 0:
                best_move = method.search(
                    board= game.get_board(),
                    available_moves= game.get_available_moves(),
                    illegal_moves= game.get_illegal_moves(),
                    save_training_data= (True, 'training_data_for_player_1_backup')
                )
                game.move(best_move)
                #print(f'Player 1 choses column {best_move}')
                win_state = game.is_win_state()
                if win_state[0]:
                    reward = win_state[1]
                continue
        reward = game.is_win_state()[1]
        method.backpropagate(reward)
        #print(f'\nThe winner is {reward} at board state:\n{game.board}')
        #print("--- %s seconds ---" % (time.time() - start_time))
        number_of_games_left -= 1
        
    clean_board_node = method.node_dictionary[hashlib.md5(np.zeros((6,7)).data.tobytes()).hexdigest()]
    print(f'\nWin rate [wins, number of games] at clean board: {clean_board_node}; {100*clean_board_node[0]/clean_board_node[1]:.3f}%')
    #print("--- %s seconds ---" % (time.time() - start_time))

    dataHandler.save_node_dictionary(
        dict= method.node_dictionary,
        filename= 'visited_nodes',
        debug= True,
    )
    training_data = dataHandler.reshape_training_data(
        data= method.training_data,
        debug= False,
    )

    #split the data into training, cross validation and evaluation sets
    np.random.shuffle(training_data)
    epochs = 2**4
    batch_size = len(training_data)
    split = (np.array([0.7, 0.85, 1])*batch_size).astype(int) # [training, cross validation, evaluation]
    train_data = training_data[:split[0]]
    crossValidation_data = training_data[split[0]:split[1]]
    evaluation_data = training_data[split[1]:split[2]]
        
    # Training network
    train_losses_policy, train_losses_value = trainingEnvironment.train(
        network= neural_network,
        training_data= train_data,
        epochs= epochs,
        debug= False,
        overfit_test= False,
    )
    #print("--- %s seconds ---" % (time.time() - start_time))
    #print(f'''\nResults from network training: \n
    #Policy losses: {train_losses_policy} \n
    #Values losses: {train_losses_value}
    #''')
    # Cross validating network
    crossValidation_loss_policy, crossValidation_loss_value = trainingEnvironment.cross_validate(
        network= neural_network,
        validation_data= crossValidation_data,
        debug= False
    )
    #print("--- %s seconds ---" % (time.time() - start_time))
    print(f'''\nResults from cross validation:
    Policy loss: {crossValidation_loss_policy}, Values loss: {crossValidation_loss_value}
    ''')
    # Evaluate network
    policy_eval, value_eval = trainingEnvironment.evaluate(
        network= neural_network,
        eval_data= evaluation_data,
        debug= False
    )
    policy_accuracy = (100*policy_eval[0]/policy_eval[1])
    value_accuracy = (100*value_eval[0]/value_eval[1])
    #print("--- %s seconds ---" % (time.time() - start_time))
    print(f'''Results from evaluation:
    Policy_eval: {policy_eval} ; {policy_accuracy:.0f}%
    Value_eval: {value_eval} ; {value_accuracy:.0f}%
    ''')

    #trainingEnvironment.train(
    #    network= neural_network,
    #    training_data= training_data,
    #    epochs= 10,
    #    debug= False,
    #)
    trainingEnvironment.save_network(
        network= neural_network,
        name= network_name,
        debug= True,
    )
    return policy_accuracy, value_accuracy

def iterate(iterations, games_each=1):
    while iterations > 0:
        # Play several games + train model on local game history + evaluate model & fetch accuracy data 
        policy_accuracy, value_accuracy = play(number_of_games= games_each)

        # Pickle accuracy data to save in history
        dataHandler.save_policy_accuracy(policy_accuracy)
        dataHandler.save_value_accuracy(value_accuracy)

        # Load accuracy history from pickled data
        policy_accuracy_list = dataHandler.load_policy_accuracy()
        value_accuracy_list = dataHandler.load_value_accuracy()

        # Plot accuracy history
        plt.clf()
        x = np.linspace(1, len(policy_accuracy_list), len(policy_accuracy_list)).astype(int).tolist()
        plt.plot(x, policy_accuracy_list, label='Policy Accuracy')
        plt.plot(x, value_accuracy_list, label='Value Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy [%]')
        plt.legend()
        plt.savefig('Model Accuracy History')

        iterations -= 1
 
def train_exclusively():
    
    # Loading neural network
    network_name = './ResNet_connect4_test.pth'
    ResNet_connect4 = trainingEnvironment.load_network(name= network_name, debug=True)
    
    # Loading training data
    training_data_p1 = dataHandler.reshape_training_data(
            data= dataHandler.load_training_data('training_data_for_player_1_backup', debug=False),
            debug= True
        )
    training_data_p2 = dataHandler.reshape_training_data(
            data= dataHandler.load_training_data('training_data_for_player_-1_backup', debug=False),
            debug= True
        )
    training_data = dataHandler.reshape_training_data(
        np.append(training_data_p1, training_data_p2)
    )
    training_data = training_data[1000:1200]
    print(training_data[42])
    np.random.shuffle(training_data)

    #split the data into training, cross validation and evaluation sets
    epochs = 200
    batch_size = 200
    split = (np.array([0.6, 0.8, 1])*batch_size).astype(int) # [training, cross validation, evaluation]
    train_data = training_data[:split[0]]
    crossValidation_data = training_data[split[0]:split[1]]
    evaluation_data = training_data[split[1]:split[2]]
        
    # Training network
    train_losses_policy, train_losses_value = trainingEnvironment.train(
        network= ResNet_connect4,
        training_data= train_data,
        epochs= epochs,
        debug= False,
        overfit_test= False,
    )

    # Cross validating network
    crossValidation_losses_policy, crossValidation_losses_value = trainingEnvironment.cross_validate(
        network= ResNet_connect4,
        validation_data= crossValidation_data,
        debug= False
    )

    # Evaluate network
    policy_eval, value_eval = trainingEnvironment.evaluate(
        network= ResNet_connect4,
        eval_data= evaluation_data,
        debug= False
    )
    print(f'''\n\nResults from evaluation: \n
    Policy_eval: {policy_eval} ; {(100*policy_eval[0]/policy_eval[1]):.0f}% \n
    Value_eval: {value_eval} ; {(100*value_eval[0]/value_eval[1]):.0f}% \n
    ''')

    # Plot learning curve for train loss and cross validation loss
    x = np.linspace(1, epochs, epochs).astype(int).reshape(epochs).tolist()
    plt.plot(x, train_losses_policy, label='Policy Training Loss')
    plt.plot(x, train_losses_value, label='Value Training Loss')
    #plt.axhline(crossValidation_losses_policy, label='Policy Cross Validation Loss')
    #plt.axhline(crossValidation_losses_value, label='Value Cross Validation Loss')
    plt.title('Learning Curve (SGD)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Learning Curve (SGD)')
    #plt.show()

    # Performance benchmark
    # N/A ...

    # Saving trained network
    #trainingEnvironment.save_network(
    #    network= ResNet_connect4,
    #    name= network_name,
    #    debug= True,
    #)
    return None
    
#train_exclusively()
iterate(iterations= 2**12, games_each= 2**4)
