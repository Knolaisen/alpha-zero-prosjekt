from pickle import load, dump
import numpy as np

# Loading dictionary with stored empirical values for each unique board state: key; board-id, values; [empirical value, visit count]
def load_nodes(filename='nodes', debug=False) -> dict:
	try:
		with open(filename, 'rb') as infile:
			node_dictionary = load(infile)
			if debug:
				print('Loading node dictionary at the length of', len(node_dictionary))
			return node_dictionary
	except (OSError, IOError) as e:
		if debug:
			print('No node dictionary found. Returning empty dictionary')
		return {}
		
# Loading array with training data: board state, policy target and value target respectively
def load_training_data(filename='training_data', debug=False) -> object:
	try:
		training_data = np.array([], dtype='object')
		infile = open(filename, 'rb')
		infile.seek(0)
		while True:
			try:
				training_data = np.append(training_data, load(infile))
				if debug:
					print('Loading training data at the shape of', np.shape(training_data))
			except EOFError as e:
				print(f'No more input to read. Closing file {filename} and returning training data in the shape of {np.shape(training_data)}')
				infile.close()
				return training_data
	except (OSError, IOError) as e:
		if debug:
			print('No training data found. Returning empty array')
		return np.array([], dtype='object')

def save_node_dictionary(dict, filename='nodes', debug=False) -> None:
	with open(filename, 'wb') as outfile:
		dump(dict, outfile)
		if debug:
			print('Saving node dictionary at the length of', len(dict))

def reset_node_dictionary(filename): #TODO FIX: reset funksjonen fungerer ikke enda, så filene må slettes manuelt
	empty_node_dictionary = {}
	file = open(filename, 'wb')
	dump(empty_node_dictionary, file)
	file.close()

def reshape_training_data(data, debug=False) -> object:
	columns = 3
	rows = int(len(data)/columns)
	training_data = np.reshape(data, newshape=(rows, columns))
	if debug:
		print(f'Reshaped training data to shape: {np.shape(training_data)}')
	return training_data

def save_training_data(data, filename='training_data', debug=False, append=False) -> None:
	if append:
		if debug:
			file = open(filename, 'ab+')
			dump(data, file)
			print(f'Appended training_data of shape {np.shape(data)}')
	
			file.seek(0)
			training_data = np.array([], dtype='object')
			while True:
				try:
					training_data = np.append(training_data, load(file))
				except EOFError as e:
					print(f'Closing file with updated shape of {np.shape(training_data)}')
					file.close()
					break
		else:
			with open(filename, 'ab') as outfile:
				dump(data, outfile)
	else:
		with open(filename, 'wb') as outfile:
			dump(data, outfile)
		if debug:
			print('Dumped training data at the shape of', np.shape(data))

def show_training_data(data, reshape=False, debug=False) -> str:
    if reshape:
        data = reshape_training_data(data, debug=debug)
        
    string = '\nTraining data:\n'
    for i in range(3):
        string += '\nRow' + str(i) + ':' + str(data[i])
    for i in range(len(data)-1,len(data)-3, -1):
        string += '\nRow' + str(i) + ':' + str(data[i])
    print(string)    
    
def reset_training_data(filename='training_data'): # TODO FIX: reset funksjonen fungerer ikke enda, så filene må slettes manuelt
	print(f'Resetting {filename}')
	empty_training_data = np.array([], dtype='object')
	file = open(filename, 'wb')
	dump(empty_training_data, file)
	file.close()
