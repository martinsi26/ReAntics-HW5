#Resources: https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
#I untilized this website to help me with my backpropagation function
import numpy as np
import csv
import random
import glob

NUM_OUTPUTS = 1


##
#initializeWeights
#
#Description: Randomly sets up weights for all of the hidden notes and output nodes
#
#Return: All of the weights and baises for the nodes
##
def initializeWeights(num_inputs, num_hidden):
    num_outputs = NUM_OUTPUTS


    # weights_hidden = np.random.uniform(-1, 1, (num_hidden, num_inputs))
    # bias_hidden = np.random.uniform(-1, 1, (num_hidden,))
    # weights_output = np.random.uniform(-1, 1, (num_outputs, num_hidden))
    # bias_output = np.random.uniform(-1, 1, (num_outputs,))

    weights_hidden = np.random.randn(num_hidden, num_inputs) * np.sqrt(2 / num_inputs)
    bias_hidden = np.zeros((num_hidden,))
    weights_output = np.random.randn(num_outputs, num_hidden) * np.sqrt(2 / num_hidden)
    bias_output = np.zeros((num_outputs,))

    return weights_hidden, bias_hidden, weights_output, bias_output

##
#sigmoid
#
#Description: Performs the sigmoid function calculation on nodes
#
#Parameters:
#   x - the value calculted for a node
#
#Returns: The result of the sigmoid function on a nodes value
##
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

##
#sigmoid_derivative
#
#Description: Performs the derivative of the sigmoid function calculation on nodes
#
#Parameters:
#   x - the value calculated for a node
#
#Result: The result of the derivative sigmoid function on a nodes value
##
def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return (x > 0).astype(float)

##
#feedforward
#
#Description: Finds the values of all the hidden nodes and output nodes
#
#Parameters:
#   inputs - An array of input values for the network
#   weights_hidden - An array of weights for the hidden nodes
#   bias_hidden - An array of weights for the biases of the hidden nodes
#   weights_output - An array of weights for the output nodes
#   bias_output - An array of weights for the biases of the output nodes
#
#Returns: The results of each node in the network (hidden nodes and output nodes)
##
def feedforward(inputs, weights_hidden, bias_hidden, weights_output, bias_output):
    hidden_answer = sigmoid(np.dot(weights_hidden, inputs) + bias_hidden)
    output_answer = sigmoid(np.dot(weights_output, hidden_answer) + bias_output)
    return hidden_answer, output_answer

##
#backpropagation
#
#Description: A function that calculates error of the network and recalculates the weights
#
#Parameters:
#   inputs - An array of input values for the network
#   target - The desired output of the network
#   weights_hidden - An array of weights for the hidden nodes
#   bias_hidden - An array of weights for the biases of the hidden nodes
#   weights_output - An array of weights for the output nodes
#   bias_output - An array of weights for the biases of the output nodes
#   learning_rate - A value that determins how quickly the network learns
#
#Returns: All of the updates weights for each node, including their baises, 
#and the error of the network (how far off the final value is from our target value)
##
def backpropagation(inputs, target, weights_hidden, bias_hidden, weights_output, bias_output, learning_rate):
    # Forward pass
    predicted_hidden, predicted_output = feedforward(inputs, weights_hidden, bias_hidden, weights_output, bias_output)

    # Output layer error
    output_error = target - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    # Hidden layer error
    hidden_error = np.dot(weights_output.T, output_delta)
    hidden_delta = hidden_error * sigmoid_derivative(predicted_hidden)

    # Update weights and biases
    weights_output += np.outer(output_delta, predicted_hidden) * learning_rate
    bias_output += output_delta * learning_rate

    weights_hidden += np.outer(hidden_delta, inputs) * learning_rate
    bias_hidden += hidden_delta * learning_rate

    return weights_hidden, bias_hidden, weights_output, bias_output, output_error

# --- Main test ---
if __name__ == "__main__":
    # examples = [
    #     ([0, 0, 0, 0], [0]),
    #     ([0, 0, 0, 1], [1]),
    #     ([0, 0, 1, 0], [0]),
    #     ([0, 0, 1, 1], [1]),
    #     ([0, 1, 0, 0], [0]),
    #     ([0, 1, 0, 1], [1]),
    #     ([0, 1, 1, 0], [0]),
    #     ([0, 1, 1, 1], [1]),
    #     ([1, 0, 0, 0], [1]),
    #     ([1, 0, 0, 1], [1]),
    #     ([1, 0, 1, 0], [1]),
    #     ([1, 0, 1, 1], [1]),
    #     ([1, 1, 0, 0], [0]),
    #     ([1, 1, 0, 1], [0]),
    #     ([1, 1, 1, 0], [0]),
    #     ([1, 1, 1, 1], [1])
    # ]

    counter = 0
    weights_hidden, bias_hidden, weights_output, bias_output = None, None, None, None
    for file in sorted(glob.glob("data_part*.csv")):
        print(f"On file {file}")
        counter += 1

        firstcols = []
        lastcol = []

        with open(file, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                # convert all values to float (optional)
                values = [float(x) for x in row]
                firstcols.append(values[:-1])
                lastcol.append(values[-1])

        data = list(zip(firstcols, lastcol))
        
        # inputs = np.array([0.5, -0.3, 0.8, 0.1]) # example input
        # target = np.array([1.0]) # example target output

        # Initialize Network
        inputs = len(firstcols[0])
        print(f"{inputs} parameters, {len(firstcols)}")
        if counter == 1:
            weights_hidden, bias_hidden, weights_output, bias_output = initializeWeights(inputs, int(inputs / 4))
        # Training loop
        totalError = 0
        learning_rate = 0.1
        batch = 0
        average_error = 1.0
        batch_size = 64

        # Train until we reach a 0.025 average error for an batch
        while average_error > 0.2 or batch <= len(data) / batch_size:
            # Radnomly pick 10 samples per batch
            selected = random.sample(data, batch_size)
            inputs_array = np.array([i for i, _ in selected])
            targets_array = np.array([t for _, t in selected])

            # Reset error accumulator
            total_error = 0.0

            # Train on 10 samples
            for inputs, target in zip(inputs_array, targets_array):
                weights_hidden, bias_hidden, weights_output, bias_output, error = backpropagation(
                    inputs, target, weights_hidden, bias_hidden, weights_output, bias_output, learning_rate
                )
                total_error += np.abs(error[0])

            # Adam optimization
            if random.randint(0,1000) == 0:
                rand = random.randint(0, len(weights_hidden) - 1)
                weights_hidden[rand] = 0.0
            elif random.randint(0,1000) == 0:
                rand = random.randint(0, len(weights_output) - 1)
                weights_output[rand] = 0.0

            # Compute average error for the batch
            average_error = total_error / 10
            batch += 1
            if batch % 10 == 0:
                print(f"Batch {batch} - Average Error: {average_error:.4f}")
            
        
    np.savez("weights",
            weights_hidden=weights_hidden,
            bias_hidden=bias_hidden,
            weights_output=weights_output,
            bias_output=bias_output)
        # # Final output
        # _, network_output = feedforward(inputs, weights_hidden, bias_hidden, weights_output, bias_output)
        # print("\n=== Final Network Output ===")
        # print(network_output)