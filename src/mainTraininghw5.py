#Resources: https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
#I untilized this website to help me with my backpropagation function
import numpy as np
import random


##
#initializeWeights
#
#Description: Randomly sets up weights for all of the hidden notes and output nodes
#
#Return: All of the weights and baises for the nodes
##
def initializeWeights():
    num_inputs = 4
    num_hidden = 8
    num_outputs = 1

    weights_hidden = np.random.uniform(-1, 1, (num_hidden, num_inputs))
    bias_hidden = np.random.uniform(-1, 1, (num_hidden,))
    weights_output = np.random.uniform(-1, 1, (num_outputs, num_hidden))
    bias_output = np.random.uniform(-1, 1, (num_outputs,))

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
    examples = [
        ([0, 0, 0, 0], [0]),
        ([0, 0, 0, 1], [1]),
        ([0, 0, 1, 0], [0]),
        ([0, 0, 1, 1], [1]),
        ([0, 1, 0, 0], [0]),
        ([0, 1, 0, 1], [1]),
        ([0, 1, 1, 0], [0]),
        ([0, 1, 1, 1], [1]),
        ([1, 0, 0, 0], [1]),
        ([1, 0, 0, 1], [1]),
        ([1, 0, 1, 0], [1]),
        ([1, 0, 1, 1], [1]),
        ([1, 1, 0, 0], [0]),
        ([1, 1, 0, 1], [0]),
        ([1, 1, 1, 0], [0]),
        ([1, 1, 1, 1], [1])
    ]
    
    # inputs = np.array([0.5, -0.3, 0.8, 0.1]) # example input
    # target = np.array([1.0]) # example target output

    # Initialize Network
    weights_hidden, bias_hidden, weights_output, bias_output = initializeWeights()

    # Training loop
    totalError = 0
    learning_rate = 0.1
    epoch = 0
    average_error = 1.0

    # Train until we reach a 0.05 average error for an epoch
    while average_error > 0.05:
        # Radnomly pick 10 samples per epoch
        selected = random.sample(examples, 10)
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

        # Compute average error for the epoch
        average_error = total_error / 10
        epoch += 1
        print(f"Epoch {epoch} - Average Error: {average_error:.4f}")
    
    # # Final output
    # _, network_output = feedforward(inputs, weights_hidden, bias_hidden, weights_output, bias_output)
    # print("\n=== Final Network Output ===")
    # print(network_output)