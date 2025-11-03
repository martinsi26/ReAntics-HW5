import numpy as np

def initializeWeights():
    num_inputs = 4
    num_hidden = 8
    num_outputs = 1

    weights_hidden = np.random.uniform(-1, 1, (num_hidden, num_inputs))
    bias_hidden = np.random.uniform(-1, 1, (num_hidden,))
    weights_output = np.random.uniform(-1, 1, (num_outputs, num_hidden))
    bias_output = np.random.uniform(-1, 1, (num_outputs,))

    return weights_hidden, bias_hidden, weights_output, bias_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def feedforward(inputs, weights_hidden, bias_hidden, weights_output, bias_output):
    hidden_answer = sigmoid(np.dot(weights_hidden, inputs) + bias_hidden)
    output_answer = sigmoid(np.dot(weights_output, hidden_answer) + bias_output)
    return hidden_answer, output_answer

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
    inputs = np.array([0.5, -0.3, 0.8, 0.1]) # example input
    target = np.array([1.0]) # example target output

    # Initialize Network
    weights_hidden, bias_hidden, weights_output, bias_output = initializeWeights()

    # Training loop
    learning_rate = 0.1
    epochs = 5000
    for i in range(epochs):
        weights_hidden, bias_hidden, weights_output, bias_output, error = backpropagation(
            inputs, target, weights_hidden, bias_hidden, weights_output, bias_output, learning_rate
        )
        if i % 500 == 0:
            print(f"Epoch {i}, Error: {error[0]:.4f}")
    
    # Final output
    _, network_output = feedforward(inputs, weights_hidden, bias_hidden, weights_output, bias_output)
    print("\n=== Final Network Output ===")
    print(network_output)