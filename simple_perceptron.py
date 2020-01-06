import numpy as np

# Sigmoid normalizing function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of sigmoid normalizing function


def derivative_of_sigmoid(x):
    return x * (1 - x)


# Inputs for testing
inputs = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1]])

# Outputs for testing (transposed so it becomes 4x1 matrix)
actual_outputs = np.array([[0, 0, 1, 1]]).T

# Gets the same random numbers every time
np.random.seed(1)

# Three-by-one array of random numbers from [-1, 1):
weights = 2 * np.random.random((3, 1)) - 1

# Check the starting synaptic weights
print('Random weights we\'re starting out with:')
print(weights)

# Repeat 100000 times to get as close to 0 or 1 as possible
for x in range(100000):
    input_layer = inputs

    # Dot product the input with weights then apply sigmoid function to normalize it between 0 and 1
    outputs = sigmoid(np.dot(input_layer, weights))

    # Calculate the error which is the difference between the actual outputs and obtained outputs
    error = actual_outputs - outputs

    # Adjustment is the error multiplied by the sigmoid derivative of outputs (Adjustments need to be proportional to size of error)
    # If the sigmoid derivative of the output is large, the severeness of error is large (So need larger adjustments)
    # If the sigmoid derivative of the output is small, the severeness of error is small (So only need smaller adjustment)
    adjustments = error * derivative_of_sigmoid(outputs)

    # Dot product of input and adjustments then added to itself to give synaptic weight
    # If input is 0, weights aren't adjusted
    weights += np.dot(input_layer.T, adjustments)

# Check weights after adjustments
print('Weights after adjustments: ')
print(weights)

# As sigmoid function does not allow the numbers to be completely 0 or 1, set it manually
for idx,val in enumerate(outputs):
    if val > 0.9:
        outputs[idx] = 1
    else:
        outputs[idx] = 0

# Check otained outputs
print('Obtained outputs: ')
print(outputs)
