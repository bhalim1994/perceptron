import numpy as np


class NeuralNetwork():
    def __init__(self):
        """
        Initializes the class
        """
        # Gets the same random numbers every time
        np.random.seed(1)

        # Three-by-one array of random numbers from [-1, 1)
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        """
        Sigmoid function that normalizes the input between 0 and 1
        """
        return 1 / (1 + np.exp(-x))

    def derivation_of_sigmoid(self, x):
        """
        Derivative of sigmoid function to calculate adjustments for weight
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        Model goes through trial and error, re-adjusting the weights in each iteration for a more correct output
        """
        for x in range(training_iterations):
            output = self.think(training_inputs)

            # Calculate the error which is the difference between the actual outputs and obtained outputs
            error = training_outputs - output

            # Adjustment is the error multiplied by the sigmoid derivative of outputs (Adjustments need to be proportional to size of error)
            # If the sigmoid derivative of the output is large, the severeness of error is large (So need larger adjustments)
            # If the sigmoid derivative of the output is small, the severeness of error is small (So only need smaller adjustment)
            adjustments = np.dot(training_inputs.T, error *
                                 self.derivation_of_sigmoid(output))

            # Dot product of input and adjustments then added to itself to adjust weight
            # If input is 0, weights aren't adjusted
            self.weights += adjustments

    def think(self, inputs):
        """
        Calculates output through given inputs
        """
        inputs = inputs.astype(float)

        # Dot product the input with weights then apply sigmoid function to normalize it between 0 and 1
        output = self.sigmoid(np.dot(inputs, self.weights))

        return output


if __name__ == '__main__':
    neural_network = NeuralNetwork()

    print("Random synaptic weights: ")
    print(neural_network.weights)

    # Inputs for testing
    training_inputs = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0],
                                [1, 1, 1]])

    # Outputs for testing (transposed so it becomes 4x1 matrix)
    training_outputs = np.array([[0, 0, 1, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100000)

    print("Synaptic weights after training: ")
    print(neural_network.weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))
