#!/usr/bin/python
# pylint: disable=no-member
# pylint: disable=unused-variable
# pylint: disable=old-style-class
# pylint: disable=too-many-locals
"""
This combines both Patrick Thomas'
run_neural_net.py and train_neural_net.py
into a container (class) that can do
the functions of both training and running
a FANN neural network.
"""
__author__ = "Patrick Thomas"
__version__ = "1.0"
__date__ = "12/16/15"

from pyfann import libfann
#Miscellaneous parameters about the neural network
#How many connections will be in the neural network
#1 = fully connected, 0.5 = half connected
CONNECTION_RATE = 1.0
#How fast the neural net learns the data set
LEARNING_RATE = 0.7

DESIRED_ERROR = 0.0001
MAX_ITERATIONS = 10000
ITERATIONS_BETWEEN_REPORTS = 10000

class NeuralNetContainer():
    """
    Container of the neural network and common tasks for the ANN like training
    and getting actual output from the ANN
    """
    def __init__(self):
        self.ann = libfann.neural_net()
    def load_from_file(self, net_file):
        """
        Creates the ANN from the specified file at the path net_file
        """
        self.ann.create_from_file(net_file)
    def train(self, training_file, net_file, num_input, num_neurons_hidden, num_output):
        """
        Creates an ANN, specifies parameters for the ANN, and trains it to
        a file exported by swvgsleaf.LeafContainer
        """
        ann = libfann.neural_net()
        ann.create_sparse_array(CONNECTION_RATE, (num_input, num_neurons_hidden, num_output))
        ann.set_learning_rate(LEARNING_RATE)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)
        ann.train_on_file(training_file, MAX_ITERATIONS, ITERATIONS_BETWEEN_REPORTS, DESIRED_ERROR)
        ann.save(net_file)
        ann.destroy()#frees memory associated with the ANN
        #loads the ANN from the net_file
        self.load_from_file(net_file)
    def run_numbers(self, numbers):
        """
        "Runs" the data in numbers through the artificial neural network,
        in order to get the ANN's guess on what the species of the tree
        represented by numbers is.
        """
        return self.ann.run(numbers)
#0.1132897603,0.1568627451,0.1590413943,0.1154684096,0.0087145969