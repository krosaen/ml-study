"""
Implementation of ADAptive LInear NEuron (Adaline) algorithm from Chapter 2 of
"Python Machine Learning"
"""
import numpy as np


def train_adeline(observations, labels, learning_rate=0.0001, max_training_iterations=100):
    """
    Trains a (binary) perceptron, returning a function that can predict / classify
    given a new observations, as well as insight into how the training progressed via
    a log of the weights and number of errors for each iteration.

    :param observations: array of rows
    :param labels: correct label classification for each row: [1, -1, 1, 1, ...]
    :param learning_rate: how fast to update weights
    :param max_training_iterations: max number of times to iterate through observations
    :return: (prediction_fn, weights_log, errors_log)
    """
    the_weights = np.zeros(1 + observations.shape[1])
    weights_log = []
    num_errors_log = []

    def net_input(observations, weights):
        return np.dot(observations, weights[1:]) + weights[0]

    def activated_output(output):
        return np.where(output >= 0.0, 1, -1)

    def predict(observations, weights=the_weights):
        return activated_output(net_input(observations, weights))

    for _ in range(max_training_iterations):
        weights_log.append(np.copy(the_weights))
        raw_outputs = net_input(observations, the_weights)
        errors = labels - raw_outputs
        weight_deltas = learning_rate * np.dot(observations.transpose(), errors)
        the_weights[1:] += weight_deltas
        the_weights[0] += learning_rate * np.sum(errors)

        num_errors = (activated_output(raw_outputs) != labels).sum()
        num_errors_log.append(num_errors)
        if num_errors == 0:
            break

    return predict, weights_log, num_errors_log
