"""
Implementation of perceptron algorithm from Chapter 2 of "Python Machine Learning"
"""
import numpy as np


def train_perceptron(observations, labels, learning_rate=0.1, max_training_iterations=10):
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
    errors_log = []

    def net_input(observation, weights):
        return np.dot(observation, weights[1:]) + weights[0]

    def predict(observation, weights=the_weights):
        return np.where(net_input(observation, weights) >= 0.0, 1, -1)

    for _ in range(max_training_iterations):
        errors = 0
        weights_log.append(np.copy(the_weights))
        for observation, correct_output in zip(observations, labels):
            weight_delta = learning_rate * (correct_output - predict(observation))
            the_weights[1:] += weight_delta * observation
            the_weights[0] += weight_delta
            errors += int(weight_delta != 0.0)
        errors_log.append(errors)
        if errors == 0:
            break

    return predict, weights_log, errors_log
