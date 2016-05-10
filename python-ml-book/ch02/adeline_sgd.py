"""
Implementation of ADAptive LInear NEuron (Adaline) algorithm from Chapter 2 of
"Python Machine Learning" adapted to use stochastic gradient descent instead of batch.
"""
import numpy as np


def train_adeline_sgd(observations, labels,
                      learning_rate=0.01, max_training_iterations=100,
                      shuffle=True):
    """
    Trains a (binary) perceptron, returning a function that can predict / classify
    given a new observations, as well as insight into how the training progressed via
    a log of the weights and squared errors for each iteration.

    :param observations: array of rows
    :param labels: correct label classification for each row: [1, -1, 1, 1, ...]
    :param learning_rate: how fast to update weights
    :param max_training_iterations: max number of times to iterate through observations
    :return: (prediction_fn, num_errors_log, final_weights)
    """
    the_weights = np.zeros(1 + observations.shape[1])
    num_errors_log = []

    def net_input(observations, weights):
        return np.dot(observations, weights[1:]) + weights[0]

    def quantized_output(output):
        return np.where(output >= 0.0, 1, -1)

    def predict(observations, weights=the_weights):
        return quantized_output(net_input(observations, weights))

    idxes = list(range(len(observations)))

    for _ in range(max_training_iterations):
        num_errors = 0
        if shuffle:
            np.random.shuffle(idxes)
        # for observation, correct_output in zip(observations, labels):
        for idx in idxes:
            observation = observations[idx]
            correct_output = labels[idx]
            raw_output = net_input(observation, the_weights)
            if quantized_output(raw_output) != correct_output:
                num_errors += 1
                weight_delta = learning_rate * (correct_output - raw_output)
                the_weights[1:] += weight_delta * observation
                the_weights[0] += weight_delta

        num_errors_log.append(num_errors)
        if num_errors == 0:
            break

    return predict, num_errors_log, the_weights

