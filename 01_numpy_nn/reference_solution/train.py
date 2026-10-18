"""NumPy-only XOR network: correct gradients, numerical check and a sign bug."""
from pathlib import Path
import json

import numpy as np

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([[0.], [1.], [1.], [0.]])


def initialize(seed=23):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, .5, (2, 4)), np.zeros((1, 4)),
            rng.normal(0, .5, (4, 1)), np.zeros((1, 1))]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def forward(x, parameters):
    w1, b1, w2, b2 = parameters
    hidden = sigmoid(x @ w1 + b1)
    return hidden, sigmoid(hidden @ w2 + b2)


def loss(y, prediction):
    p = np.clip(prediction, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradients(x, y, parameters):
    hidden, prediction = forward(x, parameters)
    dz2 = (prediction - y) / len(x)
    dz1 = (dz2 @ parameters[2].T) * hidden * (1 - hidden)
    return [x.T @ dz1, dz1.sum(axis=0, keepdims=True),
            hidden.T @ dz2, dz2.sum(axis=0, keepdims=True)]


def gradient_check(parameters, epsilon=1e-5):
    """Central differences of the same mean loss used by backpropagation."""
    analytic = gradients(X, Y, parameters)
    max_error = 0.
    for array, derivative in zip(parameters, analytic):
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + epsilon
            plus = loss(Y, forward(X, parameters)[1])
            array[index] = original - epsilon
            minus = loss(Y, forward(X, parameters)[1])
            array[index] = original
            max_error = max(max_error, abs((plus - minus) / (2 * epsilon) - derivative[index]))
    return float(max_error)


def train(initial, reverse=False, epochs=10_000, lr=1.):
    parameters = [a.copy() for a in initial]
    history = []
    for _ in range(epochs):
        history.append(loss(Y, forward(X, parameters)[1]))
        derivatives = gradients(X, Y, parameters)
        for value, derivative in zip(parameters, derivatives):
            value += (1 if reverse else -1) * lr * derivative
    history.append(loss(Y, forward(X, parameters)[1]))
    return parameters, history


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    initial = initialize()
    error = gradient_check(initial)
    if error > 1e-7:
        raise AssertionError(f'Gradient check failed: {error}')
    correct, right_curve = train(initial)
    _, wrong_curve = train(initial, reverse=True)
    probabilities = forward(X, correct)[1]
    classes = (probabilities >= .5).astype(int).ravel()
    if not np.array_equal(classes, Y.ravel()):
        raise AssertionError('The reference network did not learn XOR.')
    out = Path(__file__).resolve().parents[1] / 'outputs'
    out.mkdir(exist_ok=True)
    plt.plot(right_curve, label='Gradient descent')
    plt.plot(wrong_curve, label='Wrong update sign')
    plt.xlabel('Update'); plt.ylabel('Binary cross-entropy'); plt.legend()
    plt.savefig(out / 'loss_comparison.png', dpi=150); plt.close()
    result = {'gradient_max_abs_error': error, 'initial_loss': right_curve[0],
              'correct_final_loss': right_curve[-1], 'wrong_final_loss': wrong_curve[-1],
              'probabilities': probabilities.ravel().tolist(), 'classes': classes.tolist()}
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
