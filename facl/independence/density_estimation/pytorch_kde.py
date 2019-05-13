import torch
from math import pi, sqrt


class kde:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.

    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_train):
        n, d = x_train.shape

        self.n = n
        self.d = d

        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.std = self.bandwidth

        self.train_x = x_train

    def pdf(self, x):
        s = x.shape
        d = s[-1]
        s = s[:-1]
        assert d == self.d

        data = x.unsqueeze(-2)

        train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        pdf_values = (
                         torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
                     ).mean(dim=-1) / sqrt(2 * pi) / self.bandwidth

        return pdf_values


def _unsqueeze_multiple_times(input, axis, times):
    """
    Utils function to unsqueeze tensor to avoid cumbersome code
    :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
    :param axis: the axis to unsqueeze repeatedly
    :param times: the number of repetitions of the unsqueeze
    :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
    """
    output = input
    for i in range(times):
        output = output.unsqueeze(axis)
    return output
