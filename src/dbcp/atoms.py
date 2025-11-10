import cvxpy as cp


def convolve(x, y):
    """Discrete convolution of two 1-D cvxpy expressions.

    Suppose :math:`x` and :math:`y` are 1-D cvxpy expressions of lengths
    :math:`m` and :math:`n`, respectively.
    This function returns a cvxpy expression :math:`c` of length :math:`m + n - 1`, where

    .. math::

        c_k = \\sum_{i + j = k} x_i y_j,\\quad k = 1, \\ldots, m + n - 1.

    Matches numpy.convolve for 1-D arrays.

    This function extends cvxpy.convolve atom to support the convolution
    operation between two cvxpy expressions.

    Parameters
    ----------
    x : cp.Expression
        A 1-D cvxpy expression.
    y : cp.Expression
        A 1-D cvxpy expression.

    Returns
    -------
    cp.Expression
        The convolution of x and y.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be 1-D cvxpy expressions.")

    c = [0] * (x.shape[0] + y.shape[0] - 1)
    for i, a in enumerate(y):
        for j, b in enumerate(x):
            c[i + j] += a * b
    return cp.hstack(c)
