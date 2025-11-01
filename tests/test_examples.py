import pytest
import numpy as np
import cvxpy as cp
from dbcp import BiconvexProblem, BiconvexRelaxProblem

np.random.seed(10015)


def test_nmf():
    # Generate data matrix A
    m = 5
    n = 10
    k = 5
    A = np.random.rand(m, k).dot(np.random.rand(k, n))
    X = cp.Variable((m, k), nonneg=True)
    Y = cp.Variable((k, n), nonneg=True)

    # Define the biconvex problem
    obj = cp.Minimize(cp.sum_squares(X @ Y - A))
    prob = BiconvexProblem(obj, [[X], [Y]])
    prob.solve()

    # Check that the solution is non-negative
    assert np.all(X.value >= 0)
    assert np.all(Y.value >= 0)

    # Check that the reconstruction error is reasonable
    reconstruction_error = np.linalg.norm(A - X.value @ Y.value, 'fro') ** 2
    assert reconstruction_error < 1e-1
