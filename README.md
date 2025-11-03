# DBCP: Disciplined Biconvex Programming

DBCP is an extension of [CVXPY](https://github.com/cvxpy/cvxpy)
for *biconvex optimization problems* in the form

$$
\begin{array}{ll}
    \text{minimize} & f_0(x, y)\\
    \text{subject to} & f_i(x, y) \leq 0,\quad i = 1, \ldots, m\\
    & h_i(x, y) = 0,\quad i = 1, \ldots, p,
\end{array}
$$

where $x \in X$, $y \in Y$ are the optimization variables.
The functions $f_0, \ldots, f_m$ are biconvex, meaning that for fixed $y$,
the functions $f_i(\cdot, y)$ are convex,
and for fixed $x$, the functions $f_i(x, \cdot)$ are convex.
The functions $h_1, \ldots, h_p$ are biaffine in a similar sense.
A more detailed discussion about biconvex optimization problems
can be found in our accompanying paper.

## Installation

### Using pip

You can install the package using pip:

```shell
pip install dbcp
```

### Development setup

We manage dependencies through [uv](https://docs.astral.sh/uv).
Once you have installed uv you can perform the following
commands to set up a development environment:

1. Clone the repository:

    ```shell
    git clone https://github.com/nrgrp/dbcp.git
    cd dbcp
    ```

2. Create a virtual environment and install dependencies:

    ```shell
    make install
    ```

This will:

- Create a Python 3.12 virtual environment.
- Install all dependencies from pyproject.toml.

## Usage

### DBCP syntax rule for multiplications

### Specifying biconvex problems

### Solving a biconvex problem

### Solving with infeasible starting points

### Verification of biconvexity

### Problem status

## Basic example

Suppose we are given a matrix $A \in \mathbf{R}^{m \times n}$.
Consider the following nonnegative matrix factorization problem:

$$
\begin{array}{ll}
    \text{minimize} & {\|XY + Z - A\|}_F\\
    \text{subject to} & X_{ij} \geq 0,\quad i = 1, \ldots, m,
        \quad j = 1, \ldots, k\\
    & Y_{ij} \geq 0,\quad i = 1, \ldots, k,\quad j = 1, \ldots, n\\
    & {\|Z\|}_F \leq 1,
\end{array}
$$

with variables $X \in \mathbf{R}^{m \times k}$,
$Y \in \mathbf{R}^{k \times n}$,
and $Z \in \mathbf{R}^{m \times n}$.

To specify and solve this problem using `dbcp`,
one may use the following code:

```python
import cvxpy as cp
import dbcp

X = cp.Variable((m, k), nonneg=True)
Y = cp.Variable((k, n), nonneg=True)
Z = cp.Variable((m, n))

obj = cp.Minimize(cp.norm(X @ Y + Z - A, 'fro'))
constraints = [cp.norm(Z, 'fro') <= 1]
prob = dbcp.BiconvexProblem(obj, [[X], [Y]], constraints)

prob.solve()
```

## Citation
