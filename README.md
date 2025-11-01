# DBCP: Disciplined Biconvex Programming

DBCP is an extension of [CVXPY](https://github.com/cvxpy/cvxpy) for *biconvex optimization problems*.
A biconvex optimization problem has the form of

$$
\begin{array}{ll}
    \text{minimize} & f_0(x, y)\\
    \text{subject to} & f_i(x, y) \leq 0,\quad i = 1, \ldots, m\\
    & h_i(x, y) = 0,\quad i = 1, \ldots, p,
\end{array}
$$

where $x \in X$, $y \in Y$ are the optimization variables.
The functions $f_0, \ldots, f_m$ are biconvex, meaning that for fixed $y$, the functions $f_i(\cdot, y)$ are convex, and for fixed $x$, the functions $f_i(x, \cdot)$ are convex.
The functions $h_1, \ldots, h_p$ are biaffine in a similar sense.
A more detailed discussion about biconvex optimization problems can be found in our accompanying paper.

## Installation

## Usage

### DBCP syntax rule for multiplications

### Specifying biconvex problems

### Solving a biconvex problem

### Solving with infeasible starting points

### Verification of biconvexity

## Basic example

## Citation
