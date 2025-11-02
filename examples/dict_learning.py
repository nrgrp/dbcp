import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sparse Dictionary Learning
    """)
    return


@app.cell
def _():
    import os
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from dbcp import BiconvexProblem

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["mathtext.fontset"] = 'cm'
    mpl.rcParams['font.family'] = ['sans-serif']

    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    np.random.seed(10015)
    return BiconvexProblem, cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    We consider the sparse dictionary learning problem, which aims to find a dictionary matrix $D \in \mathbf{R}^{m \times k}$ and a sparse code matrix $X \in \mathbf{R}^{k \times n}$, such that the data matrix $Y \in \mathbf{R}^{m \times n}$ can be well approximated by their product $DX$, while the matrix $X$ is sparse and the matrix $D$ has bounded Frobenius norm.
    The dictionary learning problem can be formulated as the following biconvex optimization problem:

    \[
        \begin{array}{ll}
            \text{minimize} & {\|DX - Y\|}_F^2 + \alpha {\|X\|}_1\\
            \text{subject to} & {\|D\|}_F \leq \beta
        \end{array}
    \]

    with variables $D$ and $X$, where $\alpha > 0$ is the sparsity regularization parameter, and $\beta > 0$ is the bound on the Frobenius norm of the dictionary matrix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate problem data
    """)
    return


@app.cell
def _(np):
    m = 10
    n = 20
    k = 20
    beta = 1

    Y = np.random.randn(m, n)
    return Y, beta, k, m, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the problem
    """)
    return


@app.cell
def _(BiconvexProblem, Y, beta, cp, k, m, n, np):
    D = cp.Variable((m, k))
    X = cp.Variable((k, n))
    alpha = cp.Parameter(nonneg=True)
    obj = cp.Minimize(cp.sum_squares(D @ X - Y) + alpha * cp.norm1(X))
    prob = BiconvexProblem(obj, [[D], [X]], [cp.norm(D,'fro') <= beta])

    errs = []
    cards = []
    for _a in np.logspace(-5, 0, 50):
        alpha.value = _a
        D.value = None
        X.value = None
        prob.solve(cp.CLARABEL, gap_tolerance=1e-1)
        errs.append(cp.norm(D @ X - Y, 'fro').value / cp.norm(Y, 'fro').value)
        cards.append(cp.sum(cp.abs(X).value >= 1e-3).value)
    return cards, errs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(cards, errs, plt):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    axs.plot(cards, errs, marker='.', color='k')
    axs.set_xlabel(r'$\mathop{\bf card} Y$')
    axs.set_ylabel('$||DX-Y||_F/||Y||_F$')

    plt.show()
    fig.savefig('./figures/dict_learning.pdf', bbox_inches='tight')
    return


if __name__ == "__main__":
    app.run()
