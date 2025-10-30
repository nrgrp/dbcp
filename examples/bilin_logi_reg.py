import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Bilinear Logistic Regression""")
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from sklearn.datasets import make_classification
    from dbcp import BiconvexProblem

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["mathtext.fontset"] = 'cm'
    mpl.rcParams['font.family'] = ['sans-serif']

    np.random.seed(10015)
    return BiconvexProblem, cp, make_classification, mo


@app.cell
def _(make_classification):
    m = 300
    n = 20
    k = 10
    r = 5
    ninfo_frac = 0.9
    Xs, ys = make_classification(
        n_samples=m,
        n_features=n * k,
        n_informative=int(n * k * ninfo_frac),
        n_redundant=int(n * k * (1 - ninfo_frac))
    )
    Xs = Xs.reshape(m, n, k)
    return Xs, k, n, r, ys


@app.cell
def _(BiconvexProblem, Xs, cp, k, n, r, ys):
    U = cp.Variable((n, r))
    V = cp.Variable((k, r))

    obj = 0
    for _X, _y in zip(Xs, ys):
        obj += cp.sum(
            cp.multiply(_y, cp.trace(U.T @ _X @ V)) 
                - cp.logistic(cp.trace(U.T @ _X @ V))
        )
    prob = BiconvexProblem(cp.Maximize(obj), [[U], [V]])
    prob.solve(cp.CLARABEL, lbd=1, gap_tolerance=1e-4)
    return


if __name__ == "__main__":
    app.run()
