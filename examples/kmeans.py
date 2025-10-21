import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# $k$-means Clustering""")
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from sklearn.datasets import make_blobs
    from dbcp import BiconvexProblem

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["mathtext.fontset"] = 'cm'
    mpl.rcParams['font.family'] = ['sans-serif']

    np.random.seed(10015)
    return BiconvexProblem, cp, make_blobs, mo, plt


@app.cell
def _(make_blobs):
    n = 2
    m = 1000
    K = 4
    centers = [[0, 2], [0, -2], [2, 0], [-2, 0]]
    xs, labels = make_blobs(n_samples=m, centers=centers, cluster_std=0.5)
    return K, m, n, xs


@app.cell
def _(BiconvexProblem, K, cp, m, n, xs):
    xbars = cp.Variable((K, n))
    z = cp.Variable((m, K), nonneg=True)
    obj = cp.sum(cp.multiply(z, cp.vstack([cp.sum(cp.square(xs - c), axis=1) for c in xbars]).T))
    constr = [z >= 0, z <= 1, cp.sum(z, axis=1) == 1]
    prob = BiconvexProblem(cp.Minimize(obj), [[xbars], [z]], constr)
    prob.solve()
    return (xbars,)


@app.cell
def _(plt, xbars, xs):
    fig, axs = plt.subplots(1, 1, figsize=(3, 3))
    axs.scatter(xs[:, 0], xs[:, 1], s=10, color='k')
    axs.scatter(xbars.value[:, 0], xbars.value[:, 1], s=100, color='r', marker='x')
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')
    return


if __name__ == "__main__":
    app.run()
