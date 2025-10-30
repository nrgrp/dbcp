import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# $k$-means Clustering""")
    return


@app.cell
def _():
    import os
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

    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    np.random.seed(10015)
    return BiconvexProblem, cp, make_blobs, mo, np, plt


@app.cell
def _(make_blobs):
    n = 2
    m = 1000
    k = 4
    centers = [[0, 2], [0, -2], [2, 0], [-2, 0]]
    xs, labels = make_blobs(n_samples=m, centers=centers, cluster_std=0.5)
    return k, m, n, xs


@app.cell
def _(BiconvexProblem, cp, k, m, n, xs):
    xbars = cp.Variable((k, n))
    zs = cp.Variable((m, k), nonneg=True)
    obj = cp.sum(cp.multiply(zs, cp.vstack([
        cp.sum(cp.square(xs - c), axis=1) for c in xbars
    ]).T))
    constr = [zs <= 1, cp.sum(zs, axis=1) == 1]
    prob = BiconvexProblem(cp.Minimize(obj), [[xbars], [zs]], constr)
    prob.solve()
    return xbars, zs


@app.cell
def _(np, plt, xbars, xs, zs):
    fig, axs = plt.subplots(1, 1, figsize=(3, 3))
    _labels = np.argmax(zs.value, axis=-1)
    cmap = plt.cm.get_cmap('tab10', np.unique(_labels).size)
    axs.scatter(xs[:, 0], xs[:, 1], s=10, c=_labels, cmap=cmap)
    axs.scatter(xbars.value[:, 0], xbars.value[:, 1], s=100, color='k', marker='x')
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')

    plt.show()
    fig.savefig('./figures/kmeans.pdf', bbox_inches='tight')
    return


if __name__ == "__main__":
    app.run()
