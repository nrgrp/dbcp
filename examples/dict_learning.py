import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Sparse Dictionary Learning""")
    return


@app.cell
def _():
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

    np.random.seed(10015)
    return BiconvexProblem, cp, mo, np, plt


@app.cell
def _(BiconvexProblem, cp, np):
    m = 10
    n = 20
    k = 20

    X = np.random.randn(m, n)
    D = cp.Variable((m, k))
    Y = cp.Variable((k, n))
    alpha = cp.Parameter(nonneg=True)
    obj = cp.Minimize(cp.sum_squares(D @ Y - X) / 2 + alpha * cp.norm1(Y))
    prob = BiconvexProblem(obj, [[D], [Y]], [cp.norm(D,'fro') <= 1])

    errs = []
    cards = []
    for _a in np.logspace(-5, 0, 50):
        alpha.value = _a
        D.value = None
        Y.value = None
        prob.solve(cp.CLARABEL, gap_tolerance=1e-1)
        errs.append(cp.norm(D @ Y - X, 'fro').value / cp.norm(X, 'fro').value)
        cards.append(cp.sum(cp.abs(Y).value >= 1e-3).value)
    return cards, errs


@app.cell
def _(cards, errs, plt):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    axs.plot(cards, errs, marker='.', color='k')
    axs.set_xlabel(r'$\mathop{\bf card} Y$')
    axs.set_ylabel('$||DY-X||_F/||X||_F$')
    return


if __name__ == "__main__":
    app.run()
