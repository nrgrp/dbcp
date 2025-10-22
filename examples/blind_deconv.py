import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
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

    np.random.seed(10015)
    return BiconvexProblem, cp, np, plt


@app.cell
def _(BiconvexProblem, conv, cp, np):
    n = 120
    m = 40

    x0 = np.zeros(n)
    x0[6] = 1
    y0 = np.exp(-np.square(np.linspace(-2, 2, m)) * 2)
    d = np.convolve(x0, y0)

    y = cp.Variable(m, nonneg=True)
    x = cp.Variable(n, nonneg=True)
    obj = cp.sum_squares(conv(x, y) - d) + 0.1 * cp.norm1(x) + 0.2 * cp.sum_squares(cp.diff(y))
    constr = [cp.norm(y, "inf") <= 1]
    prob = BiconvexProblem(cp.Minimize(obj), [[x], [y]], constr)
    prob.solve(cp.CLARABEL, gap_tolerance=1e-5, max_iter=200)
    return d, x, x0, y, y0


@app.cell
def _(d, np, plt, x, x0, y, y0):
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(x0, linestyle='--', color='C3', linewidth=2)
    axs.plot(y0, linestyle='--', color='C1', linewidth=2)
    axs.plot(d, linestyle='--', color='k', linewidth=2)
    axs.plot(x.value, color='C0', marker='.', markersize=10)
    axs.plot(y.value, color='C2', marker='s')
    axs.plot(np.convolve(x.value, y.value), marker='D', color='C4', zorder=-1)

    axs.legend([
        "ground truth $x_0$",
        "ground truth $y_0$",
        "ground truth $d$",
        "recovered $x_0$",
        "recovered $y_0$",
        "recovered $d$"
    ], frameon=False, fontsize=12)
    axs.set_xlim(0, 60)
    axs.set_xlabel("indices")
    plt.show()
    return


@app.cell(hide_code=True)
def _(cp):
    def conv(x, y):
        c = [0] * (x.shape[0] + y.shape[0] - 1)
        for _i, _a in enumerate(y):
            for _j, _b in enumerate(x):
                c[_i + _j] += _a * _b
        return cp.hstack(c)
    return (conv,)


if __name__ == "__main__":
    app.run()
