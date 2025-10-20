import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Fitting Input-output Hidden Markov Models""")
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from dbcp import BiconvexRelaxProblem

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["mathtext.fontset"] = 'cm'
    mpl.rcParams['font.family'] = ['sans-serif']

    np.random.seed(10015)
    return BiconvexRelaxProblem, cp, mo, np, plt


@app.cell
def _(np):
    m = 1000
    n = 2
    K = 3
    coefs = np.array([[-2, 0], [2, 6], [2, -6]])
    p_tr = np.array([[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]])

    xs = np.random.uniform(-5, 5, m)
    xs = np.vstack([xs, np.ones(m)]).T

    ys = np.zeros(m)
    labels = np.zeros(m, dtype=int)

    _s = 0
    for _i, _feat in enumerate(xs):
        ys[_i] = 1 if np.random.uniform() < 1 / (1 + np.exp(-_feat @ coefs[_s])) else 0
        labels[_i] = _s
        _s = np.random.choice(K, p=p_tr[_s])
    return K, coefs, labels, m, n, xs, ys


@app.cell
def _(BiconvexRelaxProblem, K, cp, m, n, xs, ys):
    thetas = []
    r = []
    for k in range(K):
        thetas.append(cp.Variable(n))
        r.append(-(cp.multiply(ys, xs @ thetas[-1]) - cp.logistic(xs @ thetas[-1])))

    z = cp.Variable((m, K), nonneg=True)
    obj = cp.sum(cp.multiply(z, cp.vstack(r).T)) + 0.1 * cp.sum(cp.norm2(cp.vstack(thetas), axis=1)) + 1 * cp.sum(cp.kl_div(z[:-1], z[1:]))
    constr = [
        thetas[0][0] <= 0,
        thetas[1][0] >= 0,
        thetas[2][0] >= 0,
        thetas[1][1] >= thetas[2][1],
        z >= 0, z <= 1, cp.sum(z, axis=1) == 1
    ]

    prob = BiconvexRelaxProblem(cp.Minimize(obj), ([z], thetas), constr)
    prob.solve(solver=cp.CLARABEL, nu=1e3, lbd=1, gap_tolerance=1e-3)
    return thetas, z


@app.cell
def _(K, coefs, labels, m, np, plt, thetas, z):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), width_ratios=(1.2, 1))

    axs[0].plot(labels, linestyle='dashed', color='k', linewidth=1, zorder=10)
    axs[0].plot(np.argmax(z.value, axis=-1), color='r', linewidth=2)

    inputs = np.linspace(-5, 5, m)
    inputs = np.vstack([inputs, np.ones(m)]).T
    for _i in range(K):
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ coefs[_i])),
                    linestyle='dashed', color='k', zorder=10)
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ thetas[_i].value)))

    axs[0].set_xlabel('$t$')
    axs[0].set_ylabel('latent factor')
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels([1, 2])

    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$1/(1 + \exp(-x^T \theta))$', fontsize=15)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
