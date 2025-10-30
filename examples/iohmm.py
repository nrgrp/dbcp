import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Fitting Input-output Hidden Markov Models""")
    return


@app.cell
def _():
    import os
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

    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    np.random.seed(10015)
    return BiconvexRelaxProblem, cp, mo, np, plt


@app.cell
def _(np):
    m = 1800
    n = 2
    K = 3
    coefs = np.array([[-1, 0], [2, 6], [2, -6]])
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
    return K, coefs, labels, m, n, p_tr, xs, ys


@app.cell
def _(BiconvexRelaxProblem, K, cp, m, n, xs, ys):
    thetas = cp.Variable((K, n))
    zs = cp.Variable((m, K), nonneg=True)

    alpha_theta = 0.1
    alpha_z = 2

    rs = [
        -cp.multiply(ys, xs @ thetas[k]) + cp.logistic(xs @ thetas[k])
        for k in range(K)
    ]
    obj = cp.Minimize(
        cp.sum(cp.multiply(zs, cp.vstack(rs).T))
        + alpha_theta * cp.sum_squares(thetas)
        + alpha_z * cp.sum(cp.kl_div(zs[:-1], zs[1:])))
    constr = [
        thetas[0][0] <= 0,
        thetas[1][0] >= 0,
        thetas[2][0] >= 0,
        thetas[1][1] >= thetas[2][1],
        zs <= 1, cp.sum(zs, axis=1) == 1
    ]

    prob = BiconvexRelaxProblem(obj, ([zs], [thetas]), constr)
    prob.solve(solver=cp.CLARABEL, nu=1e2, lbd=0.1, gap_tolerance=1e-3)
    return thetas, zs


@app.cell
def _(K, coefs, labels, m, np, plt, thetas, zs):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), width_ratios=(1.2, 1))

    axs[0].plot(labels, linestyle='dashed', color='k', linewidth=1, zorder=10)
    axs[0].plot(np.argmax(zs.value, axis=-1), color='r', linewidth=2)

    inputs = np.linspace(-5, 5, m)
    inputs = np.vstack([inputs, np.ones(m)]).T
    for _i in range(K):
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ coefs[_i])),
                    linestyle='dashed', color='k', zorder=10)
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ thetas[_i].value)))

    axs[0].set_xlabel('$t$')
    axs[0].set_ylabel('state')
    axs[0].set_yticks([0, 1, 2])
    axs[0].set_yticklabels([1, 2, 3])

    axs[1].set_xlabel(r'$x_1$')
    axs[1].set_ylabel(r'$1/(1 + \exp(-x^T \theta))$', fontsize=15)

    plt.tight_layout()
    plt.show()
    fig.savefig('./figures/iohmm.pdf', bbox_inches='tight')
    return


@app.cell
def _(K, m, np, p_tr, zs):
    p_tr_hat = np.zeros_like(p_tr)
    z_hat = np.argmax(zs.value, axis=-1)
    for zi in range(K):
        z_idx = np.where(z_hat == zi)[0]
        z_idx = np.delete(z_idx, np.where(z_idx == m - 1)[0])
        _, nz_num = np.unique(z_hat[z_idx + 1], return_counts=True)
        p_tr_hat[zi] = nz_num / len(z_idx)

    print(p_tr_hat)
    return


if __name__ == "__main__":
    app.run()
