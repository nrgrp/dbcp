import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Value Iteration""")
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
def _(GridWorld, np, value_iteration, vi_policy):
    gridsize = 30
    wind = 0.3

    envr = GridWorld(size=gridsize, wind=wind)
    r = np.zeros(envr.num_states)
    r[envr.goals] += 100
    pi = vi_policy(envr.num_states, envr.num_actions, envr.P, r, envr.gamma, stochastic=True)
    v = value_iteration(r, envr.P, envr.num_states, envr.num_actions, envr.gamma)
    return envr, gridsize, pi, r, v


@app.cell
def _(BiconvexRelaxProblem, cp, envr, np, pi, r):
    vhat = cp.Variable(envr.num_states)
    pihat = cp.Variable((envr.num_states, envr.num_actions), nonneg=True)

    obj = cp.sum(cp.kl_div(pi, pihat))
    constr = [
        vhat == r + envr.gamma * cp.sum(cp.multiply(pi, cp.vstack([P @ vhat for P in np.moveaxis(envr.P, 2, 0)]).T), axis=1),
        cp.sum(pihat, axis=1) == 1,
        pihat <= 1
    ]
    prob = BiconvexRelaxProblem(cp.Minimize(obj), [[vhat], [pihat]], constr)
    prob.solve(cp.CLARABEL, nu=1e3)
    return pihat, vhat


@app.cell
def _(gridsize, plt, v):
    plt.imshow(v.reshape(gridsize, gridsize))
    return


@app.cell
def _(gridsize, plt, vhat):
    plt.imshow(vhat.value.reshape(gridsize, gridsize))
    return


@app.cell
def _(vhat):
    vhat.value
    return


@app.cell
def _(v):
    v
    return


@app.cell
def _(np, pihat):
    np.round(np.sum(pihat.value, axis=-1), 2)
    return


@app.cell
def _(np):
    def value_iteration(reward, P, num_states, num_actions, discount, threshold=1e-2):
        """
        calculate the optimal state value function of given enviroment.

        :param reward: reward vector. nparray. (states, )
        :param P: transition probability p(st | s, a). nparray. (states, states, actions).
        :param discount: discount rate gamma. float. Default: 0.99
        :param num_states: number of states. int.
        :param num_actions: number of actions. int.
        :param threshold: stop when difference smaller than threshold. float.
        :return: optimal state value function. nparray. (states)
        """

        v = np.zeros(num_states)

        while True:
            delta = 0

            for s in range(num_states):
                max_v = float("-inf")
                for a in range(num_actions):
                    tp = P[s, :, a]
                    max_v = max(max_v, np.dot(tp, (reward + discount * v)))

                diff = abs(v[s] - max_v)
                delta = max(delta, diff)

                v[s] = max_v

            if delta < threshold:
                break

        return v


    def vi_policy(num_states, num_actions, P, reward, discount, stochastic=True, threshold=1e-2):
        """
        Find the optimal policy.

        num_states: Number of states. int.
        num_actions: Number of actions. int.
        P: Function taking (state, action, state) to
            transition probabilities.
        reward: Vector of rewards for each state.
        discount: MDP discount factor. float.
        threshold: Convergence threshold, default 1e-2. float.
        stochastic: Whether the policy should be stochastic. Default True.
        -> Action probabilities for each state or action int for each state
            (depending on stochasticity).
        """

        v = value_iteration(reward, P, num_states, num_actions, discount, threshold)

        policy = np.zeros((num_states, num_actions))
        if stochastic:
            for s in range(num_states):
                for a in range(num_actions):
                    p = P[s, :, a]
                    policy[s, a] = p.dot(reward + discount*v)
            policy -= policy.max(axis=1).reshape((num_states, 1))  # For numerical stability.
            policy = np.exp(policy)/np.exp(policy).sum(axis=1).reshape((num_states, 1))

        else:
            def _policy(s):
                return max(range(num_actions),
                           key=lambda a: sum(P[s, k, a] *
                                             (reward[k] + discount * v[k])
                                             for k in range(num_states)))
            for s in range(num_states):
                policy[s, _policy(s)] = 1
        return policy
    return value_iteration, vi_policy


@app.cell
def _(np):
    class GridWorld:
        def __init__(self, size, wind, P=None):
            self.grid_size = size
            self.actions = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
            self.wind = wind
            self.gamma = 0.9
            self.goal_prob = 0.2

            self.num_states = self.grid_size ** 2
            self.num_actions = len(self.actions)

            # Calculate the transition probability matrix. (states, states, actions).
            if P is not None:
                self.P = P
            else:
                self.P = np.array(
                    [[[self._transition_probability(i, j, k)
                    for k in range(self.num_actions)]
                    for j in range(self.num_states)]
                    for i in range(self.num_states)])

            self.goals = []
            for s in range(self.num_states):
                if np.random.uniform() < self.goal_prob:
                    self.goals.append(s)

        def int_to_state(self, i):
            """
            Convert a state int into the corresponding coordinate.

            i: State int.
            -> (x, y) int tuple.
            """

            return i // self.grid_size, i % self.grid_size

        def state_to_int(self, p):
            """
            Convert a coordinate into the corresponding state int.

            p: (x, y) tuple.
            -> State int.
            """

            return p[0] * self.grid_size + p[1]

        def _is_neighbour(self, s1, s2):
            """
            Judge if state_1 and state_2 are neighbours.

            :param s1: state_1, int.
            :param s2: state_2, int.
            :return: if state_1 and state_2 are neighbours. bool.
            """
            x1, y1 = self.int_to_state(s1)
            x2, y2 = self.int_to_state(s2)

            return np.abs(x1 - x2) + np.abs(y1 - y2) <= 1

        def _is_corner(self, s):
            """
            Judge if a state is in the corner.

            :param s: state. int.
            :return: if the state is in the corner. bool.
            """
            x, y = self.int_to_state(s)
            return (x, y) in ((0, 0), (0, self.grid_size-1),
                              (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1))

        def _is_edge(self, s):
            """
            Judge if a state in on the edge (INCLUDING CORNER!).

            :param s: state. int.
            :return: if the state is on the edge. bool.
            """
            x, y = self.int_to_state(s)

            return (x in (0, self.grid_size-1)) or (y in (0, self.grid_size-1))

        def _is_in_the_world(self, x, y):
            """
            Judge if a stage is inside the enviroment.

            :param x: x coordinate of state. int.
            :param y: y coordinate of state. int.
            :return: if the state is inside the enviroment. bool.
            """

            return (0 <= x < self.grid_size) and (0 <= y < self.grid_size)

        def _transition_probability(self, s, st, a):
            """
            Calculate the transition probability p(st | s, a)

            :param s: initial state. int.
            :param st: target state. int.
            :param a: action to take. int.
            :return: transition probability. float
            """

            if not self._is_neighbour(s, st):
                return 0.0

            x, y = self.int_to_state(s)
            xt, yt = self.int_to_state(st)
            dx, dy = self.actions[a]

            if not self._is_edge(s):
                if (x+dx, y+dy) == (xt, yt):
                    return 1-self.wind + self.wind / self.num_actions
                else:
                    return self.wind / self.num_actions

            else:
                if self._is_corner(s):
                    if s == st:
                        if (dx, dy) == (0, 0) or not self._is_in_the_world(x+dx, y+dy):
                            return 1-self.wind + 3 * self.wind / self.num_actions
                        else:
                            return 3 * self.wind / self.num_actions
                    else:
                        if (x+dx, y+dy) == (xt, yt):
                            return 1-self.wind + self.wind / self.num_actions
                        else:
                            return self.wind / self.num_actions
                else:
                    if s == st:
                        if (dx, dy) == (0, 0) or not self._is_in_the_world(x+dx, y+dy):
                            return 1-self.wind + 2 * self.wind / self.num_actions
                        else:
                            return 2 * self.wind / self.num_actions
                    else:
                        if (x + dx, y + dy) == (xt, yt):
                            return 1 - self.wind + self.wind / self.num_actions
                        else:
                            return self.wind / self.num_actions
    return (GridWorld,)


if __name__ == "__main__":
    app.run()
