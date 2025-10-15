from collections.abc import Iterable

import numpy as np
import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from dbcp.fix import fix_prob


class BiconvexProblem(cp.Problem):
    def __init__(
            self,
            biconvex_objective,
            fix_vars: tuple[Iterable[cp.Variable], Iterable[cp.Variable]],
            constraints: list[Constraint] | None = None,
    ) -> None:
        super().__init__(biconvex_objective, constraints)
        self.fix_vars = fix_vars

        self._x_prob = fix_prob(self, self.fix_vars[1])
        self._y_prob = fix_prob(self, self.fix_vars[0])

        self._value: float | None = None
        self._status: str | None = None

    @property
    def x_prob(self) -> cp.Problem:
        return self._x_prob

    @property
    def y_prob(self) -> cp.Problem:
        return self._y_prob

    @property
    def x_prob_(self) -> cp.Problem:
        for p in self._x_prob.parameters():
            var = [v for v in self.fix_vars[1] if v.id == p.id][0]
            if var.value is not None:
                p.project_and_assign(var.value)
        return self._x_prob

    @property
    def y_prob_(self) -> cp.Problem:
        for p in self._y_prob.parameters():
            var = [v for v in self.fix_vars[0] if v.id == p.id][0]
            if var.value is not None:
                p.project_and_assign(var.value)
        return self._y_prob

    def _initialize(self, max_iter=10) -> None:
        for v in self.variables():
            v.project_and_assign(np.random.standard_normal(v.shape))

    def solve(self,
              solver: str = cp.CLARABEL,
              lbd: float = 0.5,
              max_iter: float = 100,
              eps: float = 1e-6,
              *args, **kwargs
              ) -> float | None:
        print(f"{' DBCP Summary ':=^{65}}")
        print(f"{'iter':<7} {'xcost':<20} {'ycost':<20} {'gap':<10}")
        print("-" * 65)
        prox_params = [cp.Parameter(v.shape, id=v.id, **v.attributes) for v in self.variables()]
        x_prox = cp.Problem(cp.Minimize(cp.multiply(lbd, cp.sum([
            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
            for v in self.x_prob.variables()
        ]))))
        y_prox = cp.Problem(cp.Minimize(cp.multiply(lbd, cp.sum([
            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
            for v in self.y_prob.variables()
        ]))))
        i = 0
        while True:
            for v in self.x_prob_.variables():
                [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
            if self.objective.NAME == "minimize":
                (self.x_prob_ + x_prox).solve(solver=solver, *args, **kwargs)
            else:
                (self.x_prob_ - x_prox).solve(solver=solver, *args, **kwargs)
            for v in self.y_prob_.variables():
                [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
            if self.objective.NAME == "minimize":
                (self.y_prob_ + y_prox).solve(solver=solver, *args, **kwargs)
            else:
                (self.y_prob_ - y_prox).solve(solver=solver, *args, **kwargs)

            gap = np.abs(self.x_prob.objective.value - self.y_prob.objective.value)
            print(
                f"{i:<7} {self.x_prob.objective.value:<20.9f} {self.y_prob.objective.value:<20.9f} {gap:<10.9f}")
            if gap < eps:
                self._status = "converge"
                break
            else:
                i += 1
            if i == max_iter:
                self._status = "converge_inaccurate"
                break
        print("-" * 65)
        print(f"Terminated with status: {self.status}")
        print("=" * 65)
        self._value = self.y_prob.objective.value
        return self.value

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def value(self) -> float | None:
        return self._value

    def is_dbcp(self) -> bool:
        if self.x_prob.is_dcp() and self.y_prob.is_dcp():
            return True
        return False
