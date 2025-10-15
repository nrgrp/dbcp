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
        for p in self._x_prob.parameters():
            var = [v for v in self.fix_vars[1] if v.id == p.id][0]
            if var.value is not None:
                p.project_and_assign(var.value)
        return self._x_prob

    @property
    def y_prob(self) -> cp.Problem:
        for p in self._y_prob.parameters():
            var = [v for v in self.fix_vars[0] if v.id == p.id][0]
            if var.value is not None:
                p.project_and_assign(var.value)
        return self._y_prob

    def initialize(self, max_iter=10) -> None:
        for v in self.variables():
            v.project_and_assign(np.random.standard_normal(v.shape))

    def solve(self, solver=cp.CLARABEL, max_iter=100, eps=1e-6, *args, **kwargs) -> float:
        for i in range(max_iter):
            self.x_prob.solve(solver=solver, *args, **kwargs)
            self.y_prob.solve(solver=solver, *args, **kwargs)
            print(f"Iter {i}: x_prob value = {self._x_prob.value}, y_prob value = {self._y_prob.value}")
            if np.abs(self._x_prob.value - self._y_prob.value) < eps:
                break
        return self.value

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def value(self) -> float | None:
        """
        Returns the value of the objective function at the solution, and None if the problem has not
        been successfully solved.
        """
        return self._value

    def is_dbcp(self) -> bool:
        if self.x_prob.is_dcp() and self.y_prob.is_dcp():
            return True
        return False
