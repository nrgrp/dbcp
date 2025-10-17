import warnings
from collections.abc import Iterable

import numpy as np
import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from dbcp.fix import fix_prob
from dbcp.transform import relax_with_slack
from dbcp.error import InitiationError, SolveError


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
            if p.id in [v.id for v in self.fix_vars[1]]:
                var = [v for v in self.fix_vars[1] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._x_prob

    @property
    def y_prob_(self) -> cp.Problem:
        for p in self._y_prob.parameters():
            if p.id in [v.id for v in self.fix_vars[0]]:
                var = [v for v in self.fix_vars[0] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._y_prob

    def _project(self, solver, proj_max_iter) -> None:
        print("Initiation start...")
        for v in self.variables():
            if v.value is None:
                v.project_and_assign(np.random.standard_normal(v.shape))
        if all([c.value() for c in self.constraints]):
            print("All constraints satisfied.")
        else:
            print("Finding a feasible initial point...")
            print("-" * 85)
            print(f"{'iter':<7} {'residual':<20}")
            print("-" * 65)
            proj_prob, _ = relax_with_slack(self)
            xproj_prob = fix_prob(proj_prob, self.fix_vars[1])
            yproj_prob = fix_prob(proj_prob, self.fix_vars[0])
            i = 0
            while True:
                for p in xproj_prob.parameters():
                    p.project_and_assign([v for v in self.fix_vars[1] if v.id == p.id][0].value)
                xproj_prob.solve(solver=solver)
                for p in yproj_prob.parameters():
                    p.project_and_assign([v for v in self.fix_vars[0] if v.id == p.id][0].value)
                yproj_prob.solve(solver=solver)

                print(
                    f"{i:<7} {yproj_prob.value:<20.9f}")
                if all([c.value() for c in self.constraints]):
                    print("-" * 65)
                    print(f'Found feasible point in {i + 1} iterations.')
                    break
                else:
                    i += 1
                if i == proj_max_iter:
                    raise InitiationError("Cannot find a feasible point. Try different initial values.")

    def solve(self,
              solver: str = cp.SCS,
              lbd: float = 0.1,
              max_iter: float = 100,
              gap_tolerance: float = 1e-6,
              *args, **kwargs
              ) -> float | None:
        print(f"{' DBCP Summary ':=^{85}}")
        if self.status is None:
            self._project(solver, kwargs.get('proj_max_iter', 10))
        else:
            print("Warm start with previous solution.")

        print(f"Block coordinate descent start with solver {solver}...")
        print("-" * 65)
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
        if self.objective.NAME == "minimize":
            xprox_prob = self.x_prob + x_prox
            yprox_prob = self.y_prob + y_prox
        else:
            xprox_prob = self.x_prob - x_prox
            yprox_prob = self.y_prob - y_prox
        i = 0
        try:
            while True:
                for v in self.x_prob_.variables():
                    [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                xprox_prob.solve(solver=solver, *args, **kwargs)
                for v in self.y_prob_.variables():
                    [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                yprox_prob.solve(solver=solver, *args, **kwargs)

                if ((xprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) or
                        (yprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))):
                    raise SolveError(f"Solver {solver} failed. Try a different solver.")
                gap = np.abs(self.x_prob.objective.value - self.y_prob.objective.value)
                print(
                    f"{i:<7} {self.x_prob.objective.value:<20.9f} {self.y_prob.objective.value:<20.9f} {gap:<10.9f}")
                if gap < gap_tolerance:
                    self._status = "converge"
                    break
                else:
                    i += 1
                if i == max_iter:
                    self._status = "converge_inaccurate"
                    break
        except cp.SolverError as e:
            raise SolveError("Solver failed. Try with larger 'lbd' value.") from e

        print("-" * 65)
        print(f"Terminated with status: {self.status}.")
        print("=" * 85)
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


class BiconvexRelaxProblem(cp.Problem):
    def __init__(
            self,
            biconvex_objective,
            fix_vars: tuple[Iterable[cp.Variable], Iterable[cp.Variable]],
            constraints: list[Constraint] | None = None,
    ) -> None:
        super().__init__(biconvex_objective, constraints)
        self.fix_vars = fix_vars

        self.nu = cp.Parameter((), nonneg=True)
        self._rlx_prob, self.slack_vars = relax_with_slack(self, self.nu)
        self._x_prob = fix_prob(self._rlx_prob, self.fix_vars[1])
        self._y_prob = fix_prob(self._rlx_prob, self.fix_vars[0])

        self._value: float | None = None
        self._status: str | None = None

    @property
    def x_prob(self) -> cp.Problem:
        return self._x_prob

    @property
    def y_prob(self) -> cp.Problem:
        return self._y_prob

    @property
    def rlx_prob(self) -> cp.Problem:
        return self._rlx_prob

    @property
    def x_prob_(self) -> cp.Problem:
        for p in self._x_prob.parameters():
            if p.id in [v.id for v in self.fix_vars[1]]:
                var = [v for v in self.fix_vars[1] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._x_prob

    @property
    def y_prob_(self) -> cp.Problem:
        for p in self._y_prob.parameters():
            if p.id in [v.id for v in self.fix_vars[0]]:
                var = [v for v in self.fix_vars[0] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._y_prob

    def solve(self,
              solver: str = cp.SCS,
              lbd: float = 0.1,
              nu: float = 1,
              max_iter: float = 100,
              gap_tolerance: float = 1e-6,
              slack_tolerance: float = 1e-6,
              *args, **kwargs
              ) -> float | None:
        print(f"{' DBCP Summary ':=^{85}}")
        print(f"Block coordinate descent start with solver {solver}...")
        print("-" * 85)
        print(f"{'iter':<7} {'xcost':<20} {'ycost':<20} {'gap':<20} {'total_slack':<20}")
        print("-" * 85)
        for v in self.variables():
            if v.value is None:
                v.project_and_assign(np.random.standard_normal(v.shape))
        slack_ids = sorted([s.id for s in self.slack_vars])
        prox_params = [cp.Parameter(v.shape, id=v.id, **v.attributes) for v in self.variables()]
        x_prox = cp.Problem(cp.Minimize(cp.multiply(lbd, cp.sum([
            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
            for v in self.x_prob.variables() if v.id not in slack_ids
        ]))))
        y_prox = cp.Problem(cp.Minimize(cp.multiply(lbd, cp.sum([
            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
            for v in self.y_prob.variables() if v.id not in slack_ids
        ]))))
        if self.objective.NAME == "minimize":
            xprox_prob = self.x_prob + x_prox
            yprox_prob = self.y_prob + y_prox
        else:
            xprox_prob = self.x_prob - x_prox
            yprox_prob = self.y_prob - y_prox
        self.nu.value = nu
        i = 0
        try:
            while True:
                for v in self.x_prob_.variables():
                    if v.id not in slack_ids:
                        [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                xprox_prob.solve(solver=solver, *args, **kwargs)
                for v in self.y_prob_.variables():
                    if v.id not in slack_ids:
                        [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                yprox_prob.solve(solver=solver, *args, **kwargs)

                if ((xprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) or
                        (yprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))):
                    raise SolveError(f"Solver {solver} failed. Try a different solver.")
                gap = np.abs(self.x_prob.objective.value - self.y_prob.objective.value)
                total_slack = np.sum([np.sum(np.abs(s.value)) for s in self.slack_vars])
                print(
                    f"{i:<7} "
                    f"{self.x_prob.objective.value:<20.9f} "
                    f"{self.y_prob.objective.value:<20.9f} "
                    f"{gap:<20.4e} "
                    f"{total_slack:<20.4e} "
                )
                if gap < gap_tolerance:
                    if total_slack < slack_tolerance:
                        self._status = "converge"
                    else:
                        self._status = "converge_infeasible"
                    break
                else:
                    i += 1
                if i == max_iter:
                    if total_slack < slack_tolerance:
                        self._status = "converge_inaccurate"
                    else:
                        self._status = "converge_inaccurate_infeasible"
                    break
        except cp.SolverError as e:
            raise SolveError("Solver failed. Try with larger 'lbd' value.") from e

        print("-" * 85)
        print(f"Terminated with status: {self.status}.")
        print("=" * 85)
        self._value = self.objective.value
        if 'infeasible' in self.status:
            warnings.warn(
                f"The returned solution is infeasible with total constraint violation {total_slack}."
                f"Consider increasing 'nu' value or trying another initial point.")
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
