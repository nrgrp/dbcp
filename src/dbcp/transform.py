import numpy as np
import cvxpy as cp
from cvxpy.constraints.nonpos import Inequality, NonPos, NonNeg
from cvxpy.constraints.zero import Equality, Zero
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC


def transform_with_slack(
        prob: cp.Problem
) -> cp.Problem:
    proj_obj = 0
    proj_constr = []
    for c in prob.constraints:
        if isinstance(c, Inequality):
            s = cp.Variable(shape=c.shape, nonneg=True)
            proj_obj += cp.sum(s)
            proj_constr.append(c.args[0] <= c.args[1] + s)
        elif isinstance(c, Equality):
            s = cp.Variable(shape=c.shape)
            proj_obj += cp.norm1(s)
            proj_constr.append(c.args[0] == c.args[1] + s)
        elif isinstance(c, Zero):
            s = cp.Variable(shape=c.shape)
            proj_obj += cp.norm1(s)
            proj_constr.append(Zero(c.expr + s))
        elif isinstance(c, NonPos):
            s = cp.Variable(shape=c.shape, nonneg=True)
            proj_obj += cp.sum(s)
            proj_constr.append(NonNeg(-c.expr + s))
        elif isinstance(c, NonNeg):
            s = cp.Variable(shape=c.shape, nonneg=True)
            proj_obj += cp.sum(s)
            proj_constr.append(NonNeg(c.expr + s))
        elif isinstance(c, PSD):
            s = cp.Variable((), nonneg=True)
            proj_obj += cp.sum(s)
            proj_constr.append(PSD(c.expr + s * np.eye(c.shape[0])))
        elif isinstance(c, SOC):
            s = cp.Variable(shape=c.shape, nonneg=True)
            proj_obj += cp.sum(s)
            proj_constr.append(cp.SOC(c.args[0] + s, c.args[1], axis=c.axis))
        else:
            raise TypeError(f"Constraint type {type(c)} not supported.")
    return cp.Problem(cp.Minimize(proj_obj), proj_constr)