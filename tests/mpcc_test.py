import numpy as np
from pyoptsparse import Optimization

from bimpcc.mpcc import solve_mpcc

def comp1(xdict):
    x = xdict["x"]
    y = xdict["y"]
    return np.array([3*x-y-3,-x+0.5*y+4,-x-y+7])

def comp2(xdict):
    l = xdict["l"]
    return l

def objfunc(xdict):
    # print(comp1(xdict),comp2(xdict))
    # print(np.dot(comp1(xdict),comp2(xdict)))
    x = xdict["x"]
    y = xdict["y"]
    l = xdict["l"]
    funcs = {}
    funcs["obj"] = (x-5)**2 + (2*y+1)**2 + 100.0*(l[0]*(3*x-y-3)+l[1]*(-x-0.5*y+4)+l[2]*(-x-y+7))
    conval = [0]
    conval[0] = 2*(y-1) - 1.5 * x + l[0] - 0.5 * l[1] + l[2]
    ineq_con = [0]*3
    ineq_con[0] = 3*x-y-3
    ineq_con[1] = -x+0.5*y+4
    ineq_con[2] = -x-y+7
    funcs["con"] = conval
    funcs["ineq_con"] = ineq_con
    fail = False

    return funcs, fail

optProb = Optimization('mpcc',objfunc)
optProb.addVar('x')
optProb.addVar('y')
optProb.addVarGroup('l',3,lower=0)

optProb.addConGroup('con',1,lower=0,upper=0)
optProb.addConGroup('ineq_con',3,lower=0)

optProb.addObj("obj")

solve_mpcc(optProb,comp1,comp2,γ=0.5)