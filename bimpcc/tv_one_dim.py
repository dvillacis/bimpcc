import numpy as np
from pyoptsparse import Optimization, OPT
from scipy.sparse import spdiags

class TV_1D_MPCC:
    def __init__(self,true_sgn,noisy_sgn,K,R,Q,xdict,α_size=1) -> None:
        self.true_sgn = true_sgn
        self.noisy_sgn = noisy_sgn
        self.K = K.tosparse()
        self.R = R.tosparse()
        self.Q = Q.tosparse()
        
        self.xdict = xdict
        
        self.optProb = Optimization('TV_1D_MPCC',self.objfun)
        
        # Dimensions
        self.dim_u = len(self.true_sgn)
        self.dim_c = self.K.shape[0]
        self.dim_α = α_size
        print(f'dim_u = {self.dim_u}, dim_c = {self.dim_c}, dim_α = {self.dim_α}')
        
        # Design variables
        self.optProb.addVarGroup('u',self.dim_u,value=self.xdict['u'])
        self.optProb.addVarGroup('c',self.dim_c,value=self.xdict['c'])
        self.optProb.addVarGroup('α',self.dim_α,lower=1e-8,value=self.xdict['α'])
        
        # Nonlinear Constraints
        # self.optProb.addConGroup('nl_con_1',self.dim_c,lower=0)
        # self.optProb.addConGroup('nl_con_2',self.dim_c,lower=0)
        # self.optProb.addConGroup('nl_con_3',self.dim_c,lower=0)
        
        # Linear Constraints
        # jac_u = spdiags(np.ones(self.dim_u),0,self.dim_u,self.dim_u)
        # rhs = self.noisy_sgn.ravel()
        # jac_c = self.K.T
        self.optProb.addConGroup('lin_con_1',self.dim_u,lower=0,upper=0,wrt=['u','c']) #jac={'u':jac_u,'c':jac_c}
        self.optProb.addConGroup('lin_con_2',self.dim_c,lower=0,wrt=['c','α'])
        self.optProb.addConGroup('lin_con_3',self.dim_c,lower=0,wrt=['c','α'])
        self.optProb.addConGroup('nl_con_1',self.dim_c,lower=0,wrt=['c','u','α'])
        self.optProb.addConGroup('nl_con_2',self.dim_c,lower=0,wrt=['c','u','α'])
        self.optProb.addConGroup('nl_con_3',self.dim_c,lower=0,wrt=['c','u'])
        # jac_c = spdiags(np.ones(self.dim_c),0,self.dim_c,self.dim_c)
        # jac_α = self.Q
        # self.optProb.addConGroup('lin_con_2',self.dim_c,lower=0,wrt=['c','α'],jac={'c':jac_c,'α':jac_α},linear=True)
        
        # jac_c = spdiags(-np.ones(self.dim_c),0,self.dim_c,self.dim_c)
        # jac_α = self.Q
        # self.optProb.addConGroup('lin_con_3',self.dim_c,lower=0,wrt=['c','α'],jac={'c':jac_c,'α':jac_α},linear=True)
        
        self.optProb.addObj('obj')
        
    def objfun(self,xdict):
        funcs = {}
        funcs['obj'] = 0.5 * np.linalg.norm(xdict['u'] - self.true_sgn)**2
        
        funcs['lin_con_1'] = self.K.T @ xdict['c'] + xdict['u'] - self.noisy_sgn
        funcs['lin_con_2'] = xdict['c'] + self.Q @ xdict['α']
        funcs['lin_con_3'] = -xdict['c'] + self.Q @ xdict['α']
        
        funcs['nl_con_1'] = xdict['c'] * (self.K @ xdict['u']) + (self.Q @ xdict['α']) * (self.K @ xdict['u'])
        funcs['nl_con_2'] = xdict['c'] * (self.K @ xdict['u']) - (self.Q @ xdict['α']) * (self.K @ xdict['u'])
        funcs['nl_con_3'] = xdict['c'] * (self.K @ xdict['u'])
        
        fail = False
        return funcs, fail
    
    def solve(self, print_level=0):
        opt = OPT('IPOPT',
                options={
                    'print_level': print_level,
                    'linear_solver': 'ma86',
                })
        sol = opt(self.optProb, sens='FD')
        if print_level > 0:
            self.optProb.printSparsity()
        param = sol.xStar['α']
        rec = sol.xStar['u']
        return param, rec, sol
    
    def __str__(self) -> str:
        print(self.optProb)
        return ''
    
    
def solve_mpcc(true_sgn,noisy_sgn,K,R,Q,α_size=1,print_level=0):
    xdict_init = {
        'u': noisy_sgn,
        'c': 1e-9*np.ones(K.shape[0]),
        'α': 1e-9*np.ones(α_size),
    }
    tv_1d_mpcc = TV_1D_MPCC(true_sgn,noisy_sgn,K,R,Q,xdict_init,α_size)
    param,rec,sol = tv_1d_mpcc.solve(print_level=print_level)
    print(sol)
    return param,rec,sol