import numpy as np
from abc import ABC, abstractmethod
from pyoptsparse import Optimization, OPT

class MPCC(ABC):
    def __init__(self,μ=0.1,θ=10) -> None:
        self.μ = μ
        self.θ = θ
    
    @abstractmethod
    def objfun(self,xdict):
        pass
    
    @abstractmethod
    def complementarity(self,xdict):
        pass
    
    def solve(self):
        tol_p = self.θ * self.μ
        opt = OPT('IPOPT',
            options={
                'print_level':2,
                'mu_init':self.μ,
                # 'mu_target':self.μ,
                'dual_inf_tol':tol_p,
                'constr_viol_tol':tol_p,
                'compl_inf_tol':tol_p
            })
        sol = opt(self.optProb,sens='FD')
        return sol

class BARD1(MPCC):
    def __init__(self,x_init=None,y_init=None,l_init=None,μ=0.1,θ=10,π = 1.0) -> None:
        super().__init__(μ,θ)
        
        self.π = π
        
        if x_init is None:
            x_init = 0
        if y_init is None:
            y_init = 0
        if l_init is None:
            l_init = np.zeros(3)
        
        self.optProb = Optimization('BARD1 Problem',self.objfun)
        ## Define Options
        

        ## Add variables
        self.optProb.addVar('x',value=x_init)
        self.optProb.addVar('y',value=y_init)
        self.optProb.addVarGroup('l',3,lower=0,value=l_init)
        
        ## Add constraints
        self.optProb.addConGroup('con',1,lower=0,upper=0)
        self.optProb.addConGroup('con2',3,lower=0)
        
        ## Add objective
        self.optProb.addObj('obj')
    
    def objfun(self,xdict):
        x = xdict['x']
        y = xdict['y']
        l = xdict['l']
        funcs = {}
        funcs['obj'] = (x-5)**2 + (2*y+1)**2 + self.π*(l[0]*(3*x-y-3)+l[1]*(-x+0.5*y+4)+l[2]*(-x-y+7))
        fail = False
        funcs['con'] = [0]
        funcs['con'][0] = 2*(y-1) - 1.5 * x + l[0] - 0.5 * l[1] + l[2]
        funcs['con2'] = [0] * 3
        funcs['con2'][0] = 3*x-y-3
        funcs['con2'][1] = -x+0.5*y+4
        funcs['con2'][2] = -x-y+7
        fail = False
        return funcs, fail
    
    def complementarity(self,xdict):
        x = xdict['x']
        y = xdict['y']
        l = xdict['l']
        return l[0]*(3*x-y-3)+l[1]*(-x+0.5*y+4)+l[2]*(-x-y+7)

def solve_mpcc(optimization_problem,comp1,comp2,**kwargs):
    
    # Defining parameters
    γ = kwargs.get('γ',0.4)
    κ = kwargs.get('κ',0.2)
    σ = kwargs.get('σ',10)
    θ = kwargs.get('θ',10)
    μ = kwargs.get('μ',0.1)
    π = kwargs.get('π',1.0)
    tol = kwargs.get('tol',1e-3)
    k_max = kwargs.get('k_max',10)
    
    # Initializarion
    k = 1
    last_obj = 0
    bard = BARD1(π=π,θ=θ)
    sol = bard.solve()
    x,y,l,f,c = sol.xStar['x'],sol.xStar['y'],sol.xStar['l'],sol.fStar,bard.complementarity(sol.xStar)
    
    # Print output header
    print(f'{"Iter":>5}\t{"Termination_status":>15}\t{"Objective":>15}\t{"MPCC_compl":>15} {"lg(mu)":>15} {"π":>15}\n')
    
    while k < k_max:
        tol_c = μ**γ
        
        bard = BARD1(π=π,θ=θ,x_init=x,y_init=y,l_init=l)
        sol = bard.solve()
        x,y,l,f,c = sol.xStar['x'],sol.xStar['y'],sol.xStar['l'],sol.fStar,bard.complementarity(sol.xStar)
        
        if c < tol_c:
            print(f'{k}\t{sol.optInform["text"]}\t{f[0]:>15}\t{c:>15} {np.log(μ)} {π}')
            if np.abs(f-last_obj) < tol:
                print(f'Obtained solution satisfies the complementarity condition at {tol_c} at {k} iteraitons')
                break
            else:
                last_obj = f
            k += 1
            μ *= κ
        else:
            if π < 1e14:
                π *= σ
            else:
                print(f'Couldnt find a suitable value for π.')
                break
    return x,y,l,f,c