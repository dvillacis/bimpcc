import numpy as np
from pyoptsparse import Optimization, OPT
from scipy.sparse import spdiags
from bimpcc.utils import sparse_to_coosparse, diag_to_coodiag, block_diag_to_coosparse

class TV_2D_MPCC:
    def __init__(self,true_img,noisy_img,K,R,Q,ϵ,xdict,α_size=1) -> None:
        self.true_img = true_img.ravel()
        self.noisy_img = noisy_img.ravel()
        self.K = K.tosparse()
        self.R = R.tosparse()
        self.Q = Q.tosparse()
        
        self.xdict = xdict
        self.ϵ = ϵ
        
        self.optProb = Optimization('TV_2D_MPCC',self.objfun)
        
        # Dimensions
        self.dim_u = len(self.true_img)
        self.dim_q = self.K.shape[0]
        self.dim_r = self.K.shape[0] // 2
        self.dim_α = α_size
        print(f'dim_u = {self.dim_u}, dim_q = {self.dim_q}, dim_r = {self.dim_r}, dim_α = {self.dim_α}, ϵ = {ϵ}')
        
        # Design variables
        self.optProb.addVarGroup('u',self.dim_u,value=self.xdict['u'])
        self.optProb.addVarGroup('q',self.dim_q,value=self.xdict['q'])
        self.optProb.addVarGroup('r',self.dim_r,lower=0,value=self.xdict['r'])
        self.optProb.addVarGroup('δ',self.dim_r,lower=0,value=self.xdict['δ'])
        self.optProb.addVarGroup('θ',self.dim_r,value=self.xdict['θ'])
        self.optProb.addVarGroup('α',self.dim_α,lower=1e-8,value=self.xdict['α'])
        
        # Constraints
        
        ## Linear Constraint 1
        RTR = self.R.H @ self.R
        jac_u = sparse_to_coosparse(RTR)
        rhs = np.real(self.R.H * self.noisy_img.ravel())
        jac_q = sparse_to_coosparse(self.K.T)
        self.optProb.addConGroup('lin_con_1',self.dim_u,lower=rhs.ravel(),upper=rhs.ravel(),wrt=['u','q'],jac={'u':jac_u,'q':jac_q},linear=True)
        
        ## Linear Constraint 2
        # jac_alpha = self.Q
        jac_alpha = sparse_to_coosparse(self.Q)
        jac_delta = diag_to_coodiag(-np.ones(self.dim_r))
        # jac_delta = spdiags(-np.ones(self.dim_r),0,self.dim_r,self.dim_r)
        self.optProb.addConGroup('lin_con_2',self.dim_r,lower=0,wrt=['α','δ'],jac={'α':jac_alpha,'δ':jac_delta},linear=True)
        
        ## Nonlinear Constraint 1
        jac_nl1_u = sparse_to_coosparse(self.K)
        jac_nl1_r = block_diag_to_coosparse(np.ones(self.dim_r),np.ones(self.dim_r))
        jac_nl1_theta = block_diag_to_coosparse(np.ones(self.dim_r),np.ones(self.dim_r))
        self.optProb.addConGroup('nl_con_1',self.dim_q,lower=0,upper=0,wrt=['u','r','θ'],jac={'u':jac_nl1_u,'r':jac_nl1_r,'θ':jac_nl1_theta})
        
        ## Nonlinear Constraint 2
        jac_nl2_q = diag_to_coodiag(np.ones(self.dim_q))
        jac_nl2_delta = block_diag_to_coosparse(np.ones(self.dim_r),np.ones(self.dim_r))
        jac_nl2_theta = block_diag_to_coosparse(np.ones(self.dim_r),np.ones(self.dim_r))
        self.optProb.addConGroup('nl_con_2',self.dim_q,lower=0,upper=0,wrt=['q','δ','θ'],jac={'q':jac_nl2_q,'δ':jac_nl2_delta,'θ':jac_nl2_theta})
        
        ## Nonlinear Constraint 3
        jac_nl3_alpha = sparse_to_coosparse(self.Q)
        jac_nl3_r = diag_to_coodiag(np.ones(self.dim_r))
        jac_nl3_delta = diag_to_coodiag(np.ones(self.dim_r))
        self.optProb.addConGroup('nl_con_3',self.dim_r,upper=ϵ,wrt=['r','α','δ'],jac={'r':jac_nl3_r,'α':jac_nl3_alpha,'δ':jac_nl3_delta})
        
        self.optProb.addObj('obj')
        
    def objfun(self,xdict):
        funcs = {}
        funcs['obj'] = 0.5 * np.linalg.norm(xdict['u'] - self.true_img)**2 
        
        # funcs['lin_con_1'] = self.K.T @ xdict['q'] + self.R.T @ (self.R @ xdict['u']) - self.R.T * self.noisy_img
        # funcs['lin_con_2'] = (self.Q @ xdict['α']) - xdict['δ']
        
        funcs['nl_con_1'] = self.K @ xdict['u'] - np.concatenate([xdict['r'] * np.cos(xdict['θ']),xdict['r'] * np.sin(xdict['θ'])])
        funcs['nl_con_2'] = xdict['q'] - np.concatenate([xdict['δ'] * np.cos(xdict['θ']),xdict['δ'] * np.sin(xdict['θ'])])
        funcs['nl_con_3'] = xdict['r'] * ((self.Q @ xdict['α'])-xdict['δ'])
        
        fail = False
        return funcs, fail
    
    def usr_jac(self,xdict,fdict):
        gout = {}
        
        gout['obj'] = {}
        gout['obj']['u'] = xdict['u']-self.true_img.ravel()
        
        gout['nl_con_1'] = {}
        gout['nl_con_1']['u'] = sparse_to_coosparse(self.K)
        gout['nl_con_1']['r'] = block_diag_to_coosparse(-np.cos(xdict['θ']),-np.sin(xdict['θ']))
        gout['nl_con_1']['θ'] = block_diag_to_coosparse(xdict['r'] * np.sin(xdict['θ']),-xdict['r'] * np.cos(xdict['θ']))
        
        gout['nl_con_2'] = {}
        gout['nl_con_2']['q'] = diag_to_coodiag(np.ones(self.dim_q))
        gout['nl_con_2']['δ'] = block_diag_to_coosparse(-np.cos(xdict['θ']),-np.sin(xdict['θ']))
        gout['nl_con_2']['θ'] = block_diag_to_coosparse(xdict['δ']*np.sin(xdict['θ']),-xdict['δ'] * np.cos(xdict['θ']))
        
        gout['nl_con_3'] = {}
        gout['nl_con_3']['α'] = sparse_to_coosparse(spdiags(xdict['r'],0,self.dim_r,self.dim_r)@self.Q)
        gout['nl_con_3']['r'] = diag_to_coodiag(self.Q @ xdict['α'] - xdict['δ'])
        gout['nl_con_3']['δ'] = diag_to_coodiag(-xdict['r'])
        
        return gout, False
        
    
    def solve(self, print_level=0):
        opt = OPT('IPOPT',
                options={
                    'print_level': print_level,
                    'linear_solver': 'ma86',
                    'ma86_print_level':0,
                    'limited_memory_max_history':0,
                    'tol':self.ϵ,
                    # 'limited_memory_update_type':'sr1',
                    # 'ma27_ignore_singularity': 'yes',
                    # 'ma86_scaling':'none',
                    # 'linear_scaling_on_demand':'no',
                    'fast_step_computation':'yes',
                })
        if print_level > 0:
            self.optProb.printSparsity()
        # sol = opt(self.optProb, sens='FD')
        sol = opt(self.optProb,sens=self.usr_jac,sensMode='pgc')
        
        # grad = self.optProb.sens
        
        extra = {
            'nCon': self.optProb.nCon,
            'nVar': self.optProb.ndvs,
            # 'jacNZ': grad.nnz,
            'fStar':sol.fStar,
            'optInform':sol.optInform,
            'info':{
                "optTime":sol.optTime,
                "userObjTime":sol.userObjTime,
                "userSensTime":sol.userSensTime,
                "userObjCalls":sol.userObjCalls,
                "userSensCalls":sol.userSensCalls,
                "interfaceTime":sol.interfaceTime,
                "optCodeTime":sol.optCodeTime,  
            }
        }
        print(extra)
        return sol, extra
    
    def __str__(self) -> str:
        print(self.optProb)
        return ''
    
    
def solve_mpcc(true_img,noisy_img,K,R,Q,α_size=1,print_level=0):
    extras = []
    xdict = {
        'u': np.real(R.H @ noisy_img.ravel()).ravel(),
        'q': 1e-9*np.ones(K.shape[0]),
        'r': 1e-1*np.ones(K.shape[0]//2),
        'δ': 1e-9*np.ones(K.shape[0]//2),
        'θ': 1e-7*np.ones(K.shape[0]//2),
        'α': 1e-9*np.ones(α_size),
    }
    ϵ = 0.1
    for i in range(3):
        print(f'********* Solve with tol={ϵ} **********')
        tv_2d_mpcc = TV_2D_MPCC(true_img,noisy_img,K,R,Q,ϵ,xdict,α_size)
        sol,extra = tv_2d_mpcc.solve(print_level=print_level)
        xdict = {
            'u': sol.xStar['u'],
            'q': sol.xStar['q'],
            'r': sol.xStar['r'],
            'δ': sol.xStar['δ'],
            'θ': sol.xStar['θ'],
            'α': sol.xStar['α'],
        }
        extras.append(extra)
        ϵ *= 0.1
    # print(sol)
    return sol,extras