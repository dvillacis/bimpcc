import numpy as np
from pyoptsparse import Optimization, OPT, pyOpt_utils
from scipy.sparse import spdiags, vstack, dia_matrix, csr_matrix,eye, hstack
from pylops import Diagonal, VStack

class MPCC_RELAXED:
    def __init__(self, true_img, noisy_img, K, R, Q, tik=0.1, alpha_max=1.0, alpha_size=1, tol=1e-5, max_iter=1000, init_alpha=None, init_u=None, init_q=None, init_r=None, init_delta=None, init_theta=None):
        print(K.shape)
        
        self.dim_q = K.shape[0]
        self.dim_u = K.shape[1]
        self.dim_alpha = alpha_size
        self.dim_r = K.shape[0] // 2
        
        print(f'dim_q={self.dim_q} dim_u={self.dim_u} dim_alpha={self.dim_alpha}, dim_r={self.dim_r}')
        
        # Setting initial values
        if init_alpha is None:
            init_alpha = np.zeros(alpha_size)
        if init_u is None:
            init_u =np.real(R.H * noisy_img.ravel()).ravel()
        if init_q is None:
            init_q = np.zeros(self.dim_q)
        if init_r is None:
            init_r = 0.0001*np.ones(self.dim_r)
        if init_delta is None:
            init_delta = 0.0001*np.ones(self.dim_r)
        if init_theta is None:
            init_theta = 0.0001*np.ones(self.dim_r)
            
        # print(noisy_img.shape)
        self.max_iter = max_iter
        self.m,self.n = true_img.shape
        self.true_img = true_img
        self.noisy_img = noisy_img
        print(f'Image size: {self.m}x{self.n}')
        self.tik = tik
        self.alpha_max = alpha_max
        self.K = K
        self.R = R  # Imaging Forward Model
        self.Q = Q # Patch Operator
        
        # Define the optimization problem
        self.optProb = Optimization('TV MPCC Problem',self.objfun)
        # Design variables
        self.optProb.addVarGroup('u',self.dim_u,lower=0,upper=None,value=init_u)
        self.optProb.addVarGroup('q',self.dim_q,value=init_q)
        self.optProb.addVarGroup('α',self.dim_alpha,lower=0,upper=alpha_max,value=init_alpha)
        self.optProb.addVarGroup('r',self.dim_r,lower=0,value=init_r)
        self.optProb.addVarGroup('δ',self.dim_r,lower=1e-8,value=init_delta)
        self.optProb.addVarGroup('θ',self.dim_r,lower=-np.pi,upper=np.pi,value=init_theta)
        # Nonlinear constraints
        jac_con_1_u = self.K.tosparse()
        jac_con_1_r = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        jac_con_1_theta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        self.optProb.addConGroup('nonlin_con_1',self.dim_q,lower=0,upper=0,wrt=['u','r','θ'],jac={'u':jac_con_1_u,'r':jac_con_1_r,'θ':jac_con_1_theta})
        
        jac_con_2_q = Diagonal(np.ones(self.dim_q)).tosparse()
        jac_con_2_delta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        jac_con_2_theta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        self.optProb.addConGroup('nonlin_con_2',self.dim_q,lower=0,upper=0,wrt=['q','δ','θ'],jac={'q':jac_con_2_q,'δ':jac_con_2_delta,'θ':jac_con_2_theta})
        
        jac_con_3_alpha = self.Q.tosparse()
        jac_con_3_r = Diagonal(np.ones(self.dim_r)).tosparse()
        jac_con_3_delta = Diagonal(np.ones(self.dim_r)).tosparse()
        self.optProb.addConGroup('nonlin_con_3',self.dim_r,lower=None,upper=tol,wrt=['α','r','δ'],jac={'α':jac_con_3_alpha,'r':jac_con_3_r,'δ':jac_con_3_delta})
        
        # Linear constraints
        jac_u = np.real((self.R.H*self.R).tosparse())#+ 0.001*eye(self.dim_u)
        rhs = np.real(self.R.H * self.noisy_img.ravel())
        # print(self.K.adjoint().shape)
        # print(self.K.T.shape)
        jac_q = self.K.tosparse().T
        self.optProb.addConGroup('lin_con1',self.dim_u,lower=rhs.ravel(),upper=rhs.ravel(),wrt=['u','q'],jac={'u':jac_u,'q':jac_q},linear=True)

        jac_alpha = self.Q.tosparse()
        jac_delta = -Diagonal(np.ones(self.dim_r)).tosparse()
        self.optProb.addConGroup('lin_con2',self.dim_r,lower=0,wrt=['α','δ'],jac={'α':jac_alpha,'δ':jac_delta},linear=True)
        
    def objfun(self,xdict):
        # print(f'Objective function called. {xdict["u"].shape}')
        u = xdict['u']
        q = xdict['q']
        α = xdict['α']
        r = xdict['r']
        δ = xdict['δ']
        θ = xdict['θ']
        
        # print(f'α={α}')
        
        funcs = {}
        funcs['obj'] = 0.5 * np.linalg.norm(u-self.true_img.ravel())**2 + 0.5 * self.tik * np.linalg.norm(α)**2
        
        funcs['nonlin_con_1'] = self.K @ u - np.concatenate([r * np.cos(θ),r * np.sin(θ)])
        funcs['nonlin_con_2'] = q - np.concatenate([δ * np.cos(θ),δ * np.sin(θ)])
        funcs['nonlin_con_3'] = r * (self.Q.matvec(α)-δ)
        
        fail = False
        return funcs, fail
    
    def usr_jac(self,xdict,fdict):
        # print(f'Jacobian called. {xdict["u"].shape}')
        u = xdict['u']
        r = xdict['r']
        α = xdict['α']
        θ = xdict['θ']
        δ = xdict['δ']

        gout = {}
        
        gout['obj'] = {}
        gout['obj']['u'] = u-self.true_img.ravel()
        gout['obj']['α'] = self.tik * α
        
        gout['nonlin_con_1'] = {}
        gout['nonlin_con_1']['u'] = self.K.tosparse()
        D1 = spdiags(-np.cos(θ),0,self.dim_r,self.dim_r)
        D2 = spdiags(-np.sin(θ),0,self.dim_r,self.dim_r)
        gout['nonlin_con_1']['r'] = vstack([D1,D2])
        D3 = spdiags(r*np.sin(θ),0,self.dim_r,self.dim_r)
        D4 = spdiags(-r*np.cos(θ),0,self.dim_r,self.dim_r)
        gout['nonlin_con_1']['θ'] = vstack([D3,D4])
        
        gout['nonlin_con_2'] = {}
        gout['nonlin_con_2']['q'] = Diagonal(np.ones(self.dim_q)).tosparse()
        gout['nonlin_con_2']['δ'] = vstack([D1,D2])
        D5 = spdiags(δ*np.sin(θ),0,self.dim_r,self.dim_r)
        D6 = spdiags(-δ*np.cos(θ),0,self.dim_r,self.dim_r)
        gout['nonlin_con_2']['θ'] = vstack([D5,D6])
        
        gout['nonlin_con_3'] = {}
        gout['nonlin_con_3']['α'] = Diagonal(r).tosparse() @ self.Q.tosparse()
        gout['nonlin_con_3']['r'] = spdiags(self.Q.matvec(α)-δ,0,self.dim_r,self.dim_r)
        gout['nonlin_con_3']['δ'] = spdiags(-r,0,self.dim_r,self.dim_r)
        
        fail = False
        
        return gout, fail
    
    def solve(self,print_sparsity=False):
        if print_sparsity: self.optProb.printSparsity()
        self.optProb.addObj('obj')
        opt = OPT('IPOPT',options={
            'print_level':5,
            'acceptable_tol':1e-2,
            'acceptable_iter':5,
            'max_iter':self.max_iter,
            'linear_solver':'ma86',
            # 'hessian_approximation_space':'all-variables'
            # 'nlp_scaling_method':'none',
            # 'mu_init':1e-2,
            # 'warm_start_init_point':'yes',
            # 'warm_start_bound_push':1e-9,
            # 'warm_start_bound_frac':1e-9,
            # 'warm_start_slack_bound_frac':1e-9,
            # 'warm_start_slack_bound_push':1e-9,
            # 'warm_start_mult_bound_push':1e-9
            # 'nlp_scaling_max_gradient': 1.0
        })
        sol = opt(self.optProb,sens=self.usr_jac,sensMode='pgc')
        param = sol.xStar['α']
        rec = sol.xStar['u']
        print(type(sol))
        extra = {
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
        return param, rec, sol.xStar['q'], sol.xStar['r'], sol.xStar['δ'], sol.xStar['θ'], extra
    
def solve_mpcc(true_img, noisy_img, K, R, Q, tik=0.1, alpha_max=1.0, alpha_size=1, tol_max=1000.0, max_iter=3000):
    extras = []
    print(f'********* Solve with tol={tol_max} **********')
    mpcc = MPCC_RELAXED(true_img=true_img,noisy_img=noisy_img,K=K,R=R,Q=Q,alpha_size=alpha_size,tik=tik,alpha_max=alpha_max,tol=tol_max,max_iter=max_iter)
    param,sol,q,r,delta,theta,extra = mpcc.solve()
    extras.append(extra)
    
    t = tol_max/10
    
    for i in np.arange(3):
        print(f'********* Solve with tol={t} **********')
        mpcc = MPCC_RELAXED(true_img=true_img,noisy_img=noisy_img,K=K,R=R,Q=Q,alpha_size=alpha_size,tik=tik,alpha_max=alpha_max,tol=t,max_iter=max_iter,init_alpha=param,init_u=sol.ravel(),init_q=q, init_r=r, init_delta=delta, init_theta=theta)
        param,sol,q,r,delta,theta,extra = mpcc.solve()
        extras.append(extra)
        
        t = t / 10
    
    return param,sol,q,r,delta,theta,extras