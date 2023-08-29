import numpy as np
from pyoptsparse import Optimization, OPT
from pylops import Diagonal, VStack
from scipy.sparse import spdiags,vstack

class TV_MPCC_RELAXED:
    def __init__(self,true_img,noisy_img,K,R,Q,xdict_init,α_size=1,α_max=1,π=1.0,μ=0.1,ν=10,tik=1e-3) -> None:
        # print(f'ν={ν}, μ={μ}, π={π}')
        self.tol_p = ν * μ
        self.μ = μ
        self.π = π
        self.tik = tik
        
        # Setting dimensions
        self.dim_q = K.shape[0]
        self.dim_u = K.shape[1]
        self.dim_α = α_size
        self.dim_r = K.shape[0]
        # self.m,self.n = true_img.shape
        self.true_img = true_img
        self.noisy_img = noisy_img
        self.K = K.tosparse()
        self.R = R.tosparse()
        self.Q = Q.tosparse()
        
        print(f'dim_q={self.dim_q} dim_u={self.dim_u} dim_α={self.dim_α} dim_r={self.dim_r}')
        
        # Defining optimization problem
        self.optProb = Optimization('RELAXED TV MPCC Problem',self.objfun)
        
        # Design variables
        self.optProb.addVarGroup('u',self.dim_u,lower=1e-8,upper=None,value=xdict_init['u'])
        self.optProb.addVarGroup('q',self.dim_q,value=xdict_init['q'])
        self.optProb.addVarGroup('α',self.dim_α,lower=1e-8,upper=α_max,value=xdict_init['α'])
        self.optProb.addVarGroup('r',self.dim_r,lower=1e-8,value=xdict_init['r'])
        self.optProb.addVarGroup('δ',self.dim_r,lower=1e-8,value=xdict_init['δ'])
        self.optProb.addVarGroup('θ',self.dim_r,value=xdict_init['θ'])
        
        # Nonlinear constraints
        # jac_con_1_u = self.K
        # jac_con_1_r = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        # jac_con_1_theta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        # self.optProb.addConGroup('nonlin_con_1',self.dim_q,lower=0,upper=0,wrt=['u','r','θ'],jac={'u':jac_con_1_u,'r':jac_con_1_r,'θ':jac_con_1_theta})
        self.optProb.addConGroup('nonlin_con_1',self.dim_q,lower=0,upper=0)
        
        # jac_con_2_q = Diagonal(np.ones(self.dim_q)).tosparse()
        # jac_con_2_delta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        # jac_con_2_theta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        # self.optProb.addConGroup('nonlin_con_2',self.dim_q,lower=0,upper=0,wrt=['q','δ','θ'],jac={'q':jac_con_2_q,'δ':jac_con_2_delta,'θ':jac_con_2_theta})
        self.optProb.addConGroup('nonlin_con_2',self.dim_q,lower=0,upper=0)
        
        # Linear constraints
        jac_u = np.real((self.R.H*self.R))
        rhs = np.real(self.R.H * self.noisy_img.ravel())
        jac_q = self.K.T
        self.optProb.addConGroup('lin_con1',self.dim_u,lower=rhs.ravel(),upper=rhs.ravel(),wrt=['u','q'],jac={'u':jac_u,'q':jac_q},linear=True)
        
        jac_alpha = self.Q
        jac_delta = Diagonal(-np.ones(self.dim_r)).tosparse()
        self.optProb.addConGroup('lin_con2',self.dim_r,lower=0,wrt=['α','δ'],jac={'α':jac_alpha,'δ':jac_delta},linear=True)
        
        ## Add objective
        self.optProb.addObj('obj')
        
    def objfun(self,xdict):
        u = xdict['u']
        q = xdict['q']
        α = xdict['α']
        r = xdict['r']
        δ = xdict['δ']
        θ = xdict['θ']
        
        # print(f'α={α}')
        
        funcs = {}
        funcs['obj'] = 0.5 * np.linalg.norm(u-self.true_img.ravel())**2 + 0.5 * self.tik * np.linalg.norm(α)**2 + self.π * self.complementarity(xdict)[1]
        
        funcs['nonlin_con_1'] = self.K @ u - np.concatenate([r * np.cos(θ),r * np.sin(θ)])
        funcs['nonlin_con_2'] = q - np.concatenate([δ * np.cos(θ),δ * np.sin(θ)])
        
        fail = False
        return funcs, fail
    
    def usr_jac(self,xdict,fdict):
        u = xdict['u']
        r = xdict['r']
        α = xdict['α']
        θ = xdict['θ']
        δ = xdict['δ']
        
        gout = {}
        
        gout['obj'] = {}
        gout['obj']['u'] = u-self.true_img.ravel()
        gout['obj']['α'] = self.tik * α + self.π * (self.Q.T @ r)
        gout['obj']['r'] = self.π * (self.Q @ α - δ)
        gout['obj']['δ'] = -self.π * r
                
        gout['nonlin_con_1'] = {}
        gout['nonlin_con_1']['u'] = self.K
        D1 = spdiags(-np.cos(θ),0,self.dim_r,self.dim_r)
        D2 = spdiags(-np.sin(θ),0,self.dim_r,self.dim_r)
        gout['nonlin_con_1']['r'] = vstack([D1,D2])
        D3 = spdiags(r*np.sin(θ),0,self.dim_r,self.dim_r)
        D4 = spdiags(-r*np.cos(θ),0,self.dim_r,self.dim_r)
        gout['nonlin_con_1']['θ'] = vstack([D3,D4])
        
        
        # # print(f'delta={δ}')
        gout['nonlin_con_2'] = {}
        gout['nonlin_con_2']['q'] = Diagonal(np.ones(self.dim_q)).tosparse()
        gout['nonlin_con_2']['δ'] = vstack([D1,D2])
        D5 = spdiags(δ*np.sin(θ),0,self.dim_r,self.dim_r)
        D6 = spdiags(-δ*np.cos(θ),0,self.dim_r,self.dim_r)
        gout['nonlin_con_2']['θ'] = vstack([D5,D6])
        
        fail = False
        
        return gout, fail
    
    def complementarity(self,xdict):
        α = xdict['α']
        r = xdict['r']
        δ = xdict['δ']
        G1 = r
        G2 = self.Q @ α - δ
        return np.linalg.norm(np.minimum(G1,G2),np.inf),np.dot(r, self.Q @ α - δ)
        
    def solve(self,print_level=0):
        # print(f'Solving relaxed NLP with π={self.π} and μ={self.μ} tol_p={self.tol_p}')
        opt = OPT('IPOPT',
            options={
                'print_level':print_level,
                # 'acceptable_tol':1e-6,
                # 'acceptable_iter':5,
                'linear_solver':'ma86',
                'max_iter':1000,
                'mu_init':self.μ,
                'mu_strategy':'monotone',
                # 'mu_target':self.μ,
                'dual_inf_tol':self.tol_p,
                'constr_viol_tol':self.tol_p,
                'compl_inf_tol':self.tol_p,
                # 'tol':1e-3,
                # 'check_derivatives_for_naninf':'yes',
            })
        sol = opt(self.optProb,sens='FD')
        # sol = opt(self.optProb,sens=self.usr_jac,sensMode='pgc')
        param = sol.xStar['α']
        rec = sol.xStar['u']
        extra = {
            'compl':self.complementarity(sol.xStar)[0],
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
    
def solve_mpcc(true_img,noisy_img,K,R,Q,π_init=1.0,σ=10,ν=10,μ=0.1,γ=0.4,κ=0.2,α_size=1,tik=1e-3,k_max=100,tol=1e-3,α_max=1.0):
    # Initialization
    extras = []
    π = π_init
    k = 1
    last_obj = 0
    print(K)
    xdict_init = {
        'u':np.real(R.H * noisy_img.ravel()).ravel(),
        'q':np.zeros(K.shape[0]),
        'α':np.zeros(α_size),
        'r':0.0001*np.ones(K.shape[0]),
        'δ':0.0001*np.ones(K.shape[0]),
        'θ':0.0001*np.ones(K.shape[0])
    }
    tv_mpcc_relaxed = TV_MPCC_RELAXED(true_img,noisy_img,K,R,Q,xdict_init,π=π,ν=ν,α_size=α_size,μ=μ,tik=tik,α_max=α_max)
    α,u,q,r,δ,θ,extra = tv_mpcc_relaxed.solve()
    extras.append(extra)
    
    print(f'{"Iter":>5}\t{"Termination_status":>15}\t{"Objective":>15}\t{"MPCC_compl":>15}\t{"lg(mu)":>15}\t{"π":>15}\t{"NLP_iter":>15}\n')
    print(f'{k}\t{extra["optInform"]["text"]}\t{extra["fStar"][0]:>15}\t{extra["compl"]:>15} {np.log(μ)} {π} {extra["info"]["userObjCalls"]}')
    
    # Main loop
    while k < k_max:
        tol_c = μ**γ
        tv_mpcc_relaxed = TV_MPCC_RELAXED(true_img,noisy_img,K,R,Q,xdict_init,π=π,ν=ν,μ=μ,tik=tik,α_size=α_size,α_max=α_max)
        α,u,q,r,δ,θ,extra = tv_mpcc_relaxed.solve()
        
        
        if extra['compl'] <= tol_c:
            k += 1
            print(f'{k}\t{extra["optInform"]["text"]}\t{extra["fStar"][0]:>15}\t{extra["compl"]:>15} {np.log10(μ)} {π} {extra["info"]["userObjCalls"]}')
            
            if np.abs(extra["fStar"][0]-last_obj) < tol:
                print(f'Obtained solution satisfies the complementarity condition at {extra["compl"]} at {k} iterations')
                break
            else:
                last_obj = extra["fStar"][0]
            
            μ *= κ
            xdict_init = {
                'u':u,
                'q':q,
                'α':α,
                'r':r,
                'δ':δ,
                'θ':θ
            }
            
        else:
            if π < 1e14:
                π *= σ
            else:
                print(f'Couldnt find a suitable value for π.')
                break
    print(f'{α=}')
    return α,u,q,r,δ,θ,extras

def solve_mpcc_one_shot(true_img,noisy_img,K,R,Q,π_init=1e4,σ=10,ν=10,μ=0.1,γ=0.4,κ=0.2,α_size=1,tik=1e-3,k_max=100,tol=1e-3,α_max=1.0):
    # Initialization
    extras = []
    π = π_init
    k = 1
    xdict_init = {
        'u':np.real(R.H * noisy_img.ravel()).ravel(),
        'q':np.zeros(K.shape[0]),
        'α':0.1*np.ones(α_size),
        'r':0.1*np.ones(K.shape[0]),
        'δ':0.1*np.ones(K.shape[0]),
        'θ':0.1*np.ones(K.shape[0])
    }
    tv_mpcc_relaxed = TV_MPCC_RELAXED(true_img,noisy_img,K,R,Q,xdict_init,π=π,ν=ν,α_size=α_size,μ=μ,tik=tik,α_max=α_max)
    α,u,q,r,δ,θ,extra = tv_mpcc_relaxed.solve(print_level=5)
    extras.append(extra)
    
    print(f'{"Iter":>5}\t{"Termination_status":>15}\t{"Objective":>15}\t{"MPCC_compl":>15}\t{"lg(mu)":>15}\t{"π":>15}\t{"NLP_iter":>15}\n')
    print(f'{k}\t{extra["optInform"]["text"]}\t{extra["fStar"][0]:>15}\t{extra["compl"]:>15} {np.log(μ)} {π} {extra["info"]["userObjCalls"]}')
    
    print(f'{α=}')
    
    return α,u,q,r,δ,θ,extras
    
        
        
    
    