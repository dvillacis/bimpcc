import numpy as np
from pyoptsparse import Optimization, OPT
from scipy.sparse import spdiags, vstack, dia_matrix, csr_matrix,eye, hstack
from pylops import Diagonal, VStack, Zero
from bimpcc.tv_mpcc import solve_mpcc as tv_mpcc_solve

class MPCC_RELAXED:
    def __init__(self, true_img, noisy_img, K, E, R, Q,S, tik=0.1, alpha_max=1.0, alpha_size=1, beta_max = 1.0, beta_size=1, tol=1e-5, max_iter=1000, init_alpha=None, init_beta=None, init_v=None, init_w=None, init_q=None, init_r=None, init_delta=None, init_theta=None, init_phi=None, init_varphi=None, init_tau=None,init_Lambda=None, init_rho=None):
        print(K.shape)
        
        self.dim_q = K.shape[0]
        self.dim_v = K.shape[1]
        self.dim_alpha = alpha_size
        self.dim_beta = beta_size
        self.dim_beta = beta_size
        self.dim_r = K.shape[0] // 2
        self.dim_tau = E.shape[0] // 3
        
        print(f'dim_q={self.dim_q} dim_v={self.dim_v} dim_alpha={self.dim_alpha}, dim_r={self.dim_r}, dim_tau={self.dim_tau}')
        
        ϵ = 1e-3
        
        # Setting initial values
        if init_alpha is None:
            init_alpha = ϵ*np.ones(alpha_size)
        if init_beta is None:
            init_beta = ϵ*np.ones(beta_size)
        if init_v is None:
            init_v =np.real(R.H * noisy_img.ravel()).ravel()
        if init_w is None:
            init_w = np.zeros(self.dim_q)
        if init_q is None:
            init_q = np.zeros(self.dim_q)
        if init_r is None:
            init_r = ϵ*np.ones(self.dim_r)
        if init_delta is None:
            init_delta = 2*ϵ*np.ones(self.dim_r)
        if init_theta is None:
            init_theta = ϵ*np.ones(self.dim_r)
        if init_phi is None:
            init_phi = ϵ*np.ones(self.dim_tau)
        if init_varphi is None:
            init_varphi = ϵ*np.ones(self.dim_tau)
        if init_tau is None:
            init_tau = 2*ϵ*np.ones(self.dim_tau)
        if init_Lambda is None:
            init_Lambda = ϵ*np.ones(3*self.dim_tau)
        if init_rho is None:
            init_rho = ϵ*np.ones(self.dim_tau)
            
        # print(noisy_img.shape)
        self.max_iter = max_iter
        self.m,self.n = true_img.shape
        self.true_img = true_img
        self.noisy_img = noisy_img
        print(f'Image size: {self.m}x{self.n} Noisy size: {self.noisy_img.shape}')
        self.tik = tik
        self.alpha_max = alpha_max
        self.K = K
        self.E = E
        self.R = R  # Imaging Forward Model
        self.Q = Q # Patch Operator
        self.S = S # Patch Operator Second Order
        
        # Define the optimization problem
        self.optProb = Optimization('TV MPCC Problem',self.objfun)
        # Design variables
        self.optProb.addVarGroup('v',self.dim_v,lower=0,upper=None,value=init_v)
        self.optProb.addVarGroup('w',self.dim_q,lower=0,upper=None,value=init_w)
        self.optProb.addVarGroup('q',self.dim_q,value=init_q)
        self.optProb.addVarGroup('Λ',3*self.dim_tau,value=init_Lambda)
        self.optProb.addVarGroup('α',self.dim_alpha,lower=1e-8,upper=alpha_max,value=init_alpha)
        self.optProb.addVarGroup('β',self.dim_beta,lower=1e-8,upper=beta_max,value=init_beta)
        self.optProb.addVarGroup('r',self.dim_r,lower=1e-8,value=init_r)
        self.optProb.addVarGroup('ρ',self.dim_tau,lower=1e-8,value=init_rho)
        self.optProb.addVarGroup('δ',self.dim_r,lower=1e-8,value=init_delta)
        self.optProb.addVarGroup('τ',self.dim_tau,lower=1e-8,value=init_tau)
        self.optProb.addVarGroup('ϕ',self.dim_tau,lower=-np.pi,upper=np.pi,value=init_phi)
        self.optProb.addVarGroup('φ',self.dim_tau,lower=-np.pi,upper=np.pi,value=init_varphi)
        self.optProb.addVarGroup('θ',self.dim_r,lower=-np.pi,upper=np.pi,value=init_theta)
        # Nonlinear constraints
        jac_con_1_v = self.K.tosparse()
        jac_con_1_w = Diagonal(np.ones(self.dim_q)).tosparse()
        jac_con_1_r = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        jac_con_1_theta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        self.optProb.addConGroup('nonlin_con_1',self.dim_q,lower=0,upper=0,wrt=['v','w','r','θ'],jac={'v':jac_con_1_v,'w':jac_con_1_w,'r':jac_con_1_r,'θ':jac_con_1_theta})
        
        jac_con_2_q = Diagonal(np.ones(self.dim_q)).tosparse()
        jac_con_2_delta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        jac_con_2_theta = VStack([Diagonal(np.ones(self.dim_r)),Diagonal(np.ones(self.dim_r))]).tosparse()
        self.optProb.addConGroup('nonlin_con_2',self.dim_q,lower=0,upper=0,wrt=['q','δ','θ'],jac={'q':jac_con_2_q,'δ':jac_con_2_delta,'θ':jac_con_2_theta})
        
        jac_con_3_w = self.E.tosparse()
        jac_con_3_rho = VStack([Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau))]).tosparse()
        jac_con_3_phi = VStack([Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau))]).tosparse()
        jac_con_3_varphi = VStack([Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau)),Zero(self.dim_tau)]).tosparse()
        self.optProb.addConGroup('nonlin_con_3',3*self.dim_tau,lower=0,upper=0,wrt=['w','ρ','ϕ','φ'],jac={'w':jac_con_3_w,'ρ':jac_con_3_rho,'ϕ':jac_con_3_phi,'φ':jac_con_3_varphi})
        
        jac_con_4_Lambda = Diagonal(np.ones(3*self.dim_tau)).tosparse()
        jac_con_4_tau = VStack([Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau))]).tosparse()
        jac_con_4_phi = VStack([Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau))]).tosparse()
        jac_con_4_varphi = VStack([Diagonal(np.ones(self.dim_tau)),Diagonal(np.ones(self.dim_tau)),Zero(self.dim_tau)]).tosparse()
        self.optProb.addConGroup('nonlin_con_4',3*self.dim_tau,lower=0,upper=0,wrt=['Λ','τ','ϕ','φ'],jac={'Λ':jac_con_4_Lambda,'τ':jac_con_4_tau,'ϕ':jac_con_4_phi,'φ':jac_con_4_varphi})
        
        jac_con_5_alpha = self.Q.tosparse()
        jac_con_5_r = Diagonal(np.ones(self.dim_r)).tosparse()
        jac_con_5_delta = Diagonal(np.ones(self.dim_r)).tosparse()
        self.optProb.addConGroup('nonlin_con_5',self.dim_r,lower=None,upper=tol,wrt=['α','r','δ'],jac={'α':jac_con_5_alpha,'r':jac_con_5_r,'δ':jac_con_5_delta})
        
        jac_con_6_beta = self.S.tosparse()
        jac_con_6_rho = Diagonal(np.ones(self.dim_tau)).tosparse()
        jac_con_6_tau = Diagonal(np.ones(self.dim_tau)).tosparse()
        self.optProb.addConGroup('nonlin_con_6',self.dim_tau,lower=None,upper=tol,wrt=['β','ρ','τ'],jac={'β':jac_con_6_beta,'ρ':jac_con_6_rho,'τ':jac_con_6_tau})
        
        # Linear constraints
        jac_v = np.real((self.R.H*self.R).tosparse())
        rhs = np.real(self.R.H * self.noisy_img.ravel())
        jac_q = self.K.tosparse().transpose()
        self.optProb.addConGroup('lin_con1',self.dim_v,lower=rhs.ravel(),upper=rhs.ravel(),wrt=['v','q'],jac={'v':jac_v,'q':jac_q},linear=True)
        
        jac_q = Diagonal(np.ones(self.dim_q)).tosparse()
        jac_Lambda = self.E.T.tosparse()
        self.optProb.addConGroup('lin_con2',self.dim_q,lower=0,upper=0,wrt=['q','Λ'],jac={'q':jac_q,'Λ':jac_Lambda},linear=True)

        jac_alpha = self.Q.tosparse()
        jac_delta = Diagonal(-np.ones(self.dim_r)).tosparse()
        self.optProb.addConGroup('lin_con3',self.dim_r,lower=0,wrt=['α','δ'],jac={'α':jac_alpha,'δ':jac_delta},linear=True)
        
        jac_beta = self.S.tosparse()
        jac_tau = Diagonal(-np.ones(self.dim_tau)).tosparse()
        self.optProb.addConGroup('lin_con4',self.dim_tau,lower=0,wrt=['β','τ'],jac={'β':jac_beta,'τ':jac_tau},linear=True)
        
    def objfun(self,xdict):
        # print(f'Objective function called. {xdict["u"].shape}')
        v = xdict['v']
        w = xdict['w']
        q = xdict['q']
        α = xdict['α']
        β = xdict['β']
        r = xdict['r']
        δ = xdict['δ']
        θ = xdict['θ']
        ρ = xdict['ρ']
        ϕ = xdict['ϕ']
        φ = xdict['φ']
        Λ = xdict['Λ']
        τ = xdict['τ']
        
        funcs = {}
        funcs['obj'] = 0.5 * np.linalg.norm(v-self.true_img.ravel())**2 + 0.5 * self.tik * np.linalg.norm(α)**2 #+ 0.5 * self.tik * np.linalg.norm(β)**2
        
        funcs['nonlin_con_1'] = self.K @ v - w - np.concatenate([r * np.cos(θ),r * np.sin(θ)])
        funcs['nonlin_con_2'] = q - np.concatenate([δ * np.cos(θ),δ * np.sin(θ)])
        funcs['nonlin_con_3'] = self.E * w - np.concatenate([ρ * np.sin(ϕ)* np.cos(φ),ρ * np.sin(ϕ)* np.sin(φ),ρ * np.cos(ϕ)])
        funcs['nonlin_con_4'] = Λ - np.concatenate([τ * np.sin(ϕ)* np.cos(φ),τ * np.sin(ϕ)* np.sin(φ),τ * np.cos(ϕ)])
        funcs['nonlin_con_5'] = r * (self.Q.matvec(α)-δ)
        funcs['nonlin_con_6'] = ρ * (self.S.matvec(β)-τ)
        fail = False
        return funcs, fail
    
    def usr_jac(self,xdict,fdict):
        # print(f'Jacobian called. {xdict["u"].shape}')
        v = xdict['v']
        r = xdict['r']
        ρ = xdict['ρ']
        α = xdict['α']
        β = xdict['β']
        θ = xdict['θ']
        ϕ = xdict['ϕ']
        φ = xdict['φ']
        δ = xdict['δ']
        τ = xdict['τ']

        gout = {}
        
        gout['obj'] = {}
        gout['obj']['v'] = v-self.true_img.ravel()
        gout['obj']['α'] = self.tik * α
        # gout['obj']['β'] = self.tik * β
        
        gout['nonlin_con_1'] = {}
        gout['nonlin_con_1']['v'] = self.K.tosparse()
        D1 = spdiags(-np.cos(θ),0,self.dim_r,self.dim_r)
        D2 = spdiags(-np.sin(θ),0,self.dim_r,self.dim_r)
        gout['nonlin_con_1']['w'] = spdiags(-np.ones(self.dim_q),0,self.dim_q,self.dim_q)
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
        gout['nonlin_con_3']['w'] = self.E.tosparse()
        gout['nonlin_con_3']['ρ'] = VStack([Diagonal(-np.sin(ϕ)*np.cos(φ)),Diagonal(-np.sin(ϕ)*np.sin(φ)),Diagonal(-np.cos(ϕ))]).tosparse()
        gout['nonlin_con_3']['ϕ'] = VStack([Diagonal(-ρ*np.cos(ϕ)*np.cos(φ)),Diagonal(-ρ*np.cos(ϕ)*np.sin(φ)),Diagonal(ρ*np.sin(ϕ))]).tosparse()
        gout['nonlin_con_3']['φ'] = VStack([Diagonal(ρ*np.sin(ϕ)*np.sin(φ)),Diagonal(-ρ*np.sin(ϕ)*np.cos(φ)),Zero(self.dim_tau)]).tosparse()
        
        gout['nonlin_con_4'] = {}
        gout['nonlin_con_4']['Λ'] = Diagonal(np.ones(3*self.dim_tau)).tosparse()
        gout['nonlin_con_4']['τ'] = VStack([Diagonal(-np.sin(ϕ)*np.cos(φ)),Diagonal(-np.sin(ϕ)*np.sin(φ)),Diagonal(-np.cos(ϕ))]).tosparse()
        gout['nonlin_con_4']['ϕ'] = VStack([Diagonal(-τ*np.cos(ϕ)*np.cos(φ)),Diagonal(-τ*np.cos(ϕ)*np.sin(φ)),Diagonal(τ*np.sin(ϕ))]).tosparse()
        gout['nonlin_con_4']['φ'] = VStack([Diagonal(τ*np.sin(ϕ)*np.sin(φ)),Diagonal(-τ*np.sin(ϕ)*np.cos(φ)),Zero(self.dim_tau)]).tosparse()
        
        gout['nonlin_con_5'] = {}
        gout['nonlin_con_5']['α'] = Diagonal(r).tosparse() @ self.Q.tosparse()
        gout['nonlin_con_5']['r'] = spdiags(self.Q.matvec(α)-δ,0,self.dim_r,self.dim_r)
        gout['nonlin_con_5']['δ'] = spdiags(-r,0,self.dim_r,self.dim_r)
        
        # print(f'τ={τ}')
        # print(f'β={β}')
        gout['nonlin_con_6'] = {}
        gout['nonlin_con_6']['β'] = Diagonal(ρ).tosparse() @ self.S.tosparse()
        gout['nonlin_con_6']['ρ'] = Diagonal(self.S.matvec(β)-τ).tosparse()
        gout['nonlin_con_6']['τ'] = Diagonal(-ρ).tosparse()
        
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
            'linear_solver':'ma77',
            'hessian_approximation_space':'all-variables'
            # 'expect_infeasible_problem':'yes'
            # 'nlp_scaling_method':'gradient-based',
            # 'nlp_scaling_max_gradient': 1.0
        })
        sol = opt(self.optProb,sens=self.usr_jac,sensMode='pgc')
        # sol = opt(self.optProb,sens='FD')
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
        return sol.xStar['v'], sol.xStar['w'], sol.xStar['q'], sol.xStar['Λ'], sol.xStar['α'], sol.xStar['β'], sol.xStar['r'], sol.xStar['δ'], sol.xStar['ρ'], sol.xStar['τ'], sol.xStar['θ'], sol.xStar['ϕ'], sol.xStar['φ'], extra
    
def solve_mpcc(true_img, noisy_img, K, E, R, Q, S, tik=0.1, alpha_max=1.0, alpha_size=1, tol_max=1000.0, max_iter=3000):
    #Solve TV MPCC
    # print(f'********* Solve TV with tol={tol_max} **********')
    # α,v,q,r,δ,θ,extra = tv_mpcc_solve(true_img=true_img,noisy_img=noisy_img,K=K,R=R,Q=Q,tik=tik,alpha_max=alpha_max,tol_max=tol_max,max_iter=max_iter)
    
    
    extras = []
    print(f'********* Solve with tol={tol_max} **********')
    mpcc = MPCC_RELAXED(true_img=true_img,noisy_img=noisy_img,K=K,E=E,R=R,Q=Q,S=S,alpha_size=alpha_size,tik=tik,alpha_max=alpha_max,tol=tol_max,max_iter=max_iter)#,init_alpha=α, init_beta=2*α,init_v=v.ravel(),init_q=q, init_r=r, init_delta=δ, init_theta=θ)
    v,w,q,Λ,α,β,r,δ,ρ,τ,θ,ϕ,φ,extra = mpcc.solve()
    extras.append(extra)
    
    t = tol_max/10
    
    for i in np.arange(3):
        print(f'********* Solve with tol={t} **********')
        mpcc = MPCC_RELAXED(true_img=true_img,noisy_img=noisy_img,K=K,E=E,R=R,Q=Q,S=S,alpha_size=alpha_size,tik=tik,alpha_max=alpha_max,tol=t,max_iter=max_iter,init_alpha=α,init_beta=β,init_v=v.ravel(),init_w=w.ravel(),init_q=q, init_r=r, init_delta=δ, init_theta=θ,init_phi=ϕ,init_varphi=φ,init_tau=τ,init_rho=ρ,init_Lambda=Λ)
        v,w,q,Λ,α,β,r,δ,ρ,τ,θ,ϕ,φ,extra = mpcc.solve()
        extras.append(extra)
        
        t = t / 10
    
    return v,w,q,Λ,α,β,r,δ,ρ,τ,θ,ϕ,φ,extras