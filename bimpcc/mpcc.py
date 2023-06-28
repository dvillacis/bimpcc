import numpy as np
from scipy.sparse import spdiags, vstack, dia_matrix, csr_matrix,eye, hstack
from pyoptsparse import Optimization, OPT

class MPCC_RELAXED:
    def __init__(self, true_img, noisy_img, Kx, Ky, R, Q, tik=0.1, alpha_max=1.0, alpha_size=1, tol=1e-5, max_iter=1000, init_alpha=None, init_u=None, init_q=None, init_r=None, init_delta=None, init_theta=None):
        
        # Setting initial values
        if init_alpha is None:
            init_alpha = np.zeros(alpha_size)
        if init_u is None:
            init_u =np.real(R.H * noisy_img.ravel()).ravel()
        if init_q is None:
            init_q = np.zeros(2*Kx.shape[0])
        if init_r is None:
            init_r = 0.0001*np.ones(Kx.shape[0])
        if init_delta is None:
            init_delta = 0.0001*np.ones(Kx.shape[0])
        if init_theta is None:
            init_theta = 0.0001*np.ones(Kx.shape[0])
            
        print(noisy_img.shape)
        self.max_iter = max_iter
        self.m,self.n = true_img.shape
        self.true_img = true_img
        self.noisy_img = noisy_img
        print(f'Image size: {self.m}x{self.n} Noisy size: {self.noisy_img.shape}')
        self.tik = tik
        self.alpha_max = alpha_max
        self.Kx = Kx
        self.Ky = Ky
        self.R = R  # Imaging Forward Model
        self.Q = Q # Patch Operator
        self.Qsp = Q.tosparse()
        self.dim_q = Kx.shape[0]
        self.dim_u = Kx.shape[1]
        self.dim_alpha = alpha_size
        # Define the optimization problem
        self.optProb = Optimization('MPCC Problem',self.objfun)
        # Design variables
        self.optProb.addVarGroup('u',self.dim_u,lower=0,upper=None,value=init_u)
        self.optProb.addVarGroup('q',2*self.dim_q,value=init_q)
        self.optProb.addVarGroup('alpha',self.dim_alpha,lower=0,upper=alpha_max,value=init_alpha)
        self.optProb.addVarGroup('r',self.dim_q,lower=0,value=init_r)
        self.optProb.addVarGroup('delta',self.dim_q,lower=0,value=init_delta)
        self.optProb.addVarGroup('theta',self.dim_q,value=init_theta)
        # Nonlinear constraints
        jac_con_1_u = Kx.tosparse()
        jac_con_1_r = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        jac_con_1_theta = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        self.optProb.addConGroup('nonlin_con_1',self.dim_q,lower=0,upper=0,wrt=['u','r','theta'],jac={'u':jac_con_1_u,'r':jac_con_1_r,'theta':jac_con_1_theta})
        
        jac_con_2_u = Ky.tosparse()
        jac_con_2_r = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        jac_con_2_theta = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        self.optProb.addConGroup('nonlin_con_2',self.dim_q,lower=0,upper=0,wrt=['u','r','theta'],jac={'u':jac_con_2_u,'r':jac_con_2_r,'theta':jac_con_2_theta})
        
        jac_con_3_alpha = csr_matrix(dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)))*csr_matrix(self.Qsp)
        jac_con_3_r = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        jac_con_3_delta = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        self.optProb.addConGroup('nonlin_con_3',self.dim_q,lower=None,upper=tol,wrt=['alpha','r','delta'],jac={'alpha':jac_con_3_alpha,'r':jac_con_3_r,'delta':jac_con_3_delta})
        
        jac_con_4_q = dia_matrix(np.eye(2*self.dim_q),(2*self.dim_q,2*self.dim_q)).tolil()
        jac_con_4_delta = vstack([dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)),dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q))])
        jac_con_4_theta = vstack([dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)),dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q))])
        self.optProb.addConGroup('nonlin_con_4',2*self.dim_q,lower=0,upper=0,wrt=['q','delta','theta'],jac={'q':jac_con_4_q,'delta':jac_con_4_delta,'theta':jac_con_4_theta})
        
        # Linear constraints
        jac_u = np.real((self.R.H*self.R).tosparse()) + 0.001*eye(self.dim_u)
        # print(jac_u.shape)
        rhs = np.real(self.R.H * self.noisy_img.ravel())
        # print(rhs)
        jac_q = hstack([Kx.tosparse().transpose(),Ky.tosparse().transpose()])
        self.optProb.addConGroup('lin_con1',self.dim_u,lower=rhs.ravel(),upper=rhs.ravel(),wrt=['u','q'],jac={'u':jac_u,'q':jac_q},linear=True)

        jac_alpha = self.Qsp
        # jac_alpha = np.ones(self.dim_q).reshape((self.dim_q,1))
        jac_delta = dia_matrix(-np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        self.optProb.addConGroup('lin_con2',self.dim_q,lower=np.zeros(self.dim_q),wrt=['alpha','delta'],jac={'alpha':jac_alpha,'delta':jac_delta},linear=True)
        
    def objfun(self,xdict):
        u = xdict['u']
        q = xdict['q']
        alpha = xdict['alpha']
        r = xdict['r']
        delta = xdict['delta']
        theta = xdict['theta']
        funcs = {}
        funcs['obj'] = 0.5 * np.linalg.norm(u-self.true_img.ravel())**2 + self.tik * np.linalg.norm(alpha)**2
        funcs['nonlin_con_1'] = self.Kx * u - r * np.cos(theta)
        funcs['nonlin_con_2'] = self.Ky * u - r * np.sin(theta)
        funcs['nonlin_con_3'] = r * (self.Q.matvec(alpha)-delta)
        funcs['nonlin_con_4'] = q - np.concatenate([delta * np.cos(theta),delta * np.sin(theta)])
        fail = False
        return funcs, fail
    
    def usr_jac(self,xdict,fdict):
        u = xdict['u']
        r = xdict['r']
        q = xdict['q']
        alpha = xdict['alpha']
        theta = xdict['theta']
        delta = xdict['delta']
        # print(spdiags(alpha-delta,0,dim_q,dim_q).tolil())
        gout = {}
        
        gout['obj'] = {}
        gout['obj']['u'] = u-self.true_img.ravel()
        gout['obj']['alpha'] = 2 * self.tik * alpha
        gout['nonlin_con_1'] = {}
        gout['nonlin_con_1']['u'] = self.Kx.tosparse()
        gout['nonlin_con_1']['r'] = spdiags(-np.cos(theta),0,self.dim_q,self.dim_q).tolil()
        gout['nonlin_con_1']['theta'] = spdiags(r*np.sin(theta),0,self.dim_q,self.dim_q).tolil()
        
        gout['nonlin_con_2'] = {}
        gout['nonlin_con_2']['u'] = self.Ky.tosparse()
        gout['nonlin_con_2']['r'] = spdiags(-np.sin(theta),0,self.dim_q,self.dim_q).tolil()
        gout['nonlin_con_2']['theta'] = spdiags(-r*np.cos(theta),0,self.dim_q,self.dim_q).tolil()
        
        gout['nonlin_con_3'] = {}
        gout['nonlin_con_3']['alpha'] = csr_matrix(spdiags(r,0,self.dim_q,self.dim_q))*csr_matrix(self.Qsp)
        gout['nonlin_con_3']['r'] = spdiags(r,0,self.dim_q,self.dim_q).tolil()
        gout['nonlin_con_3']['delta'] = spdiags(-r,0,self.dim_q,self.dim_q).tolil()
        
        gout['nonlin_con_4'] = {}
        gout['nonlin_con_4']['q'] = spdiags(np.ones(2*self.dim_q),0,2*self.dim_q,2*self.dim_q).tolil()
        gout['nonlin_con_4']['delta'] = vstack([spdiags(-np.cos(theta),0,self.dim_q,self.dim_q),spdiags(-np.sin(theta),0,self.dim_q,self.dim_q)])
        gout['nonlin_con_4']['theta'] = vstack([spdiags(delta*np.sin(theta),0,self.dim_q,self.dim_q),spdiags(-delta*np.cos(theta),0,self.dim_q,self.dim_q)])
        
        fail = False
        
        return gout, fail
    
    def solve(self,print_sparsity=True):
        if print_sparsity: self.optProb.printSparsity()
        self.optProb.addObj('obj')
        opt = OPT('IPOPT',options={
            'print_level':5,
            'acceptable_tol':1e-1,
            'acceptable_iter':3,
            'max_iter':self.max_iter,
            # 'nlp_scaling_method':'gradient-based',
            # 'nlp_scaling_max_gradient': 1.0
        })
        sol = opt(self.optProb,sens=self.usr_jac,sensMode='pgc')
        param = sol.xStar['alpha']
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
        return param, rec, sol.xStar['q'], sol.xStar['r'], sol.xStar['delta'], sol.xStar['theta'], extra
    
def solve_mpcc(true_img, noisy_img, Kx, Ky, R, Q, tik=0.1, alpha_max=1.0, alpha_size=1, tol_max=1000.0, tol_min=0.1, max_iter=2000):
    extras = []
    print(f'********* Solve with tol={tol_max} **********')
    mpcc = MPCC_RELAXED(true_img=true_img,noisy_img=noisy_img,Kx=Kx,Ky=Ky,R=R,Q=Q,alpha_size=alpha_size,tik=tik,alpha_max=alpha_max,tol=tol_max,max_iter=max_iter)
    param,sol,q,r,delta,theta,extra = mpcc.solve()
    extras.append(extra)
    
    step = -(tol_max-tol_min)/10
    
    for t in np.arange(tol_max-step,tol_min+step+0.001,step):
        print(f'********* Solve with tol={t} **********')
        mpcc = MPCC_RELAXED(true_img=true_img,noisy_img=noisy_img,Kx=Kx,Ky=Ky,R=R,Q=Q,alpha_size=alpha_size,tik=tik,alpha_max=alpha_max,tol=t,max_iter=max_iter,init_alpha=param,init_u=sol.ravel(),init_q=q, init_r=r, init_delta=delta, init_theta=theta)
        param,sol,q,r,delta,theta,extra = mpcc.solve()
        extras.append(extra)
    
    return param,sol,q,r,delta,theta,extras
        