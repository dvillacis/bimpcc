import numpy as np
from scipy.sparse import spdiags, vstack, dia_matrix, csr_matrix,eye
from pyoptsparse import Optimization, OPT

class MPCC:
    def __init__(self, true_img, noisy_img, Kx, Ky, R, tik=1, alpha_max=1.0, alpha_size=1):
        self.m,self.n = true_img.shape
        self.true_img = true_img
        self.noisy_img = noisy_img
        print(f'Image size: {self.m}x{self.n} Noisy size: {self.noisy_img.shape}')
        self.tik = tik
        self.alpha_max = alpha_max
        self.Kx = Kx
        self.Ky = Ky
        self.R = R  # Imaging Forward Model
        self.dim_q = Kx.shape[0]
        self.dim_u = Kx.shape[1]
        self.dim_alpha = alpha_size
        # Define the optimization problem
        self.optProb = Optimization('Scalar TV Denoising',self.objfun)
        # Design variables
        self.optProb.addVarGroup('u',self.dim_u,lower=0,upper=None,value=self.noisy_img.ravel())
        self.optProb.addVarGroup('q',2*self.dim_q,value=0.0000*np.ones(2*self.dim_q))
        self.optProb.addVarGroup('alpha',self.dim_alpha,lower=0,upper=alpha_max,value=0.0001*np.ones(self.dim_alpha))
        self.optProb.addVarGroup('r',self.dim_q,lower=0,value=0.0001*np.ones(self.dim_q))
        self.optProb.addVarGroup('delta',self.dim_q,lower=0,value=0.0001*np.ones(self.dim_q))
        self.optProb.addVarGroup('theta',self.dim_q,value=0.0001*np.ones(self.dim_q))
        # Nonlinear constraints
        jac_con_1_u = Kx.tosparse()
        jac_con_1_r = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        jac_con_1_theta = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        self.optProb.addConGroup('nonlin_con_1',self.dim_q,lower=0,upper=0,wrt=['u','r','theta'],jac={'u':jac_con_1_u,'r':jac_con_1_r,'theta':jac_con_1_theta})
        
        jac_con_2_u = Ky.tosparse()
        jac_con_2_r = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        jac_con_2_theta = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        self.optProb.addConGroup('nonlin_con_2',self.dim_q,lower=0,upper=0,wrt=['u','r','theta'],jac={'u':jac_con_2_u,'r':jac_con_2_r,'theta':jac_con_2_theta})
        
        jac_con_3_alpha = np.ones(self.dim_q).reshape((self.dim_q,1))
        jac_con_3_r = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        jac_con_3_delta = dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)).tolil()
        self.optProb.addConGroup('nonlin_con_3',self.dim_q,lower=None,upper=0.1,wrt=['alpha','r','delta'],jac={'alpha':jac_con_3_alpha,'r':jac_con_3_r,'delta':jac_con_3_delta})
        
        jac_con_4_q = dia_matrix(np.eye(2*self.dim_q),(2*self.dim_q,2*self.dim_q)).tolil()
        jac_con_4_delta = vstack([dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)),dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q))])
        jac_con_4_theta = vstack([dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q)),dia_matrix(np.eye(self.dim_q),(self.dim_q,self.dim_q))])
        self.optProb.addConGroup('nonlin_con_4',2*self.dim_q,lower=0,upper=0,wrt=['q','delta','theta'],jac={'q':jac_con_4_q,'delta':jac_con_4_delta,'theta':jac_con_4_theta})
        
        # Linear constraints
        jac_u = (self.R.T*self.R).tosparse() + 0.001*eye(self.dim_u)
        jac_q = csr_matrix(np.hstack([Kx.transpose().todense(),Ky.transpose().todense()]))
        self.optProb.addConGroup('lin_con1',self.dim_u,lower=self.noisy_img.ravel(),upper=self.noisy_img.ravel(),wrt=['u','q'],jac={'u':jac_u,'q':jac_q},linear=True)

        jac_alpha = np.ones(self.dim_q).reshape((self.dim_q,1))
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
        funcs['obj'] = 0.5 * np.linalg.norm(u-self.true_img)**2 + self.tik * alpha**2
        funcs['nonlin_con_1'] = self.Kx * u - r * np.cos(theta)
        funcs['nonlin_con_2'] = self.Ky * u - r * np.sin(theta)
        funcs['nonlin_con_3'] = r * (alpha-delta)
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
        gout['obj']['u'] = u-self.true_img
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
        gout['nonlin_con_3']['alpha'] = r.reshape((self.dim_q,1))
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
        opt = OPT('IPOPT',options={'print_level':5,'acceptable_tol':1e-2,'acceptable_iter':5})
        sol = opt(self.optProb,sens_type=self.usr_jac,store_hst=True)
        return sol.xStar['alpha'],np.clip(sol.xStar['u'].reshape((self.m,self.n)),0,1)
