import numpy as np
from pyoptsparse import Optimization, OPT
from scipy.sparse import spdiags
from bimpcc.tv_two_dim import solve_mpcc as solve_tv
from bimpcc.utils import sparse_to_coosparse, diag_to_coodiag, block_diag_to_coosparse, triple_block_diag_to_coosparse

class TGV_2D_MPCC:
    def __init__(self,true_img,noisy_img,K,E,R,Q,S,ϵ,xdict,α_size=1,β_size=1) -> None:
        self.true_img = true_img.ravel()
        self.noisy_img = noisy_img.ravel()
        self.E = E.tosparse()
        self.K = K.tosparse()
        self.R = R.tosparse()
        self.Q = Q.tosparse()
        self.S = S.tosparse()
        
        self.ϵ = ϵ
        
        self.xdict = xdict
        
        self.optProb = Optimization('TGV_2D_MPCC',self.objfun)
        
        # Dimensions
        self.dim_v = len(self.true_img)
        self.dim_q = self.K.shape[0]
        self.dim_r = self.K.shape[0] // 2
        self.dim_Λ = self.E.shape[0]
        self.dim_ρ = self.E.shape[0] // 3
        self.dim_α = α_size
        self.dim_β = β_size 
        print(f'dim_v = {self.dim_v}, dim_q = {self.dim_q}, dim_r = {self.dim_r}, dim_Λ = {self.dim_Λ}, dim_ρ = {self.dim_ρ}, dim_α = {self.dim_α}, dim_β = {self.dim_β}, ϵ = {ϵ}')
        # print(K)
        # print(E)
        # print(Q)
        # print(S)
        
        # Design variables
        self.optProb.addVarGroup('v',self.dim_v,value=self.xdict['v'])
        self.optProb.addVarGroup('w',self.dim_q,value=self.xdict['w'])
        self.optProb.addVarGroup('q',self.dim_q,value=self.xdict['q'])
        self.optProb.addVarGroup('r',self.dim_r,lower=0,value=self.xdict['r'])
        self.optProb.addVarGroup('δ',self.dim_r,lower=1e-9,value=self.xdict['δ'])
        self.optProb.addVarGroup('θ',self.dim_r,value=self.xdict['θ'])
        self.optProb.addVarGroup('Λ',self.dim_Λ,value=self.xdict['Λ'])
        self.optProb.addVarGroup('ρ',self.dim_ρ,lower=0,value=self.xdict['ρ'])
        self.optProb.addVarGroup('τ',self.dim_ρ,lower=0,value=self.xdict['τ'])
        self.optProb.addVarGroup('ϕ',self.dim_ρ,value=self.xdict['ϕ'])
        self.optProb.addVarGroup('φ',self.dim_ρ,value=self.xdict['φ'])
        self.optProb.addVarGroup('α',self.dim_α,lower=1e-8,value=self.xdict['α'])
        self.optProb.addVarGroup('β',self.dim_β,lower=1e-8,value=self.xdict['β'])
        
        # Constraints
        
        ## Linear Constraint 1
        jac_l1_v = sparse_to_coosparse(self.R.H @ self.R)
        jac_l1_q = sparse_to_coosparse(self.K.T)
        rhs = np.real(self.R.H * self.noisy_img.ravel()).ravel()
        self.optProb.addConGroup('lin_con_1',self.dim_v,lower=rhs,upper=rhs,wrt=['v','q'],jac={'v':jac_l1_v,'q':jac_l1_q},linear=True)
        
        ## Linear Constraint 2
        jac_l2_q = diag_to_coodiag(np.ones(self.dim_q))
        jac_l2_Λ = sparse_to_coosparse(self.E.T)
        self.optProb.addConGroup('lin_con_2',self.dim_q,lower=0,upper=0,wrt=['Λ','q'],jac={'Λ':jac_l2_Λ,'q':jac_l2_q},linear=True)
        
        ## Linear Constraint 3
        jac_l3_α = sparse_to_coosparse(self.Q)
        jac_l3_δ = diag_to_coodiag(-np.ones(self.dim_r))
        self.optProb.addConGroup('lin_con_3',self.dim_r,lower=0,wrt=['α','δ'],jac={'α':jac_l3_α,'δ':jac_l3_δ},linear=True)
        
        ## Linear Constraint 4
        jac_l4_β = sparse_to_coosparse(self.S)
        jac_l4_τ = diag_to_coodiag(-np.ones(self.dim_ρ))
        self.optProb.addConGroup('lin_con_4',self.dim_ρ,lower=0,wrt=['β','τ'],jac={'β':jac_l4_β,'τ':jac_l4_τ},linear=True)
        
        ## Nonlinear Constraint 1
        jac_nl1_v = sparse_to_coosparse(self.K)
        jac_nl1_w = diag_to_coodiag(np.ones(self.dim_q))
        jac_nl1_r = block_diag_to_coosparse(np.cos(self.xdict['θ']),np.sin(self.xdict['θ']))
        jac_nl1_θ = block_diag_to_coosparse(-self.xdict['r']*np.sin(self.xdict['θ']),self.xdict['r']*np.cos(self.xdict['θ']))
        self.optProb.addConGroup('nl_con_1',self.dim_q,lower=0,upper=0,wrt=['v','w','r','θ'],jac={'v':jac_nl1_v,'w':jac_nl1_w,'r':jac_nl1_r,'θ':jac_nl1_θ})
        
        ## Nonlinear Constraint 2
        jac_nl2_q = diag_to_coodiag(np.ones(self.dim_q))
        jac_nl2_δ = block_diag_to_coosparse(np.cos(self.xdict['θ']),np.sin(self.xdict['θ']))
        jac_nl2_θ = block_diag_to_coosparse(-self.xdict['δ']*np.sin(self.xdict['θ']),self.xdict['δ']*np.cos(self.xdict['θ']))
        self.optProb.addConGroup('nl_con_2',self.dim_q,lower=0,upper=0,wrt=['q','δ','θ'],jac={'q':jac_nl2_q,'δ':jac_nl2_δ,'θ':jac_nl2_θ})
        
        ## Nonlinear Constraint 3
        jac_nl3_w = sparse_to_coosparse(self.E)
        jac_nl3_ρ = triple_block_diag_to_coosparse(np.ones(self.dim_ρ),np.ones(self.dim_ρ),np.ones(self.dim_ρ))
        jac_nl3_ϕ = triple_block_diag_to_coosparse(np.ones(self.dim_ρ),np.ones(self.dim_ρ),np.ones(self.dim_ρ))
        jac_nl3_φ = triple_block_diag_to_coosparse(np.ones(self.dim_ρ),np.ones(self.dim_ρ),np.ones(self.dim_ρ))
        self.optProb.addConGroup('nl_con_3',self.dim_Λ,lower=0,upper=0,wrt=['w','ρ','ϕ','φ'],jac={'w':jac_nl3_w,'ρ':jac_nl3_ρ,'ϕ':jac_nl3_ϕ,'φ':jac_nl3_φ})
        
        ## Nonlinear Constraint 4
        jac_nl4_Λ = diag_to_coodiag(np.ones(self.dim_Λ))
        jac_nl4_τ = triple_block_diag_to_coosparse(np.sin(self.xdict['ϕ']) * np.cos(self.xdict['φ']),np.sin(self.xdict['ϕ']) * np.sin(self.xdict['φ']),np.cos(self.xdict['ϕ']))
        jac_nl4_ϕ = triple_block_diag_to_coosparse(self.xdict['τ']*np.cos(self.xdict['ϕ']) * np.cos(self.xdict['φ']),self.xdict['τ']*np.cos(self.xdict['ϕ']) * np.sin(self.xdict['φ']),-self.xdict['τ']*np.sin(self.xdict['ϕ']))
        jac_nl4_φ = triple_block_diag_to_coosparse(self.xdict['τ']*np.sin(self.xdict['ϕ']) * np.sin(self.xdict['φ']),-self.xdict['τ']*np.sin(self.xdict['ϕ']) * np.cos(self.xdict['φ']),np.ones(self.dim_ρ))
        self.optProb.addConGroup('nl_con_4',self.dim_Λ,lower=0,upper=0,wrt=['Λ','τ','ϕ','φ'],jac={'Λ':jac_nl4_Λ,'τ':jac_nl4_τ,'ϕ':jac_nl4_ϕ,'φ':jac_nl4_φ})
        
        ## Nonlinear Constraint 5
        jac_nl5_r = diag_to_coodiag(self.Q @ self.xdict['α'] - self.xdict['δ'])
        jac_nl5_α = sparse_to_coosparse(spdiags(self.xdict['r'],0,self.dim_r,self.dim_r)@self.Q)
        jac_nl5_δ = diag_to_coodiag(-self.xdict['r'])
        self.optProb.addConGroup('nl_con_5',self.dim_r,upper=ϵ,wrt=['r','α','δ'],jac={'r':jac_nl5_r,'α':jac_nl5_α,'δ':jac_nl5_δ})
        
        ## Nonlinear Constraint 6
        jac_nl6_ρ = diag_to_coodiag(self.S @ self.xdict['β'] - self.xdict['τ'])
        jac_nl6_β = sparse_to_coosparse(spdiags(self.xdict['ρ'],0,self.dim_ρ,self.dim_ρ)@self.S)
        jac_nl6_τ = diag_to_coodiag(-self.xdict['ρ'])
        self.optProb.addConGroup('nl_con_6',self.dim_ρ,upper=ϵ,wrt=['ρ','β','τ'],jac={'ρ':jac_nl6_ρ,'β':jac_nl6_β,'τ':jac_nl6_τ})
        
        self.optProb.addObj('obj')
        
    def objfun(self,xdict):
        funcs = {}
        funcs['obj'] = 0.5 * np.linalg.norm(xdict['v'] - self.true_img)**2
        
        # funcs['lin_con_1'] = self.K.T @ xdict['q'] + xdict['v'] - self.noisy_img
        # funcs['lin_con_2'] = self.E.T @ xdict['Λ'] + xdict['q']
        # funcs['lin_con_3'] = (self.Q @ xdict['α']) - xdict['δ']
        # funcs['lin_con_4'] = (self.S @ xdict['β']) - xdict['τ']
        
        funcs['nl_con_1'] = (self.K @ xdict['v']) - xdict['w'] - np.concatenate([xdict['r'] * np.cos(xdict['θ']),xdict['r'] * np.sin(xdict['θ'])])
        funcs['nl_con_2'] = xdict['q'] - np.concatenate([xdict['δ'] * np.cos(xdict['θ']),xdict['δ'] * np.sin(xdict['θ'])])
        funcs['nl_con_3'] = self.E @ xdict['w'] - np.concatenate([xdict['ρ'] * np.sin(xdict['ϕ']) * np.cos(xdict['φ']),xdict['ρ'] * np.sin(xdict['ϕ']) * np.sin(xdict['φ']),xdict['ρ'] * np.cos(xdict['ϕ'])])
        funcs['nl_con_4'] = xdict['Λ'] - np.concatenate([xdict['τ'] * np.sin(xdict['ϕ']) * np.cos(xdict['φ']),xdict['τ'] * np.sin(xdict['ϕ']) * np.sin(xdict['φ']),xdict['τ'] * np.cos(xdict['ϕ'])])
        
        funcs['nl_con_5'] = xdict['r'] * ((self.Q @ xdict['α'])-xdict['δ'])
        funcs['nl_con_6'] = xdict['ρ'] * ((self.S @ xdict['β'])-xdict['τ'])
        
        fail = False
        return funcs, fail
    
    def usr_jac(self,xdict,fdict):
        gout = {}
        
        gout['obj'] = {}
        gout['obj']['v'] = xdict['v'] - self.true_img.ravel()
        
        gout['nl_con_1'] = {}
        gout['nl_con_1']['v'] = sparse_to_coosparse(self.K)
        gout['nl_con_1']['w'] = diag_to_coodiag(-np.ones(self.dim_q))
        gout['nl_con_1']['r'] = block_diag_to_coosparse(-np.cos(xdict['θ']),-np.sin(xdict['θ']))
        gout['nl_con_1']['θ'] = block_diag_to_coosparse(xdict['r'] * np.sin(xdict['θ']),-xdict['r'] * np.cos(xdict['θ']))
        
        gout['nl_con_2'] = {}
        gout['nl_con_2']['q'] = diag_to_coodiag(np.ones(self.dim_q))
        gout['nl_con_2']['δ'] = block_diag_to_coosparse(-np.cos(xdict['θ']),-np.sin(xdict['θ']))
        g1 = xdict['δ']*np.sin(xdict['θ'])
        g1[g1==0] = 1e-9
        g2 = -xdict['δ'] * np.cos(xdict['θ'])
        g2[g2==0] = 1e-9
        gout['nl_con_2']['θ'] = block_diag_to_coosparse(g1,g2)
        # test = block_diag_to_coosparse(g1,g2)
        # if len(test["coo"][2]) == 240:
        #     print(f'δ = {xdict["δ"]}')
        #     print(f'sinθ = {np.sin(xdict["θ"])}')
        #     print(f'cosθ = {np.cos(xdict["θ"])}')
        
        gout['nl_con_3'] = {}
        gout['nl_con_3']['w'] = sparse_to_coosparse(self.E)
        gout['nl_con_3']['ρ'] = triple_block_diag_to_coosparse(-np.sin(xdict['ϕ']) * np.cos(xdict['φ']),-np.sin(xdict['ϕ']) * np.sin(xdict['φ']),-np.cos(xdict['ϕ']))
        gout['nl_con_3']['ϕ'] = triple_block_diag_to_coosparse(-xdict['ρ']*np.cos(xdict['ϕ']) * np.cos(xdict['φ']),-xdict['ρ']*np.cos(xdict['ϕ']) * np.sin(xdict['φ']),xdict['ρ']*np.sin(xdict['ϕ']))
        gout['nl_con_3']['φ'] = triple_block_diag_to_coosparse(xdict['ρ']*np.sin(xdict['ϕ']) * np.sin(xdict['φ']),-xdict['ρ']*np.sin(xdict['ϕ']) * np.cos(xdict['φ']),1e-9*np.ones(self.dim_ρ))
        
        gout['nl_con_4'] = {}
        gout['nl_con_4']['Λ'] = diag_to_coodiag(np.ones(self.dim_Λ))
        gout['nl_con_4']['τ'] = triple_block_diag_to_coosparse(-np.sin(xdict['ϕ']) * np.cos(xdict['φ']),-np.sin(xdict['ϕ']) * np.sin(xdict['φ']),-np.cos(xdict['ϕ']))
        gout['nl_con_4']['ϕ'] = triple_block_diag_to_coosparse(-xdict['τ']*np.cos(xdict['ϕ']) * np.cos(xdict['φ']),-xdict['τ']*np.cos(xdict['ϕ']) * np.sin(xdict['φ']),xdict['τ']*np.sin(xdict['ϕ']))
        gout['nl_con_4']['φ'] = triple_block_diag_to_coosparse(xdict['τ']*np.sin(xdict['ϕ']) * np.sin(xdict['φ']),-xdict['τ']*np.sin(xdict['ϕ']) * np.cos(xdict['φ']),1e-9*np.ones(self.dim_ρ))
        
        gout['nl_con_5'] = {}
        gout['nl_con_5']['r'] = diag_to_coodiag(self.Q @ xdict['α'] - xdict['δ'])
        gout['nl_con_5']['α'] = sparse_to_coosparse(spdiags(xdict['r'],0,self.dim_r,self.dim_r)@self.Q)
        gout['nl_con_5']['δ'] = diag_to_coodiag(-xdict['r'])
        
        gout['nl_con_6'] = {}
        gout['nl_con_6']['ρ'] = diag_to_coodiag(self.S @ xdict['β'] - xdict['τ'])
        gout['nl_con_6']['β'] = sparse_to_coosparse(spdiags(xdict['ρ'],0,self.dim_ρ,self.dim_ρ)@self.S)
        gout['nl_con_6']['τ'] = diag_to_coodiag(-xdict['ρ'])
        
        return gout,False
    
    def solve(self, print_level=0):
        if print_level > 0:
            self.optProb.printSparsity()
        opt = OPT('IPOPT',
                options={
                    'print_level': print_level,
                    'linear_solver': 'ma86',
                    'limited_memory_max_history':0,
                    'tol':self.ϵ,
                    'fast_step_computation':'yes',
                })
        # sol = opt(self.optProb, sens='FD')
        sol = opt(self.optProb,sens=self.usr_jac,sensMode='pgc')
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
        return sol,extra
    
    def __str__(self) -> str:
        print(self.optProb)
        return ''
    
    
def solve_mpcc(true_img,noisy_img,K,E,R,Q,S,α_size=1,β_size=1,print_level=0):
    print(f'Warm start with TV solution')
    sol_tv,extra_tv = solve_tv(true_img,noisy_img,K,R,Q,α_size,print_level)
    
    extras = []
    xdict = {
        'v': sol_tv.xStar['u'],
        'w': 1e-1*np.ones(K.shape[0]),
        'q': sol_tv.xStar['q'],
        'Λ': 1e-1*np.ones(E.shape[0]),
        'r': sol_tv.xStar['r'],
        'ρ': 1e-1*np.ones(E.shape[0]//3),
        'δ': sol_tv.xStar['δ'],
        'τ': 1e-7*np.ones(E.shape[0]//3),
        'θ': sol_tv.xStar['θ'],
        'ϕ': 1e-7*np.ones(E.shape[0]//3),
        'φ': 1e-7*np.ones(E.shape[0]//3),
        'α': sol_tv.xStar['α'],
        'β': 1e-7*np.ones(β_size)
    }
    ϵ = 0.1
    for i in range(3):
        tgv_2d_mpcc = TGV_2D_MPCC(true_img,noisy_img,K,E,R,Q,S,ϵ,xdict,α_size,β_size)
        sol,extra = tgv_2d_mpcc.solve(print_level=print_level)
        xdict = {
            'v': sol.xStar['v'],
            'w': sol.xStar['w'],
            'q': sol.xStar['q'],
            'Λ': sol.xStar['Λ'],
            'r': sol.xStar['r'],
            'ρ': sol.xStar['ρ'],
            'δ': sol.xStar['δ'],
            'τ': sol.xStar['τ'],
            'θ': sol.xStar['θ'],
            'ϕ': sol.xStar['ϕ'],
            'φ': sol.xStar['φ'],
            'α': sol.xStar['α'],
            'β': sol.xStar['β']
        }
        extras.append(extra)
        ϵ *= 0.1
    # print(sol)
    return sol,extras