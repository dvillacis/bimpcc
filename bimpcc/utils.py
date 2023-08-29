import numpy as np
import scipy.sparse as sp

def sparse_to_coosparse(sparse):
    coo = sparse.tocoo()
    return {'coo':[coo.row,coo.col,coo.data],'shape':[sparse.shape[0],sparse.shape[1]]}

def diag_to_coodiag(diag):
    return {'coo':[np.arange(len(diag)),np.arange(len(diag)),diag],'shape':[len(diag),len(diag)]}

def block_diag_to_coosparse(diag1,diag2):
    D1 = sp.spdiags(diag1,0,len(diag1),len(diag1))
    D2 = sp.spdiags(diag2,0,len(diag2),len(diag2))
    D = sp.vstack([D1,D2])
    return sparse_to_coosparse(D)

def triple_block_diag_to_coosparse(diag1,diag2,diag3):
    D1 = sp.spdiags(diag1,0,len(diag1),len(diag1))
    D2 = sp.spdiags(diag2,0,len(diag2),len(diag2))
    D3 = sp.spdiags(diag3,0,len(diag3),len(diag3))
    D = sp.vstack([D1,D2,D3])
    return sparse_to_coosparse(D)    