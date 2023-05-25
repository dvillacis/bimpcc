import numpy as np
import scipy.sparse as sp
from pylops import LinearOperator
from pylops.basicoperators import VStack
from pylops.utils.backend import get_array_module
# from pylops.utils._internal import _value_or_sized_to_tuple
# from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

class FirstDerivative(LinearOperator):
    '''
    Finite Differences discretization of the FirstDerivative Operator.
    This Linear Operator excludes the information on the boundaries.
    '''
    def __init__(
        self, 
        N, 
        dims=None,
        dimsd=None, 
        dir=0, 
        sampling=1.0, 
        dtype=np.float64, 
        kind='forward'
    ):
        self.N = N
        self.sampling = sampling
        if dims is None:
            self.dims = (self.N,)
            self.reshape = False
            self.shape = (self.N-1,self.N)
        else:
            if np.prod(dims) != self.N:
                raise ValueError("product of dims must equal N")
            else:
                self.dims = dims
                self.reshape = True
                self.shape = ((self.dims[0]-1)*(self.dims[1]-1),self.N)
        self.dir = dir if dir >= 0 else len(self.dims) + dir
        self.kind = kind
        self.dtype = np.dtype(dtype)
        self.matvec_count = 0
        self.rmatvec_count = 0
        self.rmatmat_count = 0
        
        if self.kind == 'forward':
            self._matvec = self._matvec_forward
            self._rmatvec = self._rmatvec_forward
        elif self.kind == 'backward':
            self._matvec = self._matvec_backward
            self._rmatvec = self._rmatvec_backward
        else:
            raise NotImplementedError("kind must be forward, " "or backward")

    def _matvec_forward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N-1, self.dtype)
            y = (x[1:]-x[:-1]) / self.sampling
        else:
            x = ncp.reshape(x, self.dims)
            if self.dir > 0:
                # x = ncp.swapaxes(x,self.dir,0)
                y = (x[1:,:-1] - x[:-1,:-1]) / self.sampling
            else:
                # y = ncp.zeros(x.shape,self.dtype)
                y = (x[:-1,1:] - x[:-1,:-1]) / self.sampling
            # if self.dir > 0:
            #     y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
    
    def _rmatvec_forward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[:-1] -= x / self.sampling
            y[1:] += x / self.sampling
        else:
            x = ncp.reshape(x, (self.dims[0]-1,self.dims[1]-1))
            #y = ncp.zeros(self.dims, self.dtype)
            # print(y)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                #x = ncp.swapaxes(x, self.dir, 0)
                y1 = np.pad(x,[(0,1),(0,1)],mode='constant', constant_values=0)
                y2 = np.pad(x,[(1,0),(0,1)],mode='constant', constant_values=0)
                y = y2-y1
            else:
                y2 = np.pad(x,[(0,1),(1,0)],mode='constant', constant_values=0)
                y1 = np.pad(x,[(0,1),(0,1)],mode='constant', constant_values=0)
                y = y2-y1
            if self.dir == 0:
                y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
    
    def _matvec_backward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[1:] = (x[1:] - x[:-1]) / self.sampling
        else:
            x = ncp.reshape(x, (self.dims[0]-1,self.dims[1]-1))
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = ncp.swapaxes(x, self.dir, 0)
            y = ncp.zeros(x.shape, self.dtype)
            y[1:] = (x[1:] - x[:-1]) / self.sampling
            if self.dir > 0:
                y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y

    def _rmatvec_backward(self, x):
        ncp = get_array_module(x)
        if not self.reshape:
            x = x.squeeze()
            y = ncp.zeros(self.N, self.dtype)
            y[:-1] -= x[1:] / self.sampling
            y[1:] += x[1:] / self.sampling
        else:
            x = ncp.reshape(x, self.dims)
            if self.dir > 0:  # need to bring the dim. to derive to first dim.
                x = ncp.swapaxes(x, self.dir, 0)
            y = ncp.zeros(x.shape, self.dtype)
            y[:-1] -= x[1:] / self.sampling
            y[1:] += x[1:] / self.sampling
            if self.dir > 0:
                y = ncp.swapaxes(y, 0, self.dir)
            y = y.ravel()
        return y
    
class Gradient(LinearOperator):
    r"""Gradient.

    Apply gradient operator to a multi-dimensional array.

    .. note:: At least 2 dimensions are required, use
      :py:func:`pylops.FirstDerivative` for 1d arrays.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    sampling : :obj:`tuple`, optional
        Sampling steps for each direction.
    edge : :obj:`bool`, optional
        Use reduced order derivative at edges (``True``) or
        ignore them (``False``).
    kind : :obj:`str`, optional
        Derivative kind (``forward``, ``centered``, or ``backward``).
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Notes
    -----
    The Gradient operator applies a first-order derivative to each dimension of
    a multi-dimensional array in forward mode.

    For simplicity, given a three dimensional array, the Gradient in forward
    mode using a centered stencil can be expressed as:

    .. math::
        \mathbf{g}_{i, j, k} =
            (f_{i+1, j, k} - f_{i-1, j, k}) / d_1 \mathbf{i_1} +
            (f_{i, j+1, k} - f_{i, j-1, k}) / d_2 \mathbf{i_2} +
            (f_{i, j, k+1} - f_{i, j, k-1}) / d_3 \mathbf{i_3}

    which is discretized as follows:

    .. math::
        \mathbf{g}  =
        \begin{bmatrix}
           \mathbf{df_1} \\
           \mathbf{df_2} \\
           \mathbf{df_3}
        \end{bmatrix}

    In adjoint mode, the adjoints of the first derivatives along different
    axes are instead summed together.

    """

    def __init__(self,
                 dims: tuple,
                 sampling: int = 1,
                 edge: bool = False,
                 kind: str = "forward",
                 dtype="float64", name: str = 'G'):
        ndims = len(dims)
        # sampling = _value_or_sized_to_tuple(sampling, repeat=ndims)
        self.sampling = (sampling,sampling)
        self.edge = edge
        self.kind = kind
        self.shape = (np.prod(dims[0]-1)*(dims[1]-1),np.prod(dims))
        
        Op = VStack([FirstDerivative(
            N=np.prod(dims),
            dims=dims,
            dir=iax,
            # sampling=self.sampling[iax],
            # edge=edge,
            kind=kind,
            dtype=dtype,
        ) for iax in range(ndims)
        ])
        super().__init__(Op=Op)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return super()._matvec(x)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return super()._rmatvec(x)
    
class DirectionalGradient_Fixed(LinearOperator):
    def __init__(self,
                 dims: tuple,
                 beta: float,
                 theta: float,
                 dir: int = 0,
                 sampling: int = 1,
                 edge: bool = False,
                 kind: str = "forward",
                 dtype = "float64", 
                 name: str = 'dG'):
        
        self.beta = beta
        self.theta = theta
        self.dir = dir
        self.sampling = sampling
        self.edge = edge
        self.kind = kind
        self.dtype = dtype
        self.name = name
        
        self.dims = dims
        self.ndim = len(self.dims)
        self.shape = ((self.dims[0]-1)*(self.dims[1]-1), np.prod(self.dims))
        
        self.gradOp = Gradient(self.dims, self.sampling, self.edge, self.kind, self.dtype)
                
        super().__init__()
        
    def _matvec(self, x: np.ndarray):
        gradx = self.gradOp.matvec(x.ravel())
        gradx = gradx.reshape((self.ndim, ) + (self.dims[0]-1,self.dims[1]-1))
        if self.dir==0:
            grad = self.beta * np.cos(self.theta) * gradx[0,:,:] - self.beta * np.sin(self.theta) * gradx[1,:,:]
        else:
            grad = np.sin(self.theta) * gradx[0,:,:] + np.cos(self.theta) * gradx[1,:,:]
        return grad.ravel()
    
    def _rmatvec(self, x):
        return self._matvec(x)
    
class DirectionalGradient(LinearOperator):
    def __init__(self, sinfo: np.ndarray,
                 eta: float,
                 gamma: float = 0.99,
                 dir: int = 0,
                 sampling: int = 1,
                 edge: bool = False,
                 kind: str = "forward",
                 dtype = "float64", 
                 name: str = 'dG'):
        
        self.sinfo = sinfo
        self.eta = eta
        self.gamma = gamma
        self.dir = dir
        self.sampling = sampling
        self.edge = edge
        self.kind = kind
        self.dtype = dtype
        self.name = name
        
        self.dims = sinfo.shape
        self.ndim = len(self.dims)
        self.shape = ((self.dims[0]-1)*(self.dims[1]-1), np.prod(self.dims))
        
        self.gradOp = Gradient(self.dims, self.sampling, self.edge, self.kind, self.dtype)
        
        self.gradsinfo = self.gradOp.matvec(sinfo.ravel())
        self.gradsinfo = self.gradsinfo.reshape((self.ndim, ) + (self.dims[0]-1,self.dims[1]-1))
        gradsinfo_norm = 0
        for gv in self.gradsinfo:
            gradsinfo_norm += np.power(abs(gv), 2)
        den = self.eta**2 + gradsinfo_norm
        self.gamma_ = -self.gamma / den
                
        super().__init__()
        
    def _matvec(self, x: np.ndarray):
        gradx = self.gradOp.matvec(x.ravel())
        gradx = gradx.reshape((self.ndim, ) + (self.dims[0]-1,self.dims[1]-1))
        aux = self.gamma_ * np.sum(self.gradsinfo * gradx, axis=0)
        grad = gradx - aux * self.gradsinfo
        return grad[self.dir,:,:].ravel()
    
    def _rmatvec(self, x):
        return self._matvec(x)
    

class BinarySelection(LinearOperator):
    def __init__(self,dims: tuple, m: int, dtype = "float64", name: str = 'BinSel'):
        self.m = m
        self.dtype = dtype
        self.name = name
        
        self.dims = dims
        self.shape = (m, np.prod(dims))
        
        # Binary Selection
        self.selection = sp.eye(np.prod(dims)).tocsr()
        rows_to_keep = np.sort(np.random.choice(np.prod(dims), m, replace=False))
        self.selection = self.selection[rows_to_keep, :]
        # print(f'rows_to_keep: {rows_to_keep}')
        # self.selection = np.sort(np.random.choice(np.prod(dims), m, replace=False))
        
        super().__init__()
        
    def _matvec(self, x: np.ndarray):
        return self.selection.dot(x.ravel())
    
    def _rmatvec(self, x):
        return self.selection.T.dot(x)
    
class PatchSelection(LinearOperator):
    def __init__(self,dims: tuple, patch_shape: tuple, dtype = "float64", name: str = 'PatchSel'):
        self.patch_shape = patch_shape
        print(f'patch_shape: {patch_shape}')
        self.dtype = dtype
        self.name = name
        
        self.dims = dims
        print(f'dims: {dims}')
        
        
        # Binary Selection
        self.selection = sp.eye(np.prod(dims)).tocsr()
        
        np.random.seed(0)
        dummy_img = np.ones(dims)
        row,col = np.random.randint(0, dims[0]-patch_shape[0]), np.random.randint(0, dims[1]-patch_shape[1])
        row2,col2 = np.random.randint(0, dims[0]-patch_shape[0]), np.random.randint(0, dims[1]-patch_shape[1])
        dummy_img[row:row+patch_shape[0],col:col+patch_shape[1]] = 0
        dummy_img[row2:row2+patch_shape[0],col2:col2+patch_shape[1]] = 0
        rows_to_keep = np.nonzero(dummy_img.ravel())[0]
        # rows_to_keep = np.sort(np.random.choice(np.prod(dims), nz_idx, replace=False))
        self.selection = self.selection[rows_to_keep, :]
        print(f'selection: {self.selection.shape}')
        # print(f'rows_to_keep: {rows_to_keep}')
        # self.selection = np.sort(np.random.choice(np.prod(dims), m, replace=False))
        
        self.shape = (len(rows_to_keep), np.prod(dims))
        print(f'shape: {self.shape}')
        
        super().__init__()
        
    def _matvec(self, x: np.ndarray):
        return self.selection.dot(x.ravel())
    
    def _rmatvec(self, x):
        return self.selection.T.dot(x)
    
class PatchOperator(LinearOperator):
    def __init__(self, dims: tuple, patch_size: tuple, dtype = "float64", name: str = 'Patch'):
        self.patch_size = patch_size
        self.dtype = dtype
        self.name = name
        
        self.dims = dims
        self.shape = (np.prod(dims), np.prod(patch_size))
        self.aux = np.ones((int(np.ceil(dims[0]/patch_size[0])), int(np.ceil(dims[1]/patch_size[1]))))
        
        super().__init__()
        
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(self.patch_size)
        m = np.kron(x, self.aux)
        return m[:self.dims[0],:self.dims[1]].ravel()
    

class DiagonalPatchOperator(LinearOperator):
    def __init__(self, dims: tuple, patch_size: tuple, dtype = "float64", name: str = 'Patch'):
        self.patch_size = patch_size
        self.dtype = dtype
        self.name = name
        
        self.dims = dims
        self.shape = (np.prod(dims), np.prod(patch_size))
        
        super().__init__()
        
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        data = x[0]*np.ones(self.dims[0])
        for el in x:
            for i in range(int(np.ceil(self.dims[0]/self.patch_size[0]))):
                # print(el,i,data)
                data = np.vstack((data,el*np.ones(self.dims[0])))
        n = data.shape[0]
        diags = np.arange(-n//2,n//2)
        # print(data,diags)
        m = sp.spdiags(data,diags,self.dims[0],self.dims[1]).todense()
        # print(m.shape,m)
        return np.ravel(m)