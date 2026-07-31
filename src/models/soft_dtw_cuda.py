# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import math
from numba import cuda, prange
import numpy as np

import torch
from torch.autograd import Function

from src.utils.utils_mlm import pytorch_cos_sim

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit(debug=False)
def compute_softdtw_cuda(D, gamma, bandwidth, path_w, max_i, max_j, n_passes, R):
    """
    :param gamma: softness of SDTW path
    :param bandwidth: bandwidth of the alignment path
    :param path_w: tensor, the path weight of SDTW, in shape (batch_size, 3), 
                each row corresponds to [w_d, w_h, w_v]
    :param max_i: tensor, the variable length of the firest sequence
    :param max_j: tensor, the variable length of the second sequence
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    w_d = path_w[b, 0]
    w_h = path_w[b, 1]
    w_v = path_w[b, 2]
    
    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes[b]):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j[b] - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i[b] and J < max_j[b]):
            # Not compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -inv_gamma * (R[b, i - 1, j - 1] + D[b, i - 1, j - 1] * w_d)
                r1 = -inv_gamma * (R[b, i - 1, j] + D[b, i - 1, j - 1] * w_h)
                r2 = -inv_gamma * (R[b, i, j - 1] + D[b, i - 1, j - 1] * w_v)

                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit(debug=False)
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, path_w, max_i, max_j, n_passes, E, H):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    ##################### needed for variable length ########################
    R[k, :, max_j[k]+1] = -math.inf
    R[k, max_i[k]+1, :] = -math.inf
    R[k, max_i[k]+1, max_j[k]+1] = R[k, max_i[k], max_j[k]]
    E[k, max_i[k]+1, max_j[k]+1] = 1
    #########################################################################

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    w_d = path_w[k, 0]
    w_h = path_w[k, 1]
    w_v = path_w[k, 2]

    for p in range(n_passes[k]):
        # Reverse the order to make the loop go backward
        rev_p = n_passes[k] - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j[k] - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i[k] and J < max_j[k]):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Not compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j] * w_h) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1] * w_v) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1] * w_d) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

                G = w_h * math.exp(-(R[k, i - 1, j] - R[k, i, j] + w_h * D[k, i, j]) * inv_gamma) + \
                    w_v * math.exp(-(R[k, i, j - 1] - R[k, i, j] + w_v * D[k, i, j]) * inv_gamma) + \
                    w_d * math.exp(-(R[k, i - 1, j - 1] - R[k, i, j] + w_d * D[k, i, j]) * inv_gamma)

                H[k, i, j] = E[k, i, j] * G

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """
    e_matrix = None # store expected alignment for analysis purposes
    h_matrix=None
    R_accumulated = None # store accumulated cost matrix for analysis purposes

    @staticmethod
    def forward(ctx, D, gamma, bandwidth, path_w, D_mask):
        dev = D.device
        dtype = D.dtype
        bandwidth = torch.tensor(bandwidth, dtype=float, device=dev)

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]

        N_ = torch.stack([D_mask[k].nonzero()[-1][0]+1 for k in range(B)])
        M_ = torch.stack([D_mask[k].nonzero()[-1][1]+1 for k in range(B)])

        threads_per_block = int(max(max(N_), max(M_)))
        n_passes = torch.stack([N_[k]+M_[k]-1 for k in range(B)])

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel. 
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), 
                                                   cuda.as_cuda_array(path_w), N_, M_, n_passes,
                                                   cuda.as_cuda_array(R))
        
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth, path_w, D_mask)
        # 2D mask flattens the matrix; the last element is exactly the right-bottom element we want
        cost_out = torch.stack([R[k][i,j] for k, (i,j) in enumerate(zip(N_, M_))])

        _SoftDTWCUDA.R_accumulate = R

        return cost_out

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth, path_w, D_mask = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        N_ = torch.stack([D_mask[k].nonzero()[-1][0]+1 for k in range(B)])
        M_ = torch.stack([D_mask[k].nonzero()[-1][1]+1 for k in range(B)])

        threads_per_block = int(max(max(N_), max(M_)))
        n_passes = torch.stack([N_[k]+M_[k]-1 for k in range(B)])

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        H = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), 
                                                            cuda.as_cuda_array(path_w), N_, M_, n_passes,
                                                            cuda.as_cuda_array(E),
                                                            cuda.as_cuda_array(H))

        E = E[:, 1:N + 1, 1:M + 1]
        _SoftDTWCUDA.e_matrix = E

        H = H[:, 1:N + 1, 1:M + 1]
        _SoftDTWCUDA.h_matrix = H
        return grad_output.view(-1, 1, 1).expand_as(H) * H, None, None, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
# @jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth, D_mask):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]

    N_ = torch.stack([sum(~D_mask[k][:,0]) for k in range(B)])
    M_ = torch.stack([sum(~D_mask[k][0,:]) for k in range(B)])

    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M_[b] + 1):
            for i in range(1, N_[b] + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

# ----------------------------------------------------------------------------------------------------------------------
# @jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth, D_mask):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    N_ = torch.stack([sum(~D_mask[k][:,0]) for k in range(B)])
    M_ = torch.stack([sum(~D_mask[k][0,:]) for k in range(B)])
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_

    for k in prange(B):
        R[k, :, M_[k]+1] = -np.inf
        R[k, N_[k]+1, :] = -np.inf
        R[k, N_[k]+1, M_[k]+1] = R[k, N_[k], M_[k]]
        E[k, N_[k]+1, M_[k]+1] = 1
        for j in range(M_[k], 0, -1):
            for i in range(N_[k], 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """
    e_matrix = None # store expected alignment for analysis purposes
    h_matrix = None # store expected alignment for analysis purposes
    R_accumulated = None # store accumulated cost matrix for analysis purposes

    @staticmethod
    def forward(ctx, D, gamma, bandwidth, D_mask):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = compute_softdtw(D_, g_, b_, D_mask)
        R = torch.Tensor(R).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth, D_mask)

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        D_mask_ = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype).bool()
        D_mask_[:, :N, :M] = ~D_mask
        cost_out = torch.stack([R[k][D_mask_[k]][-1] for k in range(R.size(0))])
        return cost_out

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth, D_mask = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_, D_mask)).to(dev).type(dtype)
        _SoftDTW.e_matrix = E
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=None, normalize=False, bandwidth=None, dist_func=None, sdtw_step_weight="equal", regularization_weight=0.0):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use: "cosine_sim" or "euclidean". 
                          If 'None', then a default Euclidean distance function will be used.
        :param sdtw_step_weight: "equal" or "lenInform". Use "equal" as default, i.e. [w_d, w_h, w_v] = [1, 1, 1]). "lenInform" customize 
                          step weights according to sequence length.
        :param regularization_weight: regularize SDTW cost according to sequence length difference.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda
        self.dtw_class = None

        # Set the distance function
        if dist_func == 'cosine_sim':
            self.dist_func = SoftDTW._cosinesim_dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

        self.sdtw_step_weight = sdtw_step_weight
        self.regularization_weight = regularization_weight

        # Register annealing as a buffer to save/load it
        self.register_buffer("gamma", torch.tensor(gamma))

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        self.dtw_class = _SoftDTWCUDA if use_cuda else _SoftDTW
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)
    
    @staticmethod
    def _cosinesim_dist_func(x, y):
        """
        Calculates the cosine similarity distance between each element in x and y per timestep
        """
        
        nbatch = x.size(0)
        batch_xlen = x.size(1)
        batch_ylen = y.size(1)

        cose_mat = torch.zeros(nbatch, batch_xlen, batch_ylen, device=x.device)

        for k in range(nbatch):
            cose_mat[k] = pytorch_cos_sim(x[k], y[k])

        return 1 - cose_mat
    

    def forward(self, X, Y, D_mask):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)

            D_xy = torch.masked_fill(D_xy, ~D_mask, math.inf).to(X.device)

            len_x = torch.stack([D_mask[k].nonzero()[-1][0]+1 for k in range(D_mask.size(0))])
            len_y = torch.stack([D_mask[k].nonzero()[-1][1]+1 for k in range(D_mask.size(0))])

            if self.sdtw_step_weight == "lenInform":
                ################# Sequence length-informed step weights ###################
                w_norm = len_x + len_y

                # Diagonal, horizontal, and vertical path weights: [w_d, w_h, w_v]
                w_h = len_y / w_norm
                w_v = len_x / w_norm
                len_norm = (len_x - 1) * w_h + (len_y - 1) * w_v + 1

                path_w = torch.cat([torch.ones_like(w_h).unsqueeze(1), w_h.unsqueeze(1), w_v.unsqueeze(1)], dim=1).to(X.device)

                loss = func_dtw(D_xy, self.gamma, self.bandwidth, path_w, D_mask) / len_norm
            else: # "euqal"
                ##################### Equal step weights (default) #########################
                path_w = torch.ones(D_mask.size(0), 3, device=X.device)
                len_norm = len_x

                loss = func_dtw(D_xy, self.gamma, self.bandwidth, path_w, D_mask) / len_norm

            # Normalized length difference
            len_diff = abs(len_x - len_y) / (torch.max(abs(len_x - len_y)) + 1e-6)

            # Sequence length difference informed regularization to the contrastive alignment loss
            return (1 - self.regularization_weight) * loss + self.regularization_weight * len_diff * (loss.max() - loss.min())

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    print("  Speedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    torch.manual_seed(1234)

    profile(128, 17, 15, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    profile(512, 256, 256, 2, tol_backward=1e-3)