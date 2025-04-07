import numpy as np
import scipy as sp
from paradigm_setting import paradigm_setting
import torch
np.set_printoptions(threshold=np.inf)

def simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N):
    """

    Parameters
    ----------
    v : integer
        The total number of voxels in the simulation
    X : float
        The length of the stimulus dimension (e.g angle range of grating)
    j : list
        A list of the condition orders (e.g [cond1, cond2, cond2, cond1])
    cond1 : float
        Stimulus value for condition 1 (1/4pi)
    cond2 : float
        Stimulus value for condition 2 (3/4pi)
    a : float
        The amount of adaptation
    b : float
        The extent of the domain adaptation
    sigma : float
        The width of the tuning curve
    model_type : integer
        The type of model demonstrated, represented by an integer from 1 to 12
    reset_after : integer
        The number of conditions to go through before needing to reset the adaptation factor c
    paradigm : string
        The paradigm being demonstrated: either 'face' or 'grating'
    N : integer
        The number of neuronal populations per voxel

    
    Returns
    -------
    out : dictionary
        A dictionary representing every run of the paradigm and the  """
    nt = len(j)
    res = 180
    dt = X/res
    x = torch.arange(dt, X + dt, dt, dtype=torch.float32, requires_grad=True)

    tuning_curves_peaks = torch.tensor([0, X/8, X*2/8, X*3/8, X*4/8, X*5/8, X*6/8, X*7/8], dtype=torch.float32, requires_grad=True)
    tuning_curves_peaks_np = [0, X/8, X*2/8, X*3/8, X*4/8, X*5/8, X*6/8, X*7/8]
    precomputed_gaussians = torch.stack([gaussian(x, u, sigma, paradigm) for u in tuning_curves_peaks])
    # print(precomputed_gaussians.shape)
    # print(torch.max(precomputed_gaussians, axis=1, keepdims=True).shape)
    precomputed_gaussians = precomputed_gaussians / torch.max(precomputed_gaussians, dim=1, keepdims=True)[0]

    pattern = torch.zeros((nt, v), dtype=torch.float32, requires_grad=True)
    activity = torch.zeros((nt, v, N), dtype=torch.float32, requires_grad=True) 
    rep = torch.zeros((nt, v, N, res), dtype=torch.float32, requires_grad=True) 

    # u = [3*X/8 X/8 3*X/8 X/8 X/8 X/8 5*X/8 X/8];
    #Randomly assign preferred tuning curves to neurons
    
    # u_vals = torch.tensor(np.random.choice(tuning_curves_peaks_np, size=(v,N), replace=True), dtype=torch.float32, requires_grad=True) #200x8
    # print(u_vals)
    # u_vals = torch.tensor(
    #     torch.multinomial(torch.ones(len(tuning_curves_peaks)), num_samples=(v, N), replacement=True),
    #     dtype=torch.float32,
    #     requires_grad=True
    # )
    indices = torch.randint(0, len(tuning_curves_peaks), (v,N))
    u_vals = tuning_curves_peaks[indices]
    u_vals.requires_grad_()
    # u = np.array([3*X/8, X/8, 3*X/8, X/8, X/8, X/8, 5*X/8, X/8])
    # u = np.tile(u, (v, 1))
    # u_vals = torch.tensor(u, dtype=torch.float32, requires_grad=True)
    u_indices = torch.searchsorted(tuning_curves_peaks, u_vals) #This maps the randomly selected values back to their positions in the original array
    init = precomputed_gaussians[u_indices] #init is 1 x 8 x 20

    #Create reset mask (as before, this is True every interval of reset after)
    # reset_mask = torch.mod(torch.arange(nt), reset_after) == 0
    c = torch.ones((nt, v, N), dtype=torch.float32, requires_grad=True) #Adaptation factor for every trial, voxel, and neuron
    #Compute adaptation for all trials at once using broadcasting
    # d = np.abs(u_vals[None, :, :] - np.array(j)[:, None, None])
    d = u_vals[None, :, :] - torch.tensor(j, dtype=torch.float32, requires_grad=True)[:, None, None]
    #u_vals[None, :, :] is 1 x 200 x 8
    #j is 48 so 48 x 1 x 1 meaning these can broadcast when subtracting!

    if paradigm == 'grating':
        d = torch.minimum(torch.abs(d), X-torch.abs(d))

    scaling_factors = {
        2: torch.minimum(torch.ones_like(d), (a + torch.abs(d / b) * (1 - a))),  # Local Scaling
        3: torch.maximum(a*torch.ones_like(d), (1 - torch.abs(d / b) * (1 - a))),  # Remote Scaling
        5: torch.minimum(torch.ones_like(d), (a + torch.abs(d / b) * (1 - a))),  # Local Sharpening
        6: torch.maximum(a*torch.ones_like(d), (1 - torch.abs(d / b) * (1 - a))),  # Remote Sharpening
        7: a * torch.sign(d),
        8: torch.sign(d) *torch.minimum(torch.ones_like(d), (a + torch.abs(d / b) * (1 - a))),  # Local Repulsion
        9: torch.sign(d) * torch.maximum(a*torch.ones_like(d), (1 - torch.abs(d / b) * (1 - a))),  # Remote Repulsion
        10: a * torch.sign(d),
        11: torch.sign(d) * torch.minimum(torch.ones_like(d), (a + torch.abs(d / b) * (1 - a))),  # Local Attraction
        12: torch.sign(d) * torch.maximum(a*torch.ones_like(d), (1 - torch.abs(d / b) * (1 - a))),  # Remote Attraction
    }

    shifting_models = {7, 8, 9, 10, 11, 12}  # Models that involve shifting
    if model_type in {1, 4}:
        e = a * torch.ones((nt, v, N), dtype=torch.float32, requires_grad=True)
    elif model_type in scaling_factors:
        e = scaling_factors[model_type]

    num_blocks = nt // reset_after
    e_reshaped = e.reshape(num_blocks, reset_after, v, N)
    e_modified = torch.ones_like(e_reshaped)
    e_modified[:, 1:, :, :] = e_reshaped[:, 1:, :, :]
    transformed_array = torch.cumprod(e_modified, dim=1)
    transformed_array = transformed_array.reshape(nt, v, N)


    if model_type in {4, 5, 6}:
        temp = gaussian(x[None, None, None, :], u_vals[None, :, :, None], transformed_array[..., None] * sigma, paradigm)
        #x becomes 1 x 1 x 1 x 180
        #u_vals becomes 1 x 200 x 8 x 1
        #c becomes 18 x 200 x 8 x 1
        temp = temp/ torch.max(temp, dim=-1, keepdims=True, dtype=torch.float32, requires_grad=True)
    elif model_type in shifting_models:  # Shift-based models
        shift_direction = 1 if model_type in {7, 8, 9} else -1  # Repulsive (+) vs. Attractive (-)
        shift_amount = transformed_array * X/2
        shift_amount[::reset_after, :, :] = 1
        temp = gaussian(x[None, None, None, :], u_vals[None, :, :, None] + shift_direction * shift_amount[..., None], sigma, paradigm)
        temp =temp / torch.max(temp, dim=-1, keepdims=True, dtype=torch.float32, requires_grad=True)
    elif model_type in {1, 2, 3}:  # Scaling models (1, 2, 3)
        temp = transformed_array[..., None] * init[None, :, :, :]
    rep = temp
    rep[::reset_after, :, :, :] = init
    print("rep.grad_fn")
    print(rep.grad_fn)
    # print(rep)


    cond_indices = (int(cond1 / dt - 1), int(cond2 / dt - 1))
    # print(rep.shape)
        # activity = np.take(rep, cond_indices[0], axis=-1) * (np.array(j)[:, None, None] == cond1) + \
        #        np.take(rep, cond_indices[1], axis=-1) * (np.array(j)[:, None, None] == cond2)

    activity = rep[..., cond_indices[0]] * (torch.tensor(j, dtype=torch.float32, requires_grad=True)[:, None, None] == cond1) + \
               rep[..., cond_indices[1]] * (torch.tensor(j, dtype=torch.float32, requires_grad=True)[:, None, None] == cond2)

    pattern = torch.mean(activity, dim=2)

    return pattern
    # return {
    #     'pattern': pattern,
    #     'rep': rep,
    #     'activity': activity
    # }

def gaussian(x, u, sigma, paradigm):
    if paradigm == 'face':
        return non_circular_g(x, sigma, u)
    elif paradigm == 'grating':
        return circular_g(2*x, 2*u, 1/sigma)
    
def non_circular_g(x, sigma, u):
    return torch.exp(-((x-u)**2)/(2*sigma*sigma))

def circular_g(x, u, sigma):
    # x = np.atleast_1d(x)
    c = 1 / (2*torch.pi*sp.special.i0(sigma))
    return c * torch.exp(sigma * torch.cos(x-u))



# v, X = 1, np.pi
# cond1, cond2 = X/4, 3*X/4
# sub_num = 18
# noise = 0.03
# N = 8
# paradigm = 'grating'
# a = 0.8
# b = 0.8
# sigma = 0.8
# model_type = 3
# j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)

# p = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N)['pattern']

# print(p)
    