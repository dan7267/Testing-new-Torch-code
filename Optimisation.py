import torch
import torch.optim as optim
# import numpy as np
from paradigm_setting import paradigm_setting
from simulate_adaptation import simulate_adaptation
from joblib import Parallel, delayed
from repeffects_fig4_sims_alt import produce_confidence_interval
from ExperimentalData import create_pattern
from graphviz import Digraph
import torchviz
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

faceData = 'face_data.mat' #We need to change the data
gratingData = 'grating_data.mat' #We need to change the data

def simulate_subject(sub, v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, ind, gaussian_noise, tuning_curves_indices):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""

    out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, tuning_curves_indices)
    # /pattern = (out.T + torch.randn(v, len(j), requires_grad=True) * noise).T
    pattern = out + gaussian_noise
    v = pattern.shape[1]
    if paradigm == 'face':
        return torch.vstack((pattern[::4, :v], pattern[1::4, :v], pattern[2::4, :v], pattern[3::4, :v]))
    elif paradigm == 'grating':
        return torch.vstack((pattern[ind[0], :v], pattern[ind[2], :v], pattern[ind[3], :v], pattern[ind[5], :v]))

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, k, n_jobs, n_simulations, v, gaussian_noise_near, tuning_curves_indices_near, sub_num, N):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    X = torch.pi
    cond1, cond2 = X/4, 3*X/4
    y = torch.zeros(sub_num, dtype=torch.float32)

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    results_list = []

    for sub in range(sub_num):
        gaussian_noise = gaussian_noise_near[sub]
        tuning_curves_indices = tuning_curves_indices_near[sub]
        results_list.append(simulate_subject(sub, v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, ind, gaussian_noise, tuning_curves_indices))
    results = torch.stack(results_list)
    return produce_confidence_interval(results, 1)

def produce_slopes_multiple_simulations(sigma, a, b, k, model_type, paradigm, n_jobs, n_simulations, v, gaussian_noise_all, tuning_curves_indices_all, sub_num, N):
    #Is this function necessary. Yes, currently just one simulation
    """Simulate data using given parameters, with specific random seed
    so that each run has different random variations but is reproducible for the
    same parameter set. Note this can also be done by generating the random
    array first and then adding this appropriately each time."""
    # torch.manual_seed(seed)
    # simulated = ...
    X = torch.pi
    cond1, cond2 = X/4, 3*X/4
    simulation = True
    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    results = torch.zeros((n_simulations, 6))
    for sim in range(n_simulations):
        gaussian_noise_near=gaussian_noise_all[sim]
        tuning_curves_indices_near = tuning_curves_indices_all[sim]
        results[sim] = produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, k, n_jobs, n_simulations, v, gaussian_noise_near, tuning_curves_indices_near, sub_num, N)
        print("results[sim]")
        print(results[sim])
    print("results")
    print(results)
    return results

def process_empirical_subject(sub, paradigm):
    print("processing empirical subject")
    full_pattern = create_pattern(paradigm)
    pattern = torch.tensor(full_pattern[sub])
    v = pattern.shape[1]
    
    cond1_p = {
        1: pattern[::4, :v],
        2: pattern[1::4, :v]
    }
    cond2_p = {
        1: pattern[2::4, :v],
        2: pattern[3::4, :v]
    }
    
    return torch.vstack([cond1_p[1], cond1_p[2], cond2_p[1], cond2_p[2]])

def produce_slopes_empirical(paradigm, sub_num):
    results = torch.stack(
        [process_empirical_subject(sub, paradigm)
        for sub in range(sub_num)
        ])
    return produce_confidence_interval(results, 1)

def empirical_data(paradigm):
    sub_num = 18
    results = produce_slopes_empirical(paradigm, sub_num)
    return results

# def objective_function(simulated_data, empiricaled_data, weights):
#     objective=torch.sum(weights * torch.abs(simulated_data - empiricaled_data))
#     return objective

def objective_function(simulated_data, empirical_data, weights):
    """Now set up to run over many simulations. Instead of simulated_data being the 6 slopes in a 1D tensor,
    simulated_data will be an n_simulations x 6 tensor which is then averaged at the end"""
    n_simulations = simulated_data.shape[0]
    objective = 0
    for i in range(n_simulations):
        objective = objective + torch.sum(weights * torch.abs(simulated_data[i] - empirical_data))
    
    objective = objective / n_simulations
    return objective

def optimise_model(a_param, b_param, log_sigma_param, raw_k_param, n_steps, lr, model_type, paradigm, empirical_data, weights, n_simulations, v, gaussian_noise_all, tuning_curves_indices_all, sub_num, N):
    optimiser = torch.optim.Adam([a_param, b_param, log_sigma_param, raw_k_param], lr=lr)
    loss_list = []

    for step in range(n_steps):
        optimiser.zero_grad()
        sigma_param = torch.exp(log_sigma_param)
        k_param = torch.nn.functional.softplus(raw_k_param)
        simulated_data = produce_slopes_multiple_simulations(sigma_param, a_param, b_param, k_param, model_type, paradigm, n_jobs, n_simulations, v, gaussian_noise_all, tuning_curves_indices_all, sub_num, N)
        loss = objective_function(simulated_data, empirical_data, weights)
        # loss.backward()
        loss.backward(retain_graph=True)
        # dot =torchviz.make_dot(loss, params = {"a": a_param, "b": b_param, "log_sigma": log_sigma_param, "raw_k": raw_k_param, "loss":loss})
        # dot.render("computation_graph", format="png")
        print("step")
        print(step)
        print(f"  a: {a_param.item():.4f}, grad: {a_param.grad.item():.6f}")
        if model_type in {2, 3, 5, 6, 8, 9, 11, 12}:
            print(f"  b: {b_param.item():.4f}, grad: {b_param.grad.item():.6f}")
        print(f"  sigma: {torch.exp(sigma_param).item():.4f}, grad: {log_sigma_param.grad.item():.6f}")
        print(f"  k: {k_param.item():.4f}, grad (raw): {raw_k_param.grad.item():.6f}")
        print(f"  Loss: {loss.item():.6f}")
        loss_list.append(loss.item())
        optimiser.step()
        print(f"  Updated Parameters:")
        print(f"    a:     {a_param.item(): .4f}")
        if model_type in {2, 3, 5, 6, 8, 9, 11, 12}:
            print(f"    b:     {b_param.item(): .4f}")
        print(f"    sigma: {sigma_param.item(): .4f}")
        print(f"    k:     {k_param.item(): .4f}")
    steps_array = np.linspace(0, n_steps - 1, n_steps)
    plt.plot(steps_array, loss_list)
    plt.show()

a_init = 0.9
b_init = 0.9
sigma_init = 0.9
k_init = 0.9

params = torch.nn.Parameter(torch.tensor([a_init, b_init, sigma_init], dtype=torch.float32, requires_grad=True))
a_param = torch.tensor(a_init, dtype=torch.float32, requires_grad=True)
b_param = torch.tensor(b_init, dtype=torch.float32, requires_grad=True)
log_sigma_param = torch.tensor(-2.3026, dtype=torch.float32, requires_grad=True)
raw_k_param = torch.tensor([torch.log(torch.exp(torch.tensor(k_init)) - 1)], dtype=torch.float32, requires_grad=True)
weights = 1/6 *torch.ones(6, requires_grad=True)

models = {
    'global scaling' : 1,
    'local scaling' : 2,
    'remote scaling' : 3,
    'global sharpening' : 4,
    'local sharpening' : 5,
    'remote sharpening' : 6,
    'global repulsion' : 7,
    'local repulsion' : 8,
    'remote repulsion' : 9,
    'global attraction' : 10,
    'local attraction' : 11,
    'remote attraction' : 12
}

n_jobs = 1

empirical_face_data = torch.tensor([-0.3657197926719654, -0.0337486931658741, -0.020352066375412297, -0.013396626790461809, 0.04041971183397424, 0.1837277393276265], requires_grad=True)
empirical_grating_data = torch.tensor([-0.6499, -0.0525, -0.0610, 0.0085, -0.0233, 0.2581])
#In order AM, WC, BC, CP, AMS, AMA


"""raw_k_param is the unconstrained variable that I am actually optimising (can be negative)"""
"""k_param is a smooth, always positive transformation that is used in the model"""

#Defining added gaussian noise so each e.g simulation 1 for a parameter set is the same as simulation 1 for a different parameter set
#Need noise to be gaussian * 0.03 with a n_simulations * n_trials * v

paradigm = 'grating'
n_simulations = 4
v = 200
N = 8
sub_num = 18
if paradigm == 'face':
    n_trials = 32
elif paradigm == 'grating':
    n_trials = 48

gaussian_noise_all = 0.03 * torch.randn((n_simulations, sub_num, n_trials, v))
tuning_curves_indices_all = torch.randint(0, N, (n_simulations, sub_num, v,N))


optimise_model(
    a_param=a_param, 
    b_param=b_param, 
    log_sigma_param=log_sigma_param, 
    raw_k_param=raw_k_param, 
    n_steps=5, 
    lr=0.1, 
    model_type=2, 
    paradigm=paradigm, 
    empirical_data=empirical_face_data if paradigm == 'face' else empirical_grating_data, 
    weights=weights,
    n_simulations = n_simulations,
    v = v,
    gaussian_noise_all=gaussian_noise_all,
    tuning_curves_indices_all=tuning_curves_indices_all,
    sub_num=sub_num,
    N=N)