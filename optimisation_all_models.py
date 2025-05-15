from joblib import Parallel, delayed
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

def simulate_subject(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, ind, gaussian_noise, tuning_curves_indices, sub_num):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""
    T = len(j)
    noisy_pattern = torch.empty((sub_num, T, v), dtype = torch.float32)
    batch_size = 9
    for i in range(0, sub_num, batch_size):
        current_batch_size = min(batch_size, sub_num - i)
        out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, tuning_curves_indices[i:i+current_batch_size, :, :], current_batch_size)
        noisy_pattern[i:i+current_batch_size] = out + gaussian_noise[i:i+current_batch_size,:,:]
    if paradigm == 'face':
        # Build condition indices using torch.arange
        cond1_p1 = torch.arange(0, 32, 4)
        cond1_p2 = torch.arange(1, 32, 4)
        cond2_p1 = torch.arange(2, 32, 4)
        cond2_p2 = torch.arange(3, 32, 4)

        # Stack condition slices
        pattern_split = torch.stack([
            noisy_pattern[:, cond1_p1, :],
            noisy_pattern[:, cond1_p2, :],
            noisy_pattern[:, cond2_p1, :],
            noisy_pattern[:, cond2_p2, :]
        ], dim=1)  # shape: (batch_size, 4, T/4, v)

        reshaped = pattern_split.reshape(sub_num, 32, v)  # shape: (batch, 32, voxel)

        return reshaped
    elif paradigm == 'grating':
        pattern_split = torch.stack([
            noisy_pattern[:, ind[0], :],
            noisy_pattern[:, ind[2], :],
            noisy_pattern[:, ind[3], :],
            noisy_pattern[:, ind[5], :]
        ], dim=1)
        reshaped = pattern_split.reshape(sub_num, 32, v)
        return reshaped

def produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, k, n_jobs, n_simulations, v, gaussian_noise_near, tuning_curves_indices_near, sub_num, N, j, ind, reset_after):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    X = torch.pi
    sub_num = 18
    cond1, cond2 = X/4, 3*X/4
    N = 8
    y = simulate_subject(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, ind, gaussian_noise_near, tuning_curves_indices_near, sub_num)
    return produce_confidence_interval(y, 1)

def produce_slopes_multiple_simulations(sigma, a, b, k, model_type, paradigm, n_jobs, n_simulations, v, gaussian_noise_all, tuning_curves_indices_all, sub_num, N, j, ind, reset_after):
    #Is this function necessary. Yes, currently just one simulation
    """Simulate data using given parameters, with specific random seed
    so that each run has different random variations but is reproducible for the
    same parameter set. Note this can also be done by generating the random
    array first and then adding this appropriately each time."""
    # torch.manual_seed(seed)
    # simulated = ...
    X = torch.pi
    results = torch.zeros((n_simulations, 6))
    for sim in range(n_simulations):
        gaussian_noise_near=gaussian_noise_all[sim] #size sub_num, trials, voxels
        tuning_curves_indices_near = tuning_curves_indices_all[sim] #size sub_num, v, N
        results[sim] = produce_slopes_one_simulation(paradigm, model_type, sigma, a, b, k, n_jobs, n_simulations, v, gaussian_noise_near, tuning_curves_indices_near, sub_num, N, j, ind, reset_after)
    return results

def objective_function(simulated_data, empirical_data, weights):
    """Now set up to run over many simulations. Instead of simulated_data being the 6 slopes in a 1D tensor,
    simulated_data will be an n_simulations x 6 tensor which is then averaged at the end"""
    n_simulations = simulated_data.shape[0]
    objective = 0
    # print("simulated_data NaN:", torch.isnan(simulated_data).any())
    # print("simulated_data:", simulated_data)
    for i in range(n_simulations):
        objective = objective + torch.sum(weights * torch.abs(simulated_data[i] - empirical_data))
    
    objective = objective / n_simulations
    return objective

def optimise_single_model(
    model_name, model_type, n_steps, lr, paradigm, empirical_data, weights,
    n_simulations, v, gaussian_noise_all, tuning_curves_indices_all,
    sub_num, N, j, ind, reset_after, a_init, b_init, sigma_init, k_init, n_jobs
):
    a_param = torch.nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
    raw_b_param = torch.nn.Parameter(torch.log(torch.exp(torch.tensor(b_init)) - 1).unsqueeze(0))
    log_sigma_param = torch.nn.Parameter(torch.tensor(-2.3026, dtype=torch.float32))  # log(0.1)
    raw_k_param = torch.nn.Parameter(torch.log(torch.exp(torch.tensor(k_init)) - 1).unsqueeze(0))

    optimiser = torch.optim.Adam([a_param, raw_b_param, log_sigma_param, raw_k_param], lr=lr)

    history = {
        'loss': [],
        'a': [],
        'b': [],
        'sigma': [],
        'k': []
    }

    for step in range(n_steps):
        optimiser.zero_grad()
        sigma_param = torch.exp(log_sigma_param)
        k_param = torch.nn.functional.softplus(raw_k_param)
        b_param = torch.nn.functional.softplus(raw_b_param)

        simulated_data = produce_slopes_multiple_simulations(
            sigma_param, a_param, b_param, k_param, model_type,
            paradigm, n_jobs, n_simulations, v,
            gaussian_noise_all, tuning_curves_indices_all,
            sub_num, N, j, ind, reset_after
        )

        loss = objective_function(simulated_data, empirical_data, weights)
        loss.backward(retain_graph=True)
        optimiser.step()

        # Store parameters and loss
        history['loss'].append(loss.item())
        history['a'].append(a_param.detach().item())
        history['b'].append(b_param.detach().item())
        history['sigma'].append(sigma_param.detach().item())
        history['k'].append(k_param.detach().item())

    final_loss = history['loss'][-1]
    best_params = {
        'a': history['a'][-1],
        'b': history['b'][-1],
        'sigma': history['sigma'][-1],
        'k': history['k'][-1],
    }

    return {
        'model_name': model_name,
        'model_type': model_type,
        'final_loss': final_loss,
        'best_params': best_params,
        'history': history
    }

def find_best_model(
    models, n_steps, lr, paradigm, empirical_data, weights, n_simulations,
    v, gaussian_noise_all, tuning_curves_indices_all, sub_num, N, j, ind,
    reset_after, a_init=0.9, b_init=0.9, sigma_init=0.9, k_init=0.9, n_jobs=1, parallel=True
):
    tasks = [
        delayed(optimise_single_model)(
            model_name, model_type, n_steps, lr, paradigm, empirical_data, weights,
            n_simulations, v, gaussian_noise_all, tuning_curves_indices_all,
            sub_num, N, j, ind, reset_after, a_init, b_init, sigma_init, k_init, n_jobs
        )
        for model_name, model_type in models.items()
    ]

    if parallel:
        results = Parallel(n_jobs=min(len(models), n_jobs))(tasks)
    else:
        results = [task() for task in tasks]

    best_result = min(results, key=lambda x: x['final_loss'])

    # Reformat results dictionary for easier downstream use
    all_results = {
        result['model_name']: {
            'type': result['model_type'],
            'final_loss': result['final_loss'],
            'best_params': result['best_params'],
            'history': result['history']
        }
        for result in results
    }

    return {
        'best_model_name': best_result['model_name'],
        'best_model_type': best_result['model_type'],
        'best_loss': best_result['final_loss'],
        'best_params': best_result['best_params'],
        'all_results': all_results
    }

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

paradigm = 'face'
empirical_face_data = torch.tensor([-0.3657197926719654, -0.0337486931658741, -0.020352066375412297, -0.013396626790461809, 0.04041971183397424, 0.1837277393276265], requires_grad=True)
empirical_grating_data = torch.tensor([-0.6499, -0.0525, -0.0610, 0.0085, -0.0233, 0.2581])
weights = 1/6 *torch.ones(6, requires_grad=True)
n_simulations = 5
v=200
cond1 = torch.pi/4
cond2 = torch.pi*3/4
j, ind, reset_after = paradigm_setting(paradigm, cond1, cond2)
sub_num = 18
N = 8


result = find_best_model(
    models=models,
    n_steps=20,
    lr=0.4,
    paradigm=paradigm,
    empirical_data=empirical_face_data if paradigm == 'face' else empirical_grating_data,
    weights=weights,
    n_simulations=n_simulations,
    v=v,
    gaussian_noise_all=gaussian_noise_all,
    tuning_curves_indices_all=tuning_curves_indices_all,
    sub_num=sub_num,
    N=N,
    j=j,
    ind=ind,
    reset_after=reset_after,
    n_jobs=3,         # or however many CPUs you want to use
    parallel=True
)


