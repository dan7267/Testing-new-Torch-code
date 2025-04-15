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

def simulate_subject(sub, v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, noise, ind):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""

    out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N)
    # /pattern = (out.T + torch.randn(v, len(j), requires_grad=True) * noise).T
    pattern = out + torch.randn_like(out)*noise
    v = pattern.shape[1]
    if paradigm == 'face':
        return torch.vstack((pattern[::4, :v], pattern[1::4, :v], pattern[2::4, :v], pattern[3::4, :v]))
    elif paradigm == 'grating':
        return torch.vstack((pattern[ind[0], :v], pattern[ind[2], :v], pattern[ind[3], :v], pattern[ind[5], :v]))

def produce_statistics_one_simulation(paradigm, model_type, sigma, a, b, k, n_jobs, n_simulations):
    """Produces the slope of each data feature for one parameter combination for one simulation"""
    # v, X = 200, torch.pi
    v, X = 200, torch.pi
    cond1, cond2 = X/4, 3*X/4
    sub_num = 18
    noise = 0.03
    N = 8
    # y = {}
    y = torch.zeros(sub_num, dtype=torch.float32)

    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)

    results = torch.stack(
        [simulate_subject(sub, v, X, j, cond1, cond2, a, b, sigma, k, model_type, reset_after, paradigm, N, noise, ind)
        for sub in range(sub_num)
        ])
    return produce_confidence_interval(results, 1)

def process_empirical_subject(sub, paradigm):
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

def produce_statistics_empirical(paradigm, sub_num):
    y = Parallel(n_jobs=-1)(delayed(process_empirical_subject)(sub, paradigm) for sub in range(sub_num))
    return produce_confidence_interval(y, 1)

def simulate_data(sigma, a, b, k, model_type, paradigm, n_jobs):
    """Simulate data using given parameters, with specific random seed
    so that each run has different random variations but is reproducible for the
    same parameter set. Note this can also be done by generating the random
    array first and then adding this appropriately each time."""
    # torch.manual_seed(seed)
    # simulated = ...
    X = torch.pi
    cond1, cond2 = X/4, 3*X/4
    n_simulations = 1
    simulation = True
    j, ind, reset_after, _ = paradigm_setting(paradigm, cond1, cond2)
    results = produce_statistics_one_simulation(paradigm, model_type, sigma, a, b, k, n_jobs, n_simulations)
    return results

def empirical_data(paradigm):
    sub_num = 18
    results = produce_statistics_empirical(paradigm, sub_num)
    return results

weights = 1/6 * torch.ones(6)

example_simulated_data = torch.tensor([-0.11214855688292176, -0.7090723202895317, 0.1294631477874019, -0.8385354680769335, -0.001882580680022596, 0.06292158608015454])
example_empiricaled_data = torch.tensor([-0.3657197926719654, -0.0337486931658741, -0.020352066375412297, -0.013396626790461809, 0.04041971183397424, 0.1837277393276265])

def objective_function(simulated_data, empiricaled_data, weights):
    objective=torch.sum(weights * torch.abs(simulated_data - empiricaled_data))
    return objective

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

parameters = {
    #Do we want parameter ranges?
    'sigma' : [0.1, 1.5],
    'a' : [0.1, 2],
    'b' : [0.1, 5],
    'y' : [0.1, 100]
}

n_jobs = 1

example_empiricaled_data = torch.tensor([-0.3657197926719654, -0.0337486931658741, -0.020352066375412297, -0.013396626790461809, 0.04041971183397424, 0.1837277393276265], requires_grad=True)

def optimise_model(a_param, b_param, log_sigma_param, raw_k_param, n_steps, lr, model_type, paradigm, empirical_data, weights):
    optimiser = torch.optim.Adam([a_param, b_param, log_sigma_param, raw_k_param], lr=lr)
    loss_list = []

    for step in range(n_steps):
        optimiser.zero_grad()
        sigma_param = torch.exp(log_sigma_param)
        k_param = torch.nn.functional.softplus(raw_k_param)
        simulated_data = simulate_data(sigma_param, a_param, b_param, k_param, model_type, paradigm, n_jobs)
        loss = objective_function(simulated_data, empirical_data, weights)
        # loss.backward()
        loss.backward(retain_graph=True)
        # dot =torchviz.make_dot(loss, params = {"a": a_param, "b": b_param, "log_sigma": log_sigma_param, "raw_k": raw_k_param, "loss":loss})
        # dot.render("computation_graph", format="png")
        print("step")
        print(step)
        print(f"  a: {a_param.item():.4f}, grad: {a_param.grad.item():.6f}")
        print(f"  b: {b_param.item():.4f}, grad: {b_param.grad.item():.6f}")
        print(f"  sigma: {torch.exp(sigma_param).item():.4f}, grad: {log_sigma_param.grad.item():.6f}")
        print(f"  k: {k_param.item():.4f}, grad (raw): {raw_k_param.grad.item():.6f}")
        print(f"  Loss: {loss.item():.6f}")
        loss_list.append(loss.item())
        optimiser.step()
        print(f"  Updated Parameters:")
        print(f"    a:     {a_param.item(): .4f}")
        print(f"    b:     {b_param.item(): .4f}")
        print(f"    sigma: {sigma_param.item(): .4f}")
        print(f"    k:     {k_param.item(): .4f}")
    steps_array = np.linspace(0, n_steps - 1, n_steps)
    # print(steps_array)
    # print(loss_list)
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
raw_k_param = torch.tensor([torch.log(torch.exp(torch.tensor(k_init)) - 1)], dtype=torch.float32, requires_grad=True)# params = torch.nn.Parameter(params.detach().numpy())
# print(params)
weights = 1/6 *torch.ones(6, requires_grad=True)

"""raw_k_param is the unconstrained variable that I am actually optimising (can be negative)"""
"""k_param is a smooth, always positive transformation that is used in the model"""

optimise_model(
    a_param=a_param, 
    b_param=b_param, 
    log_sigma_param=log_sigma_param, 
    raw_k_param=raw_k_param, 
    n_steps=30, 
    lr=0.1, 
    model_type=2, 
    paradigm='face', 
    empirical_data=example_empiricaled_data, 
    weights=weights)