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

faceData = 'face_data.mat' #We need to change the data
gratingData = 'grating_data.mat' #We need to change the data

def simulate_subject(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind):
    """Produces the voxel pattern for one simulation for one parameter combination of one paradigm"""

    out = simulate_adaptation(v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N)
    print("out.grad_fn")
    print(out.grad_fn)
    # /pattern = (out.T + torch.randn(v, len(j), requires_grad=True) * noise).T
    pattern = out + torch.randn_like(out)*noise
    v = pattern.shape[1]
    if paradigm == 'face':
        return torch.vstack((pattern[::4, :v], pattern[1::4, :v], pattern[2::4, :v], pattern[3::4, :v]))
    elif paradigm == 'grating':
        return torch.vstack((pattern[ind[0], :v], pattern[ind[2], :v], pattern[ind[3], :v], pattern[ind[5], :v]))

def produce_statistics_one_simulation(paradigm, model_type, sigma, a, b, n_jobs, n_simulations):
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
    """Removing Parallel Processing for now"""
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(simulate_subject)(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind)
    #     for sub in range(sub_num)
    # )

    # print(results)

    results = torch.stack(
        [simulate_subject(sub, v, X, j, cond1, cond2, a, b, sigma, model_type, reset_after, paradigm, N, noise, ind)
        for sub in range(sub_num)
        ])
    print("results.grad_fn")
    print(results.grad_fn)
    print(results)
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

def simulate_data(sigma, a, b, model_type, paradigm, n_jobs):
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
    results = produce_statistics_one_simulation(paradigm, model_type, sigma, a, b, n_jobs, n_simulations)
    print("results.grad_fn")
    print(results.grad_fn)
    print(results)
    # print("results")
    # print(results)
    # print("results.grad_fn")
    # print(results.grad_fn.next_functions)
    return results

def empirical_data(paradigm):
    sub_num = 18
    results = produce_statistics_empirical(paradigm, sub_num)
    return results

# print(simulate_data(0.1, 0.1, 0.1, 1, 'face', n_jobs))
# print(empirical_data('face'))

weights = 1/6 * torch.ones(6)

example_simulated_data = torch.tensor([-0.11214855688292176, -0.7090723202895317, 0.1294631477874019, -0.8385354680769335, -0.001882580680022596, 0.06292158608015454])
example_empiricaled_data = torch.tensor([-0.3657197926719654, -0.0337486931658741, -0.020352066375412297, -0.013396626790461809, 0.04041971183397424, 0.1837277393276265])

def objective_function(simulated_data, empiricaled_data, weights):
    # print(simulated_data)
    # print(empiricaled_data)
    # print(simulated_data)
    # print(empiricaled_data)
    objective=torch.sum(weights * torch.abs(simulated_data - empiricaled_data))
    # print(objective.grad_fn)
    return objective

# print(objective_function(example_simulated_data, example_empiricaled_data, weights))

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


def optimise_model(a_param, b_param, log_sigma_param, n_steps, lr, model_type, paradigm, empirical_data, weights):
    optimiser = torch.optim.Adam([a_param, b_param, log_sigma_param], lr=lr)

    for step in range(n_steps):
        optimiser.zero_grad()
        sigma_param = torch.exp(log_sigma_param)
        simulated_data = simulate_data(sigma_param, a_param, b_param, model_type, paradigm, n_jobs)
        print("simulated_data.grad_fn")
        print(simulated_data.grad_fn)
        print(simulated_data)
        loss = objective_function(simulated_data, empirical_data, weights)
        print("loss.grad_fn")
        print(loss.grad_fn)
        print(loss)
        # loss.backward()
        loss.backward(retain_graph=True)
        dot =torchviz.make_dot(loss, params = {"a": a_param, "b": b_param, "sigma": sigma_param, "loss":loss})
        dot.render("computation_graph", format="png")
        print(step)
        print(f"  a: {a_param.item():.4f}, grad: {a_param.grad.item():.6f}")
        print(f"  b: {b_param.item():.4f}, grad: {b_param.grad.item():.6f}")
        print(f"  sigma: {torch.exp(sigma_param).item():.4f}, grad: {log_sigma_param.grad.item():.6f}")
        print(f"  Loss: {loss.item():.6f}")
        optimiser.step()
        print(f"  Updated Parameters:")
        print(f"    a:     {a_param.item(): .4f}")
        print(f"    b:     {b_param.item(): .4f}")
        print(f"    sigma: {sigma_param.item(): .4f}")

a_init = 0.1
b_init = 0.1
sigma_init = 0.1

params = torch.nn.Parameter(torch.tensor([a_init, b_init, sigma_init], dtype=torch.float32, requires_grad=True))
a_param = torch.tensor(a_init, dtype=torch.float32, requires_grad=True)
b_param = torch.tensor(b_init, dtype=torch.float32, requires_grad=True)
log_sigma_param = torch.tensor(-2.3026, dtype=torch.float32, requires_grad=True)
# params = torch.nn.Parameter(params.detach().numpy())
# print(params)
weights = 1/6 *torch.ones(6, requires_grad=True)

optimise_model(a_param=a_param, b_param=b_param, log_sigma_param=log_sigma_param, n_steps=6, lr=0.01, model_type=2, paradigm='face', empirical_data=example_empiricaled_data, weights=weights)