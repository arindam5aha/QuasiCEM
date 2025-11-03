"""
Written By: Arindam Saha, ANU, 2025,
github: arindam5aha, arindam96@outlook.com
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import random
from tqdm import tqdm, trange

# func accepts numpy array and returns reward (-cost) and obs (optional)
# for batch execution, the return will be (array/list of all_rewards, array/list of all_obs)
# the no. of params are defined by the length of bounds as ((lb1, ub1), (lb2, ub2), ..., (lbn, ubn))
# for func returning cost and not reward, use neg_cost=-1.0

def cem_collect(func, bounds, 
                file_name='sampled_data.bz2', 
                select_frac=0.3, 
                populations=[200, 100, 50, 50, 50],  
                neg_cost=1.0,
                sampling_method='random', 
                with_obs=False, 
                batch_execute=False, 
                verbose=False,
                plot_dims=((0,1), (2,3))):
    
    assert sampling_method in ['random', 'sobol']
    assert neg_cost in [1.0, -1.0]
    if isinstance(select_frac, list):
        assert len(select_frac) == len(populations)
    # add plot dims to bounds assert
    
    data_size = sum(populations)
    iterations = len(populations)

    param_min = np.array([x for x, _ in bounds])
    param_max = np.array([x for _, x in bounds])
    max_std = 0.5*(param_max-param_min)
    param_mean = param_min + max_std
    param_std = np.ones(len(param_min))*max_std
    all_params = []
    all_rewards = []
    max_until = -100
    if with_obs: all_obs = []

    for it in range(iterations):
        if populations is not None:
            population = populations[it]
        else:
            population = data_size/iterations
        
        if isinstance(select_frac, list):
            select_frac_ = select_frac[it]
        else:
            select_frac_ = select_frac
            
        L_bound = param_mean - param_std
        U_bound = param_mean + param_std

        if sampling_method == 'random':
            params = np.random.uniform(L_bound, U_bound, (population, len(L_bound)))
        
        elif sampling_method == 'sobol':
            params = sobol_sampling(population, L_bound, U_bound)

        if not batch_execute:
            with trange(len(params)) as pbar:
                for x in pbar:
                    if with_obs:
                        r, o = func(params[x])
                        all_obs.append(o)
                        if r > max_until:
                            max_until = r
                        pbar.set_postfix(Max_Reached = max_until)
                    else:
                        r = func(params[x])
                    all_rewards.append(r)
        
        else:
            if with_obs:
                rs, os = func(params)
                for o in os: all_obs.append(o)
            else:
                rs = func(params)
            for r in rs: all_rewards.append(r)

        selection = int(np.ceil(select_frac_*population))
        best = np.argpartition(np.array(all_rewards[-population:])*neg_cost, -selection)[-selection:]
        best_params = params[best]
        param_mean, param_std = best_params.mean(axis=0), best_params.std(axis=0)

        if verbose:
            print(param_mean, param_std)
            cem_plot(params, best_params, plot_dims)

        for act in params: all_params.append(act)
        
        data = {'params': np.array(all_params), 'rewards': np.array(all_rewards)}
        if with_obs: data['obs'] = np.array(all_obs)
        save_bz2(data, file_name)

def sobol_sampling(num_samples, lower_bound, upper_bound):
    # Create Sobol sequence generator
    sobol_engine = qmc.Sobol(d=len(lower_bound), scramble=True)

    # Generate Sobol sequence samples
    sobol_samples = sobol_engine.random_base2(m=int(np.ceil(np.log2(num_samples))))

    # Truncate samples to the desired number
    sobol_samples = np.array(random.sample(sobol_samples.tolist(), num_samples))

    # Scale samples to the desired domain
    scaled_samples = lower_bound + (upper_bound - lower_bound) * sobol_samples

    return scaled_samples

def cem_plot(params, best_params, dims=((0,1), (2,3)), max_col=3):
    rows = int(np.ceil(len(dims)/max_col))
    fig, ax = plt.subplots(rows, len(dims), figsize=(15, 5))
    for i in range(len(dims)):
        ax[i].scatter(params[:,dims[i][0]], params[:, dims[i][1]])
        ax[i].scatter(best_params[:, dims[i][0]], best_params[:, dims[i][1]], color='r')
    plt.show()