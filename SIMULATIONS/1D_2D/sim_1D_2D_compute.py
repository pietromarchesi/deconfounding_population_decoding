import os
import numpy as np
import matplotlib.pyplot as plt
from decoding_utils import decode
import pandas as pd
from plotting_style import *
from constants import *
import pickle
from SIMULATIONS.simulation_utils import make_simulated_data

"""
Run deconfounding (CR and PR) on 1-D and 2-D simulations (simulations 1-8)
"""

# MAIN FIG. 2
figure = 'figure2'
settings_name = 'MAR20FINAL_figure2'

# MAIN FIG. 3
figure = 'figure3_signalstrength'
settings_name = 'MAR20FINAL_figure3'

# SUPPLEMENTARY FIG. 5
figure = 'figure3_nsamples'
settings_name = 'sep11_figure3_v3'

plot_data = False
set_seed = True
plot_format = 'png'

deconf_methods = ['raw',
                  'decode_from_confound',
                  'linear_confound_regression',
                  'nonlinear_confound_regression',
                  'predictability_lin',
                  'predictability_nonlin',
                  'cmi']

tune_predictability = True
regularize_naive_model = True
LR_predictability_C_list =  [0.0001, 0.001, 0.01, 0.1, 1]
SVC_predictability_C_list = [0.0001, 0.001, 0.01, 0.1, 1]
LR_predictability_penalty = 'l2'
LR_predictability_C = 1
SVC_predictability_C = 1
LR_predictability_solver = 'saga'
LR_predictability_max_iter = 8000

tune_predictability_nfolds = 5

# counterbalancing parameters
posthoc_counterbalancing_binsize = None
posthoc_counterbalancing_n_bins = None

# decode parameters
n_kfold_splits = 5
tune_hyperpars_decoder = False
LR_decode_C_list = [0.001, 0.01, 0.1, 1, 10]
SVC_decode_C_list = [0.001, 0.01, 0.1, 1, 10]
decoders = ['logistic_regression', 'SVC']
LR_decode_C=1
SVC_decode_C=1

# other
standardize_features = True
standardize_confound = True


if figure == 'figure2':
    n_repeats = 1
    n_permutations = 2
    simulation_types = ['linear',
                        'nonlinear_3',
                        'nonlinear_4_noise',
                        'joint_XOR',
                        'joint_AND']
    n_samples = 200
    coherence_p = None
    signal_strength = None
    parameter_to_vary = 'none'
    parameter_values = ['none']

if figure == 'figure3_signalstrength':
    n_repeats = 1
    n_permutations = 200
    simulation_types = ['linear_true_2',
                        'joint_XOR_true_2',
                        'nonlinear_5_noise']
    n_samples = 200
    coherence_p = 0.9
    parameter_to_vary = 'signal_strength'
    parameter_values = [0.2, 0.4, 0.6, 0.8]
    signal_strength = parameter_values

if figure == 'figure3_nsamples':
    n_repeats = 10
    n_permutations = 200
    simulation_types = ['linear_true_2',
                        'joint_XOR_true_2',
                        'nonlinear_5_noise']
    coherence_p = 0.9
    signal_strength = 0.5
    parameter_to_vary = 'n_samples'
    parameter_values = [50, 100, 250, 500]
    n_samples = parameter_values



output_folder = os.path.join(DATA_PATH, 'simulations_1D_2D_results',
                             'SIMULATION_{}'.format(settings_name))

plots_folder = os.path.join(DATA_PATH, 'plots', 'simulations',
                            'DECODE_SETTINGS_{}'.format(settings_name))

if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


df = pd.DataFrame(columns=['simulation_type',
                           'parameter_to_vary',
                           'parameter_value',
                           'repeat',
                           'permuted',
                           'permutation',
                           'decoder',
                           'method',
                           'mean_score'])

for sim_ix, simulation_type in enumerate(simulation_types):

    print('\nSimulation: {} ({:01d} of {:01d})\n'.format(simulation_type, sim_ix+1, len(simulation_types)))

    data_keys = ['{}={}'.format(parameter_to_vary, v) for v in parameter_values]
    data = {k : {n : {} for n in range(n_repeats)} for k in data_keys}

    for k, parameter_value in enumerate(parameter_values):
        print(parameter_value)
        for nrep in range(n_repeats):

            # setting the seed here, also to ensure that the permutations are the same
            # upon which different parameter values act
            if set_seed:
                np.random.seed(nrep)

            if parameter_to_vary == 'signal_strength':
                print('Setting {} to {}'.format(parameter_to_vary, parameter_value))
                X, y, C = make_simulated_data(simulation_type=simulation_type,
                                              n_samples=n_samples,
                                              signal_strength=parameter_value,
                                              seed=nrep)

            elif parameter_to_vary == 'n_samples':
                print('Setting {} to {}'.format(parameter_to_vary, parameter_value))
                X, y, C = make_simulated_data(simulation_type=simulation_type,
                                              n_samples=parameter_value,
                                              signal_strength=signal_strength,
                                              seed=nrep)
            else:
                X, y, C = make_simulated_data(simulation_type=simulation_type,
                                              n_samples=n_samples,
                                              signal_strength=signal_strength,
                                              seed=nrep)

            # generate permutations
            # permute C and y in the same way, determine later if C permutations
            # are necessary depending on the deconf method used
            C_permuted_list, y_permuted_list = [], []
            for pix in range(n_permutations + 1):
                # print(s)
                if pix == 0:
                    C_permuted = np.copy(C)
                    y_permuted = np.copy(y)
                else:
                    random_index = np.random.permutation(np.arange(X.shape[0]))
                    C_permuted = C[random_index, :]
                    y_permuted = y[random_index]
                C_permuted_list.append(C_permuted)
                y_permuted_list.append(y_permuted)

            data_key = '{}={}'.format(parameter_to_vary, parameter_value)
            data[data_key][nrep]['X'] = X
            data[data_key][nrep]['y'] = y
            data[data_key][nrep]['C'] = C

            if plot_data and nrep == 0:

                # plot data simple for 1d X we consider both 1d and 2d C,
                # for 2d X we assume 1d C
                if X.shape[1] == 1:
                    if C.shape[1] > 1:
                        fig, ax = plt.subplots(1, 2, figsize=[5, 3], sharex=True, sharey=True)
                        for iii, axx in enumerate(ax):
                            axx.scatter(C[y == 0, iii], X[y == 0, 0], c='red', label='y=0', s=20, linewidth=0,
                                       marker='o')
                            axx.scatter(C[y == 1, iii], X[y == 1, 0], c='green', label='y=1', s=20, linewidth=0,
                                       marker='o')
                            axx.legend(frameon=False, loc='upper left')
                        # ax.axis('equal')
                            axx.axvline(0, ls=':', c='grey')
                            axx.axhline(0, ls=':', c='grey')
                            axx.legend().remove()
                        ax[0].set_xlabel('$C_1$')
                        ax[0].set_ylabel('$X$')
                        ax[1].set_xlabel('$C_2$')

                        sns.despine()
                        plt.tight_layout()



                        plot_name = 'data_single_{}_p{}.{}'.format(simulation_type, k, plot_format)
                        fig.savefig(os.path.join(plots_folder, plot_name), dpi=250,
                                    bbox_inches="tight")
                    else:
                        fig, ax = plt.subplots(1, 1, figsize=[3, 3])
                        ax.scatter(C[y==0], X[y==0, 0], c='red', label='y=0', s=20, linewidth=0,
                                   marker='o')
                        ax.scatter(C[y==1], X[y==1, 0], c='green', label='y=1', s=20, linewidth=0,
                                   marker='o')
                        ax.legend(frameon=False, loc='best')
                        #ax.axis('equal')
                        ax.legend().remove()
                        ax.axvline(0, ls=':', c='grey')
                        ax.axhline(0, ls=':', c='grey')
                        ax.set_xlabel('$C$')
                        ax.set_ylabel('$X$')
                        sns.despine()
                        plt.tight_layout()

                        plot_name = 'data_single_{}_p{}.{}'.format(simulation_type, k, plot_format)
                        fig.savefig(os.path.join(plots_folder, plot_name), dpi=250,
                                  bbox_inches="tight")

                else:
                    fig, ax = plt.subplots(1, 1, figsize=[3, 3])
                    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', label='y=0', s=20, linewidth=0,
                               marker='o')
                    ax.scatter(X[y==1, 0], X[y==1, 1], c='green', label='y=1', s=20, linewidth=0,
                               marker='o')
                    ax.legend(frameon=True, loc='best')
                    #ax.axis('equal')
                    ax.legend().remove()
                    ax.axvline(0, ls=':', c='grey')
                    ax.axhline(0, ls=':', c='grey')
                    ax.set_xlabel('$X_1$')
                    ax.set_ylabel('$X_2$')
                    sns.despine()
                    plt.tight_layout()

                    plot_name = 'data_single_{}_p{}.{}'.format(simulation_type, k, plot_format)
                    fig.savefig(os.path.join(plots_folder, plot_name), dpi=250,
                              bbox_inches="tight")


            for method_idx, deconf_method in enumerate(deconf_methods):
                if np.isin(deconf_method, predictability_methods):
                    decoders_to_iterate = ['none']
                else:
                    decoders_to_iterate = decoders

                for decoder in decoders_to_iterate:
                    print('Decoding with {}, deconf method: {}'.format(decoder, deconf_method))

                    for nperm in range(n_permutations + 1):
                        if nperm == 0:
                            permuted = False
                        elif nperm > 0:
                            permuted = True

                        y_dec = y_permuted_list[nperm]

                        # only if method is predictability we permute C together with y
                        if np.isin(deconf_method, predictability_methods):
                            C_dec = C_permuted_list[nperm]
                        else:
                            C_dec = np.copy(C)

                        # print('Decoding with {}, deconf method: {}'.format(decoder, deconf_method))
                        # print('--- repeat {:02d} of {:02d} --- permutation {:02d} of {:02d}'.format(nrep, n_repeats, nperm, n_permutations))

                        dec_out = decode(X=X, y=y_dec, C=C_dec,
                                         deconf_method=deconf_method,
                                         decoder=decoder,
                                         standardize_features=True,
                                         standardize_confound=True,
                                         n_kfold_splits=n_kfold_splits,
                                         tune_hyperpars_decoder=tune_hyperpars_decoder,
                                         tune_predictability=tune_predictability,
                                         LR_predictability_C_list=LR_predictability_C_list,
                                         SVC_predictability_C_list=SVC_predictability_C_list,
                                         SVC_predictability_C=SVC_predictability_C,
                                         LR_predictability_penalty=LR_predictability_penalty,
                                         LR_predictability_C=LR_predictability_C,
                                         LR_predictability_solver=LR_predictability_solver,
                                         tune_predictability_nfolds=tune_predictability_nfolds,
                                         LR_predictability_max_iter=LR_predictability_max_iter,
                                         LR_decode_C_list=LR_decode_C_list,
                                         SVC_decode_C_list=SVC_decode_C_list,
                                         LR_decode_C=LR_decode_C,
                                         SVC_decode_C=SVC_decode_C,
                                         regularize_naive_model=regularize_naive_model)

                        mean_score = dec_out['mean_score']
                        kfold_scores = dec_out['kfold_scores']


                        df.loc[df.shape[0], :] = [simulation_type,
                                                  parameter_to_vary,
                                                  parameter_value,
                                                  nrep,
                                                  permuted,
                                                  nperm,
                                                  decoder,
                                                  deconf_method,
                                                  dec_out['mean_score']]

            df['mean_score'] = pd.to_numeric(df['mean_score'])
            # plot bars of methods scored by accuracy

    predictability_parameters = {'tune_predictability': tune_predictability,
                                 'regularize_naive_model': regularize_naive_model,
                                 'LR_predictability_C_list': LR_predictability_C_list,
                                 'SVC_predictability_C_list': SVC_predictability_C_list,
                                 'LR_predictability_penalty': LR_predictability_penalty,
                                 'LR_predictability_C': LR_predictability_C,
                                 'SVC_predictability_C': SVC_predictability_C,
                                 'LR_predictability_solver': LR_predictability_solver,
                                 'LR_predictability_max_iter': LR_predictability_max_iter,
                                 'tune_predictability_nfolds': tune_predictability_nfolds}


    decode_parameters = {'n_kfold_splits': n_kfold_splits,
                         'tune_hyperpars_decoder': tune_hyperpars_decoder,
                         'LR_decode_C_list': LR_decode_C_list,
                         'SVC_decode_C_list': SVC_decode_C_list,
                         'decoders': decoders,
                         'LR_decode_C': LR_decode_C,
                         'SVC_decode_C': SVC_decode_C}

    pars = {'settings_name' : settings_name,
            'figure' : figure,
            'n_permutations' : n_permutations,
            'n_repeats' : n_repeats,
            'n_samples' : n_samples,
            'set_seed' : set_seed,
            'simulation_type' : simulation_type,
            'simulation_types' : simulation_types,
            'decoders' : decoders,
            'deconf_methods' : deconf_methods,
            'posthoc_counterbalancing_binsize' : posthoc_counterbalancing_binsize,
            'posthoc_counterbalancing_n_bins' : posthoc_counterbalancing_n_bins,
            'n_kfold_splits' : n_kfold_splits,
            'coherence_p' : coherence_p,
            'signal_strength' : signal_strength,
            'parameter_to_vary' : parameter_to_vary,
            'parameter_values' : parameter_values,
            'standardize_features' : standardize_features,
            'standardize_confound' : standardize_confound,
            'predictability_parameters' : predictability_parameters,
            'decode_parameters' : decode_parameters}

    out = {'pars' : pars,
           'df' : df,
           'data' : data}

    output_full_path = os.path.join(output_folder, 'results_{}_{}.pkl'.format(settings_name, simulation_type))
    print('Saving simulation to {}'.format(output_full_path))
    pickle.dump(out, open(output_full_path, 'wb'))



