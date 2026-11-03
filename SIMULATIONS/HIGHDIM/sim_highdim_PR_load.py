import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from SIMULATIONS.HIGHDIM.sim_highdim_CR_load import plot_heatmap
from decoding_utils import decode
import pandas as pd
from plotting_style import *
from constants import *
from scipy.stats import bootstrap
import pickle
from scipy.stats import combine_pvalues
import glob


"""
Load PR results for high-dimensional simulations
"""

settings_name_global = 'OCT2_NFEAT_TEST_MD4'
settings_name_global = 'OCT2_NFEAT_TEST_MD4_MORESAMP'

modes = ['nocalib_notune',
         'calib_sigmoid',
         'calib_isotonic',
         'calib_vennabers',
         'nocalib_tune']

# plot options
dpi = 400
save_plots = True
plot_lineplot = True
plot_heatmap = True


alpha_level_permutations = 0.05
alpha_level_partial_test = 0.05
alpha_level_permutations_aggregate = 0.05

plot_format = 'png'

plots_folder_general = os.path.join(DATA_PATH, 'plots', 'simulations_HIGHDIM_CAL',
                                    'DECODE_SETTINGS_{}'.format(settings_name_global))
if not os.path.isdir(plots_folder_general):
    os.makedirs(plots_folder_general)

df_all = []
for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)

    plots_folder = os.path.join(DATA_PATH, 'plots', 'simulations_HIGHDIM_CAL', 'DECODE_SETTINGS_{}'.format(settings_name_global), mode)
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)

    dfs = []
    pars_list = []
    data = {}
    output_folder = os.path.join(DATA_PATH, 'simulations_highdim_PR_results',
                                 'SIMULATION_{}'.format(settings_name_global))
    files = glob.glob('{}/results_{}*'.format(output_folder, settings_name))
    print(files)

    for file in files:
        res = pickle.load(open(file, 'rb'))
        df = res['df']
        assert df['simulation_type'].unique().shape[0] == 1
        dfs.append(df)
        pars_list.append(res['pars'])
        simulation_types = res['pars']['simulation_types']
    df = pd.concat(dfs)

    dm = df.copy()
    dm['calibration_method'] = res['pars']['calibration_method']
    dm['mode'] = mode
    df_all.append(dm)

    df1 = dfs[0]
    vals = df1[(df1['simulation_type'] == 'linear_highdim') &
               (df1['parameter_value'] == 10) &
               (df1['decoder'] == 'none') &
               (df1['method'] == 'predictability_nonlin_rf') &
               (df1['permuted'] == False)]['mean_score'].__array__()

    pars = pars_list[0]
    pars['figure'] = 'figure2'

    simulation_types = df['simulation_type'].unique()

    deconf_methods = pars['deconf_methods']

    deconf_methods_lineplot = [d for d in deconf_methods if d != 'decode_from_confound']
    n_repeats = pars['n_repeats']
    n_permutations = pars['n_permutations']
    parameter_to_vary = pars['parameter_to_vary']
    parameter_values = np.sort(pars['parameter_values'])

    pred_methods_lp = [d for d in deconf_methods_lineplot if d in predictability_methods]
    other_methods_lp = [d for d in deconf_methods_lineplot if d not in predictability_methods]

    if plot_lineplot:

        for simulation_type in simulation_types:

            for method_set, method_set_label in zip([pred_methods_lp], ['predictability']):

                if method_set_label == 'predictability':
                    decoders_to_iterate = ['none']

                for decoder in decoders_to_iterate:

                    f, ax = plt.subplots(1, len(deconf_methods), figsize=[2.5*len(deconf_methods), 2.5],
                                         sharey=True)
                    if len(deconf_methods)==1:
                        ax = [ax]

                    if method_set_label == 'predictability':
                        if simulation_type == 'nonlinear_5_noise':
                            ylim = [-0.1, 0.7]
                            yticks = [-0.2, 0, 0.2, 0.4, 0.6]
                        elif np.isin(simulation_type,['joint_XOR_true_2', 'linear_true_2']):
                            ylim = [-0.1, 0.5]
                            yticks = [-0.2, 0, 0.2, 0.4]
                    else:
                        ylim = [0.3, 1]
                        yticks = [0.5, 0.75, 1.0]

                    for j, deconf_method in enumerate(method_set):
                        if deconf_method == 'decode_from_confound':
                            continue
                        if deconf_method == 'linear_confound_regression':
                            raise ValueError
                        x = []
                        err = []
                        perms_range = []
                        for k, parameter_value in enumerate(parameter_values):

                            decsel = decoder

                            vals = df[(df['simulation_type'] == simulation_type) &
                                      (df['parameter_value'] == parameter_value) &
                                      (df['decoder'] == decsel) &
                                      (df['method'] == deconf_method) &
                                      (df['permuted'] == False)]['mean_score'].__array__()
                            print(vals)
                            assert vals.shape[0] == 1

                            perms = df[(df['simulation_type'] == simulation_type) &
                                      (df['parameter_value'] == parameter_value) &
                                      (df['decoder'] == decsel) &
                                      (df['method'] == deconf_method) &
                                      (df['permuted'] == True)]['mean_score'].__array__()
                            center = vals.mean()

                            x.append(center)

                            if n_repeats > 1:
                                bootstrap_vals = (vals.__array__().reshape(-1, 1).astype(float),)
                                bres = bootstrap(bootstrap_vals,
                                                 axis=0,
                                                 statistic=np.mean,
                                                 n_resamples=100,
                                                 confidence_level=0.95,
                                                 method='basic')
                                error = np.array([bres.confidence_interval.low[0], bres.confidence_interval.high[0]])
                                err.append(error)

                            if n_permutations > 1:
                                p_range_cats, p_range = pd.qcut(perms.astype(float), q=[0.025, 0.975], retbins=True)
                                perms_range.append(p_range)

                        ax[j].plot(parameter_values, x, c=method_palette[deconf_method], zorder=-2)
                        ax[j].scatter(parameter_values, x, s=30, c=method_palette[deconf_method],
                                   edgecolor='w', label=method_labels[deconf_method], zorder=-1)
                        if n_repeats > 1:
                            err = np.vstack(err)
                            ax[j].fill_between(parameter_values, err[:, 0], err[:, 1], alpha=0.5,
                                            color=method_palette[deconf_method], linewidth=0)

                        if n_permutations > 1:
                            perms_range = np.vstack(perms_range)
                            ax[j].fill_between(parameter_values, perms_range[:, 0], perms_range[:, 1], alpha=0.3,
                                            color=method_palette[deconf_method], linewidth=0)

                    if n_permutations > 0:
                        dp = pd.DataFrame(columns=['parameter_value', 'method', 'pvalue'])
                        for j, deconf_method in enumerate(method_set):
                            x = []
                            err = []
                            for k, parameter_value in enumerate(parameter_values):
                                pvals = []
                                for nrep in range(n_repeats):

                                    if np.isin(deconf_method, predictability_methods):
                                        decsel = 'none'
                                    else:
                                        decsel = decoder

                                    dfsel = df[(df['simulation_type'] == simulation_type) &
                                              (df['parameter_value'] == parameter_value) &
                                              (df['decoder'] == decsel) &
                                              (df['method'] == deconf_method) &
                                              (df['repeat'] == nrep)]

                                    obs = dfsel[dfsel['permuted'] == False]['mean_score'].__array__()
                                    assert obs.shape[0] == 1
                                    obs = obs[0]
                                    perms = dfsel[dfsel['permuted'] == True]['mean_score'].__array__()
                                    p_val = (perms > obs).sum() / (n_permutations + 1)
                                    pvals.append(p_val)

                                res = combine_pvalues(pvals, method='pearson')

                                dp.loc[dp.shape[0], :] = [parameter_value,
                                                          deconf_method,
                                                          res.pvalue]

                        left_edges = np.array(parameter_values) - np.hstack(([0], np.diff(parameter_values) / 2))
                        right_edges = np.array(parameter_values) + np.hstack((np.diff(parameter_values) / 2, [0]))

                        for j, deconf_method in enumerate(method_set):
                            y = np.max([axx.get_ylim()[1] for axx in ax])
                            dpsel = dp[dp['method'] == deconf_method]
                            for ii, (rowix, row) in enumerate(dpsel.iterrows()):
                                if row['pvalue'] < alpha_level_permutations_aggregate:
                                    ax[j].plot([left_edges[ii], right_edges[ii]], [y, y],
                                             c=method_palette[deconf_method], linewidth=3)

                    for j, ax1 in enumerate(ax):
                        ax1.set_xticks(parameter_values)
                        ax1.set_xticklabels(parameter_values)
                        ax1.set_xlabel(parameter_labels[parameter_to_vary])
                        if method_set_label == 'predictability':
                            ax1.set_ylabel('Bits')
                            ax1.axhline(0.0, c='grey', ls=':')
                        else:
                            ax1.set_ylabel('Decoding accuracy')
                            ax1.axhline(0.5, c='grey', ls=':')
                        ax1.set_title(pr_labels_detailed[deconf_methods[j]], fontsize=10)
                        sns.despine(right=True)
                    plt.tight_layout()
                    plt.show()

                    if save_plots:
                        plot_name = 'methods_inspection_{}_{}_{}_{}.{}'.format(method_set_label, simulation_type, parameter_to_vary,
                                                                               decoder, plot_format)
                        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                                  bbox_inches="tight")




dm = pd.concat(df_all)

if plot_heatmap:
    dmp = pd.DataFrame(columns=['mode', 'simulation_type', 'decoder', 'deconf_method',
                               'parameter_value', 'pvalue'])

    for mode in modes:
        for simulation_type in simulation_types:
            for method_set, method_set_label in zip([pred_methods_lp], ['predictability']):
                if method_set_label == 'predictability':
                    decoders_to_iterate = ['none']

                for decoder in decoders_to_iterate:
                    for j, deconf_method in enumerate(method_set):
                        for k, parameter_value in enumerate(parameter_values):
                            pvals = []
                            for nrep in range(n_repeats):
                                if np.isin(deconf_method, predictability_methods):
                                    decsel = 'none'
                                else:
                                    decsel = decoder

                                dmsel = dm[(dm['mode'] == mode) &
                                           (dm['simulation_type'] == simulation_type) &
                                           (dm['parameter_value'] == parameter_value) &
                                           (dm['decoder'] == decsel) &
                                           (dm['method'] == deconf_method) &
                                           (dm['repeat'] == nrep)]

                                obs = dmsel[dmsel['permuted'] == False]['mean_score'].__array__()
                                assert obs.shape[0] == 1
                                obs = obs[0]
                                perms = dmsel[dmsel['permuted'] == True]['mean_score'].__array__()
                                p_val = (perms > obs).sum() / (n_permutations + 1)
                                pvals.append(p_val)

                            if n_repeats > 1:
                                res = combine_pvalues(pvals, method='pearson')
                                pvalue = res.pvalue
                            else:
                                pvalue = pvals[0]

                            dmp.loc[dmp.shape[0], :] = [mode, simulation_type, decoder,
                                                      deconf_method, parameter_value,
                                                      pvalue]





    f, ax = plt.subplots(len(simulation_types), len(deconf_methods), sharey=True,
                         sharex=True, figsize=[8,8])
    if len(deconf_methods) == 1:
        ax = ax[:, np.newaxis]
    for i, simulation_type in enumerate(simulation_types):
        for j, deconf_method in enumerate(method_set):
            dmpsel = dmp[(dmp['simulation_type'] == simulation_type) &
                          (dmp['deconf_method'] == deconf_method)]
            sig_arr = np.zeros(shape=(len(parameter_values), len(modes)))
            pval_arr = np.zeros(shape=(len(parameter_values), len(modes)))
            for ii, parameter_value in enumerate(parameter_values):
                for jj, mode in enumerate(modes):
                    pval = dmpsel[(dmpsel['parameter_value'] == parameter_value) &
                           (dmpsel['mode'] == mode)]['pvalue'].iloc[0]
                    pval_arr[ii, jj] = pval
                    if pval < alpha_level_permutations:
                        sig_arr[ii, jj] = 1
                    else:
                        sig_arr[ii, jj] = 0

            sns.heatmap(data=sig_arr, xticklabels=modes, yticklabels=parameter_values, ax=ax[i, j],
                        vmin=0, vmax=1, annot=pval_arr.round(2), cbar=False)
            if j==0:
                ax[i,j].set_ylabel(highdim_simulation_labels[simulation_type])
            if i==0:
                ax[i, j].set_title('{}'.format(pr_labels_detailed[deconf_method]))
            ax[i,j].set_xticklabels([calibration_modes_labels[m] for m in modes])
    sns.despine()
    plt.tight_layout()
    plt.show()

    if save_plots:
        plot_name = 'heatmap_pvalues_{}.{}'.format(settings_name_global, plot_format)
        f.savefig(os.path.join(plots_folder_general, plot_name), dpi=250,
                  bbox_inches="tight")

