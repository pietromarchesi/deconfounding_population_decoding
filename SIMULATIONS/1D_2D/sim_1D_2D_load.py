import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from decoding_utils import decode
import pandas as pd
from plotting_style import *
from constants import *
from scipy.stats import mannwhitneyu, bootstrap
import pickle
from scipy.stats import norm, gamma, combine_pvalues
import glob
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

"""
Load results and generate figures for 1D and 2D simulations (simulations 1-8)
"""


# SUPPLEMENTARY FIG. 5
settings_name = 'sep11_figure3_v3'

# MAIN FIG. 3
settings_name = 'MAR20FINAL_figure3'

# MAIN FIG. 2
settings_name = 'MAR20FINAL_figure2'

# plot options
save_plots = False
dpi = 400
plot_format = 'png'

use_first_repeat_only_for_the_violins = True
plot_only_second_to_last_parameter_for_violins = False
plot_only_first_repeat = True
plot_decode_from_confound_in_lineplot = False
plot_confound_regression = True

alpha_level_permutations = 0.05
alpha_level_partial_test = 0.05
alpha_level_permutations_aggregate = 0.05

plots_folder = os.path.join(DATA_PATH, 'plots', 'simulations', 'DECODE_SETTINGS_{}'.format(settings_name))
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)


dfs = []
pars_list = []
data = {}
output_folder = os.path.join(DATA_PATH, 'simulations_1D_2D_results', 'SIMULATION_{}'.format(settings_name))
files = glob.glob('{}/*'.format(output_folder))
simulation_types = []
for file in files:
    res = pd.read_pickle(file)
    dfs.append(res['df'])
    pars_list.append(res['pars'])
    simulation_type = res['pars']['simulation_type']
    data[simulation_type] = res['data']
    simulation_types.append(simulation_type)
df = pd.concat(dfs)
pars = pars_list[0]

if np.isin(pars['figure'], ['figure2', 'figure2_v2']):
    # plots to generate
    plot_lineplot = False
    plot_lineplot_nsamples = False
    plot_confound_regression = True
    plot_violins = True
    plot_data = True
    plot_legends = True

elif np.isin(pars['figure'], ['figure3_signalstrength']):
    # plots to generate
    plot_lineplot = True
    plot_lineplot_nsamples = False
    plot_violins = False
    plot_data = True
    plot_legends = True
    plot_confound_regression = True

elif np.isin(pars['figure'], ['figure3_nsamples']):
    plot_lineplot = False
    plot_lineplot_nsamples = True
    plot_violins = False
    plot_data = True
    plot_legends = True
    plot_confound_regression = True
    plot_significance_lineplot = True


decoders = pars['decoders']
#simulation_types = pars['simulation_types']
deconf_methods = pars['deconf_methods']
if ~ plot_decode_from_confound_in_lineplot:
    deconf_methods_lineplot = [d for d in deconf_methods if d != 'decode_from_confound']
n_repeats = pars['n_repeats']
n_permutations = pars['n_permutations']
parameter_to_vary = pars['parameter_to_vary']
parameter_values = pars['parameter_values']

pred_methods_lp = [d for d in deconf_methods_lineplot if d in predictability_methods]
other_methods_lp = [d for d in deconf_methods_lineplot if d not in predictability_methods]


if plot_lineplot:

    for simulation_type in simulation_types:

        for method_set, method_set_label in zip([pred_methods_lp, other_methods_lp],
                                                ['predictability', 'decoding']):

            if method_set_label == 'predictability':
                decoders_to_iterate = ['none']
            else:
                decoders_to_iterate = decoders

            for decoder in decoders_to_iterate:
                f, ax1 = plt.subplots(1, 1, figsize=[2.2, 3])
                #ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

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
                    x = []
                    err = []
                    for k, parameter_value in enumerate(parameter_values):

                        decsel = decoder

                        vals = df[(df['simulation_type'] == simulation_type) &
                                  (df['parameter_value'] == parameter_value) &
                                  (df['decoder'] == decsel) &
                                  (df['method'] == deconf_method) &
                                  (df['permuted'] == False)]['mean_score'].__array__()
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

                    ax = ax1

                    ax.plot(parameter_values, x, c=method_palette[deconf_method], zorder=-2)
                    ax.scatter(parameter_values, x, s=30, c=method_palette[deconf_method],
                               edgecolor='w', label=method_labels[deconf_method], zorder=-1)
                    if n_repeats > 1:
                        err = np.vstack(err)
                        ax.fill_between(parameter_values, err[:, 0], err[:, 1], alpha=0.5,
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

                                obs = dfsel[dfsel['permuted'] == False]['mean_score'].__array__()[0]
                                perms = dfsel[dfsel['permuted'] == True]['mean_score'].__array__()
                                p_val = (perms > obs).sum() / (n_permutations + 1)
                                pvals.append(p_val)

                            # this works even if you have just one repeat/pvalue
                            res = combine_pvalues(pvals, method='pearson')

                            dp.loc[dp.shape[0], :] = [parameter_value,
                                                      deconf_method,
                                                      res.pvalue]

                    left_edges = np.array(parameter_values) - np.hstack(([0], np.diff(parameter_values) / 2))
                    right_edges = np.array(parameter_values) + np.hstack((np.diff(parameter_values) / 2, [0]))

                    for j, deconf_method in enumerate(method_set):
                        y = ylim[1] - (j + 1) * 0.03
                        dpsel = dp[dp['method'] == deconf_method]
                        for ii, (rowix, row) in enumerate(dpsel.iterrows()):
                            if row['pvalue'] < alpha_level_permutations_aggregate:
                                ax1.plot([left_edges[ii], right_edges[ii]], [y, y],
                                         c=method_palette[deconf_method], linewidth=3)

                ax1.set_xticks(parameter_values)
                ax1.set_xticklabels(parameter_values)
                ax1.set_xlabel(parameter_labels[parameter_to_vary])
                ax.set_ylim(ylim)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)
                if method_set_label == 'predictability':
                    ax1.set_ylabel('Bits')
                    ax1.axhline(0.0, c='grey', ls=':')
                else:
                    ax1.set_ylabel('Decoding accuracy')
                    ax1.axhline(0.5, c='grey', ls=':')
                #ax1.legend(frameon=False, loc='lower left')
                #ax2.legend(frameon=False, loc='lower right')
                sns.despine(right=True)
                plt.tight_layout()
                plt.show()

                if save_plots:
                    plot_name = 'methods_inspection_{}_{}_{}_{}.{}'.format(method_set_label, simulation_type, parameter_to_vary, decoder, plot_format)
                    f.savefig(os.path.join(plots_folder, plot_name), dpi=250,
                              bbox_inches="tight")



if plot_lineplot_nsamples:

    for simulation_type in simulation_types:

        for method_set, method_set_label in zip([pred_methods_lp, other_methods_lp],
                                                ['predictability', 'decoding']):

            if method_set_label == 'predictability':
                decoders_to_iterate = ['none']
            else:
                decoders_to_iterate = decoders

            for decoder in decoders_to_iterate:
                f, ax1 = plt.subplots(1, 1, figsize=[2.5, 2.7])
                #ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

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
                    x = []
                    err = []
                    for k, parameter_value in enumerate(parameter_values):

                        decsel = decoder

                        vals = df[(df['simulation_type'] == simulation_type) &
                                  (df['parameter_value'] == parameter_value) &
                                  (df['decoder'] == decsel) &
                                  (df['method'] == deconf_method) &
                                  (df['permuted'] == False)]['mean_score'].__array__()
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

                    ax = ax1

                    ax.plot(parameter_values, x, c=method_palette[deconf_method], zorder=-2)
                    ax.scatter(parameter_values, x, s=30, c=method_palette[deconf_method],
                               edgecolor='w', label=method_labels[deconf_method], zorder=-1)
                    if n_repeats > 1:
                        err = np.vstack(err)
                        ax.fill_between(parameter_values, err[:, 0], err[:, 1], alpha=0.5,
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

                                obs = dfsel[dfsel['permuted'] == False]['mean_score'].__array__()[0]
                                perms = dfsel[dfsel['permuted'] == True]['mean_score'].__array__()
                                p_val = (perms > obs).sum() / (n_permutations + 1)
                                pvals.append(p_val)

                            # this works even if you have just one repeat/pvalue
                            res = combine_pvalues(pvals, method='pearson')

                            dp.loc[dp.shape[0], :] = [parameter_value,
                                                      deconf_method,
                                                      res.pvalue]

                    left_edges = np.array(parameter_values) - np.hstack(([0], np.diff(parameter_values) / 2))
                    right_edges = np.array(parameter_values) + np.hstack((np.diff(parameter_values) / 2, [0]))

                    for j, deconf_method in enumerate(method_set):
                        y = ylim[1] - (j + 1) * 0.03
                        dpsel = dp[dp['method'] == deconf_method]
                        for ii, (rowix, row) in enumerate(dpsel.iterrows()):
                            if row['pvalue'] < alpha_level_permutations_aggregate:
                                ax1.plot([left_edges[ii], right_edges[ii]], [y, y],
                                         c=method_palette[deconf_method], linewidth=3)

                ax1.set_xticks(parameter_values)
                if pars['figure'] == 'figure3_nsamples':
                    ax1.set_xticklabels(parameter_values, rotation=90)
                else:
                    ax1.set_xticklabels(parameter_values)
                ax1.set_xlabel(parameter_labels[parameter_to_vary])
                ax.set_ylim(ylim)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)
                if method_set_label == 'predictability':
                    ax1.set_ylabel('Bits')
                    ax1.axhline(0.0, c='grey', ls=':')
                else:
                    ax1.set_ylabel('Decoding accuracy')
                    ax1.axhline(0.5, c='grey', ls=':')
                #ax1.legend(frameon=False, loc='lower left')
                #ax2.legend(frameon=False, loc='lower right')
                sns.despine(right=True)
                plt.tight_layout()

                if save_plots:
                    plot_name = 'methods_inspection_{}_{}_{}_{}.{}'.format(method_set_label, simulation_type, parameter_to_vary, decoder, plot_format)
                    f.savefig(os.path.join(plots_folder, plot_name), dpi=250,
                              bbox_inches="tight")


# In the violin you can aggregate multiple repeats
# or only plot the first repeat (the one for which we save the data plot)
# combining p-values if n_repeats>1

acc_methods = [m for m in deconf_methods if ~np.isin(m, predictability_methods)]
if pars['figure'] == 'figure2':
    pred_methods = [m for m in deconf_methods if np.isin(m, predictability_methods) and m != 'cmi']
else:
    pred_methods = [m for m in deconf_methods if np.isin(m, predictability_methods) and m != 'cmi']
subsets = [acc_methods, pred_methods]
metrics = ['accuracy', 'predictability']

if plot_only_second_to_last_parameter_for_violins:
    parameter_values_violins = [parameter_values[-2]]
else:
    parameter_values_violins = parameter_values

if plot_violins:

    for simulation_type in simulation_types:

        for k, parameter_value in enumerate(parameter_values_violins):

            for subset, metric in zip(subsets, metrics):
                print(metric)
                if metric == 'accuracy':
                    fig_width = 2.5
                    if simulation_type == 'two_confounds_continuous':
                        fig_width = 2.8
                    decoders_plot = decoders
                elif metric == 'predictability':
                    if pars['figure'] == 'figure2':
                        fig_width = 1.7
                    else:
                        fig_width = 2.2
                    if simulation_type == 'two_confounds_continuous':
                        fig_width = 1.7
                    decoders_plot = ['none']

                for decoder in decoders_plot:

                    f, ax = plt.subplots(1, 1, figsize=[fig_width, 3])

                    df_sel = df[(df['simulation_type'] == simulation_type) &
                                (df['parameter_value'] == parameter_value) &
                                (df['decoder'] == decoder)]
                    perms = df_sel[(df_sel['permuted'] == True)]
                    perms = perms.reset_index()

                    if len(perms) > 1:
                        sns.violinplot(data=perms,
                                       ax=ax, x='method',
                                       y='mean_score', order=subset,
                                       color=sns.xkcd_rgb['light grey'],
                                       inner='quartile', linewidth=1, cut=0)

                    for kk, deconf_method in enumerate(subset):
                        df_sel = df[(df['simulation_type'] == simulation_type) &
                                    (df['parameter_value'] == parameter_value) &
                                    (df['decoder'] == decoder) &
                                    (df['method'] == deconf_method)]
                        #print(kk, deconf_method)

                        obs = df_sel[df_sel['permuted'] == False]
                        perms = df_sel[(df_sel['permuted'] == True)]

                        if use_first_repeat_only_for_the_violins:
                            obs_val = obs[obs['repeat'] == 0]['mean_score'].__array__()[0]
                            perms = perms[perms['repeat'] == 0]
                            p_val = (perms['mean_score'].__array__() > obs_val).sum() / (n_permutations + 1)
                            partial_p = obs['partial_p'].iloc[0]

                        else:
                            if obs.shape[0] > 1:
                                # if I have multiple repeats of the same simulations, the
                                # observed value I plot is the mean, the p-values are aggregated,
                                # and the permutations are all grouped together. The latter may not be ideal.
                                assert n_repeats > 1

                                pvals = []
                                for nrep in range(n_repeats):
                                    obs_rep = obs[obs['repeat'] == nrep]['mean_score'].__array__()[0]
                                    perms_rep = perms[perms['repeat'] == nrep]['mean_score'].__array__()
                                    p_val = (perms_rep > obs_rep).sum() / (n_permutations + 1)
                                    pvals.append(p_val)
                                res = combine_pvalues(pvals, method='pearson')
                                p_val = res.pvalue
                                obs_val = np.mean(obs['mean_score'])
                                partial_p =combine_pvalues(obs['partial_p'], method='pearson')

                            else:
                                # if I have only one repeat of the dataset, no problem
                                obs_val = np.mean(obs['mean_score'])
                                p_val = (perms['mean_score'].__array__() > obs_val).sum() / (n_permutations+1)
                                partial_p = obs['partial_p'].iloc[0]

                        print(p_val, partial_p)

                        if p_val < alpha_level_permutations:
                            ax.scatter(kk, obs_val, marker='o', s=60, c='r', zorder=10,
                                       linewidth=0, edgecolor='r')
                        else:
                            ax.scatter(kk, obs_val, marker='o', s=60, zorder=10,
                                       linewidth=2, edgecolor='r', facecolors='none')

                        if partial_p < alpha_level_partial_test:
                            ax.scatter(kk, 0.97, marker='*', s=40, c='blue', zorder=10)

                    if metric == 'predictability':
                        if pars['figure'] == 'figure2':
                            ylim = [-0.2, 0.2]
                            ax.set_yticks([-0.2, 0, 0.2])
                        else:
                            ylim = [-0.2, 0.35]
                            ax.set_yticks([-0.2, 0, 0.2, 0.4])
                        ax.set_ylim(ylim)
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position("right")
                        ax.axhline(0, ls=':', c='k', zorder=-5)
                        ax.set_ylabel('Bits')
                        sns.despine(right=False, left=True, ax=ax)
                        ax.tick_params(axis='y', right=True)

                    elif metric == 'accuracy':
                        ax.set_ylim([0.25, 0.9])
                        ax.axhline(0.5, ls=':', c='k', zorder=-5)
                        ax.axvline(1.5, c='k', zorder=-5, linewidth=0.75)
                        ax.set_ylabel('Accuracy score')
                        sns.despine(right=True, left=False, ax=ax)
                        ax.set_yticks([0.25, 0.5, 0.75, 1])

                    ax.set_xlabel('')
                    ax.set_xticklabels([method_labels[d] for d in subset], rotation=90)
                    plt.tight_layout()
                    plt.show()

                    if save_plots:
                        plot_name = 'methods_violins_{}_{}_{}_p{}.{}'.format(simulation_type, decoder,metric, k, plot_format)
                        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                                  bbox_inches="tight")



if plot_confound_regression:

    for sim_ix, simulation_type in enumerate(simulation_types):

        for k, parameter_value in enumerate(parameter_values):

            if plot_only_first_repeat:
                data_key = '{}={}'.format(parameter_to_vary, parameter_value)
                X = data[simulation_type][data_key][0]['X']
                y = data[simulation_type][data_key][0]['y']
                C = data[simulation_type][data_key][0]['C']
            else:
                raise ValueError

            kfold = StratifiedKFold(n_splits=pars['n_kfold_splits'], shuffle=True,
                                    random_state=92)

            X_original, X_deconf_lin, X_deconf_nonlin, y_shuf, c_plot = [], [], [], [], []

            for fold, (training_ind, testing_ind) in enumerate(kfold.split(X, y)):
                # decoder = SGDClassifier(random_state=92)
                pass
                X_train = X[training_ind, :]
                X_test = X[testing_ind, :]
                C_train = C[training_ind, :]
                C_test = C[testing_ind, :]
                y_shuf.append(y[testing_ind])

                ss = StandardScaler()
                X_train = ss.fit_transform(X_train)
                X_test = ss.transform(X_test)

                ss = StandardScaler()
                C_train = ss.fit_transform(C_train)
                C_test = ss.transform(C_test)
                c_plot.append(C_test)

                reg_lin = LinearRegression().fit(C_train, X_train)
                X_deconf_lin.append(X_test - reg_lin.predict(C_test))

                reg_nonlin = MultiOutputRegressor(SVR(C=1, kernel='rbf')).fit(C_train, X_train)

                if X.shape[1] == 1:
                    Xdn_fold = X_test - reg_nonlin.predict(C_test).reshape(-1, 1)
                elif X.shape[1] > 1:
                    Xdn_fold = X_test - reg_nonlin.predict(C_test)

                X_deconf_nonlin.append(Xdn_fold)
                X_original.append(X_test)

            X_nodeconf = np.vstack(X_original)
            X_l = np.vstack(X_deconf_lin)
            X_nl = np.vstack(X_deconf_nonlin)
            yplot = np.hstack(y_shuf)
            cplot = np.vstack(c_plot)

            if C.shape[1] > 1:
                fig, ax = plt.subplots(1, 2, figsize=[5, 3], sharex=True, sharey=True)
                for iii, axx in enumerate(ax):
                    axx.scatter(C[y == 0, iii], X_l[y == 0, 0], c='red', label='y=0', s=20, linewidth=0,
                                marker='o')
                    axx.scatter(C[y == 1, iii], X_l[y == 1, 0], c='green', label='y=1', s=20, linewidth=0,
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
                plt.show()

                if save_plots:
                    plot_name = 'data_single_{}_linear_confound_regression_p{}.{}'.format(simulation_type, k, plot_format)
                    fig.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                                bbox_inches="tight")
            else:
                if X.shape[1] == 1:
                    fig, ax = plt.subplots(1, 3, figsize=[9, 3], sharey=True, sharex=True)
                    for iix, (axx, XX) in enumerate(zip(ax, [X_nodeconf, X_l, X_nl])):
                        axx.scatter(cplot[yplot == 0, 0], XX[yplot == 0, 0], c='red', label='y=0', s=20, linewidth=0,
                                    marker='o')
                        axx.scatter(cplot[yplot == 1, 0], XX[yplot == 1, 0], c='green', label='y=1', s=20, linewidth=0,
                                    marker='o')
                        axx.legend(frameon=True, loc='best')
                        axx.axvline(0, ls=':', c='grey')
                        axx.axhline(0, ls=':', c='grey')
                        axx.set_xlabel('$C$')
                        axx.set_ylabel('$X$')
                        axx.legend().remove()
                    #ax[0].axis('equal')
                    ax[0].set_title('Simulated data')
                    ax[1].set_title('Linear CR')
                    ax[2].set_title('Nonlinear CR')
                    sns.despine()
                    plt.tight_layout()
                    plt.show()
                    if save_plots:
                        plot_name = 'confound_regression_{}_p{}.{}'.format(simulation_type, k, plot_format)
                        fig.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                                    bbox_inches="tight")

                elif X.shape[1] > 1:
                    fig, ax = plt.subplots(1, 3, figsize=[9, 3], sharey=True, sharex=True)
                    for iix, (axx, XX) in enumerate(zip(ax, [X_nodeconf, X_l, X_nl])):
                        axx.scatter(XX[yplot == 0, 0], XX[yplot == 0, 1], c='red', label='y=0', s=20, linewidth=0,
                                    marker='o')
                        axx.scatter(XX[yplot == 1, 0], XX[yplot == 1, 1], c='green', label='y=1', s=20, linewidth=0,
                                    marker='o')
                        axx.legend(frameon=True, loc='best')
                        axx.axvline(0, ls=':', c='grey')
                        axx.axhline(0, ls=':', c='grey')
                        axx.set_xlabel('$X_1$')
                        axx.set_ylabel('$X_2$')
                        axx.legend().remove()
                    #ax[0].axis('equal')
                    ax[0].set_title('Simulated data')
                    ax[1].set_title('Linear CR')
                    ax[2].set_title('Nonlinear CR')
                    sns.despine()
                    plt.tight_layout()
                    plt.show()
                    if save_plots:
                        plot_name = 'confound_regression_{}_p{}.{}'.format(simulation_type, k, plot_format)
                        fig.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                                    bbox_inches="tight")



if plot_legends:
    f, ax = plt.subplots(1, 1, figsize=[2, 2])
    legend_elements = [plt.Line2D([0], [0], marker='o', label='$y=0$', lw=0,   markerfacecolor='red', markeredgecolor='w', markersize=8),
                       plt.Line2D([0], [0], marker='o', label='$y=1$', lw=0,   markerfacecolor='green', markeredgecolor='w', markersize=8)]
    ax.legend(handles=legend_elements, loc='center', frameon=False,
              ncol=1)
    sns.despine(left=True, bottom=True)
    ax.axis('off')
    plt.show()
    if save_plots:
        plot_name = 'legend_scatter.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")


    f, ax = plt.subplots(1, 1, figsize=[2, 2])
    legend_elements = [plt.Line2D([0], [0], marker='o', color=method_palette[m],
                                  label=method_labels[m], lw=2.5,
                              markerfacecolor=method_palette[m],
                                  markeredgecolor='w',
                                  markersize=8) for m in pred_methods_lp]
    ax.legend(handles=legend_elements, loc='center', frameon=False,
              ncol=1)
    sns.despine(left=True, bottom=True)
    ax.axis('off')
    plt.show()
    if save_plots:
        plot_name = 'legend_overtime_pred.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")

    f, ax = plt.subplots(1, 1, figsize=[2, 2])
    legend_elements = [plt.Line2D([0], [0], marker='o', color=method_palette[m],
                                  label=method_labels[m], lw=2.5,
                              markerfacecolor=method_palette[m],
                                  markeredgecolor='w',
                                  markersize=8) for m in other_methods_lp]
    ax.legend(handles=legend_elements, loc='center', frameon=False,
              ncol=1)
    sns.despine(left=True, bottom=True)
    ax.axis('off')
    plt.show()
    if save_plots:
        plot_name = 'legend_overtime_dec.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")


