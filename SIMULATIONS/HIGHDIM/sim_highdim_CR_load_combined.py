import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotting_style import *
from constants import *
import pickle
from scipy.stats import combine_pvalues
import glob
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

"""
Load CR results on high-dimensional simulations obtained with both true and shuffled confound
"""

# SUPPLEMENTARY FIG. 7 (200 samples)
settings_name_global_normal = 'OCT2_NFEAT_TEST'
settings_name_global_shuffled = 'OCT2_NFEAT_TEST_SHUFFLE'
mode = 'notune_cr_notune_dec'

# MAIN FIG. 4 (400 samples)
settings_name_global_normal = 'OCT2_NFEAT_TEST_MORESAMP'
settings_name_global_shuffled = 'OCT2_NFEAT_TEST_MORESAMP_SHUFFLE'
mode = 'notune_cr_notune_dec'

# plot options
dpi = 400
plot_format = 'png'
save_plots = True
alpha_level_permutations = 0.05

exclude_methods = ['nonlinear_confound_regression_svc']


dms = []
setting_names = [settings_name_global_normal, settings_name_global_shuffled]
for shuffled_confound, settings_name_global in zip([False, True], setting_names):

    settings_name = '{}_{}'.format(settings_name_global, mode)

    plots_folder = os.path.join(DATA_PATH, 'plots', 'simulations_HIGHDIM_CR', 'DECODE_SETTINGS_{}_{}'.format(settings_name, mode))
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)

    dfs = []
    pars_list = []
    data = {}
    output_folder = os.path.join(DATA_PATH, 'simulations_highdim_CR_results',
                                 'SIMULATION_{}'.format(settings_name_global))
    files = glob.glob('{}/results_{}*'.format(output_folder, settings_name))


    for file in files:
        res = pickle.load(open(file, 'rb'))
        df = res['df']
        print(df.shape)
        assert df['simulation_type'].unique().shape[0] == 1
        dfs.append(df)
        pars_list.append(res['pars'])
        simulation_types = res['pars']['simulation_types']
    df = pd.concat(dfs)
    df['mode'] = mode

    pars = pars_list[0]
    decoders = pars['decoders']
    simulation_types = np.sort(df['simulation_type'].unique())
    deconf_methods = pars['deconf_methods']
    print(deconf_methods)

    deconf_methods = [d for d in deconf_methods if ~np.isin(d, exclude_methods)]

    n_repeats = pars['n_repeats']
    n_permutations = pars['n_permutations']
    parameter_to_vary = pars['parameter_to_vary']
    parameter_values = np.sort(pars['parameter_values'])

    deconf_methods_plot = [d for d in deconf_methods if d != 'raw']

    # AVERAGE FOLDS
    # we have one row per fold score, so we first averaged to get cross-validated scores
    df['mean_score'] = pd.to_numeric(df['mean_score'])
    excude_cols = ['mean_score', 'fold']
    df = df.groupby([c for c in df.columns if ~np.isin(c, excude_cols)]).mean().reset_index()


    dfp = pd.DataFrame(columns=['mode', 'simulation_type', 'decoder', 'deconf_method',
                                 'shuffled_confound', 'parameter_value', 'pvalue'])

    for simulation_type in simulation_types:
        for decoder in decoders:
            for j, deconf_method in enumerate(deconf_methods):
                for k, parameter_value in enumerate(parameter_values):
                    pvals = []
                    for nrep in range(n_repeats):

                        dmsel = df[(df['mode'] == mode) &
                                   (df['simulation_type'] == simulation_type) &
                                   (df['parameter_value'] == parameter_value) &
                                   (df['decoder'] == decoder) &
                                   (df['method'] == deconf_method) &
                                   (df['repeat'] == nrep)]

                        obs = dmsel[dmsel['permuted'] == False]['mean_score'].__array__()
                        assert obs.shape[0] == 1
                        obs = obs[0]
                        perms = dmsel[dmsel['permuted'] == True]['mean_score'].__array__()
                        p_val = (perms > obs).sum() / (n_permutations + 1)
                        pvals.append(p_val)

                    # this works even if you have just one repeat/pvalue
                    if n_repeats > 1:
                        res = combine_pvalues(pvals, method='pearson')
                        pvalue = res.pvalue
                    else:
                        pvalue = pvals[0]

                    dfp.loc[dfp.shape[0], :] = [mode, simulation_type, decoder,
                                                deconf_method, shuffled_confound,
                                                parameter_value, pvalue]
    dms.append(dfp)
dz = pd.concat(dms)

plots_folder_general = os.path.join(DATA_PATH, 'plots', 'simulations_HIGHDIM_CR', 'DECODE_SETTINGS_{}'.format(settings_name_global_normal))
if not os.path.isdir(plots_folder_general):
    os.makedirs(plots_folder_general)

dl = pd.DataFrame(columns=['simulation_type', 'deconf_method', 'mode', 'decoder', 'TPR_TNR_av', 'TPR', 'TNR', 'FPR',
                           'balanced_acc', 'f1_score'])
for i, simulation_type in enumerate(simulation_types):
    for j, deconf_method in enumerate(deconf_methods):
        for k, decoder in enumerate(decoders):
            dzel = dz[(dz['simulation_type'] == simulation_type) &
                          (dz['deconf_method'] == deconf_method) &
                          (dz['mode'] == mode) &
                          (dz['decoder'] == decoder)]
            dzp = dzel[dzel['shuffled_confound'] == True]
            dzn = dzel[dzel['shuffled_confound'] == False]

            sig_p = (dzp['pvalue'] < alpha_level_permutations).astype(int).__array__()
            sig_n = (dzn['pvalue'] < alpha_level_permutations).astype(int).__array__()

            y_pred = np.hstack([sig_p, sig_n])
            y_true = np.hstack([np.ones_like(sig_p), np.zeros_like(sig_n)])

            true_pos = np.sum(sig_p)
            false_pos = np.sum(sig_n)

            n_pos = len(sig_p)

            true_neg = np.where(sig_n == 0)[0].shape[0]
            n_neg = len(sig_n)

            TPR = true_pos / n_pos
            TNR = true_neg / n_neg
            TPR_TNR = (TPR + TNR) / 2
            FPR = false_pos / n_neg

            balanced_acc = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
            f1 = f1_score(y_pred=y_pred, y_true=y_true)

            dl.loc[dl.shape[0], :] = [simulation_type, deconf_method, mode, decoder, TPR_TNR, TPR, TNR, FPR, balanced_acc, f1]


# BARPLOTS
for score in ['TPR_TNR_av', 'FPR', 'TNR', 'TPR']:
    f, ax = plt.subplots(1, len(deconf_methods_plot), sharey=True,
                         sharex=False, figsize=[6.5,3.5])
    for i, deconf_method in enumerate(deconf_methods_plot):

        arts = sns.barplot(data=dl[dl['deconf_method'] == deconf_method],
                    x='simulation_type', order=simulation_types,
                    hue='decoder', y=score, errorbar=None, dodge=True,
                    color=method_palette[deconf_method], ax=ax[i], gap=0.15)
        xlim = ax[i].get_xlim()
        ax[i].set_xlabel('')
        #ax[i].set_xticklabels([highdim_simulation_labels[s] for s in simulation_types], rotation=90)
        ax[i].set_title(cr_labels_detailed[deconf_method], fontsize=11)
        if i == 0:
            handles_lin, labels_lin = ax[i].get_legend_handles_labels()
        if i == 1:
            handles_nonlin, labels_nonlin = ax[i].get_legend_handles_labels()
        ax[i].legend().remove()

        x_bars = []
        for patch in ax[i].patches:
            x_bars.append(patch.get_x()+patch._width/2)
        x_bars = np.array(x_bars)[0: len(decoders)*len(simulation_types)]
        ax[i].set_xticks(x_bars)
        ax[i].set_xticklabels([decoder_labels[d] for d in np.repeat(decoders, len(simulation_types))], rotation=90)
        #ax.set_xticks(x_bars)
        central_bars_x = x_bars[3:6]
        sim_labels = [highdim_simulation_labels[s] for s in simulation_types]
        for x, lab in zip(central_bars_x, sim_labels):
            ax[i].text(x, 1.17, lab, ha='center', va='top');
    for axx in ax:
        axx.set_xlim(xlim)
        axx.set_ylim([0, 1.18])
    ax[0].set_ylabel(score_labels[score])
    sns.despine()
    plt.tight_layout()
    plt.show()


    if save_plots:
        plot_name = 'barplot_singlemodes_by_simulation_{}_{}.{}'.format(settings_name_global_normal, score, plot_format)
        f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                  bbox_inches="tight")

f, ax = plt.subplots(1,2)
ax[0].legend(handles_lin, [decoder_labels[l] for l in labels_lin], frameon=False)
ax[1].legend(handles_nonlin, [decoder_labels[l] for l in labels_nonlin], frameon=False)
ax[0].axis('off')
ax[1].axis('off')
plt.show()
if save_plots:
    plot_name = 'legend_modes_{}.{}'.format(settings_name_global_normal, plot_format)
    f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
              bbox_inches="tight")
