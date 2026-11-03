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
from baycomp import two_on_multiple
import scipy
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

"""
Load PR results on high-dimensional simulations obtained with both true and shuffled confound
"""

# SUPPLEMENTARY FIG. 7 (200 samples)
settings_name_global_normal = 'OCT2_NFEAT_TEST_MD4'
settings_name_global_shuffled = 'OCT2_NFEAT_TEST_MD4_SHUFFLE'

# MAIN FIG. 4 (400 samples)
settings_name_global_normal = 'OCT2_NFEAT_TEST_MD4_MORESAMP'
settings_name_global_shuffled = 'OCT2_NFEAT_TEST_MD4_MORESAMP_SHUFFLE'


modes = ['nocalib_notune',
         'nocalib_tune',
         'calib_sigmoid',
         'calib_isotonic',
         'calib_vennabers']

# plot options
dpi = 400
plot_format = 'png'
save_plots = True
alpha_level_tests = 0.05
alpha_level_permutations = 0.05

plots_folder_general = os.path.join(DATA_PATH, 'plots', 'simulations_HIGHDIM_CAL',
                                        'DECODE_SETTINGS_{}'.format(settings_name_global_normal))

if not os.path.isdir(plots_folder_general):
    os.makedirs(plots_folder_general)


dms = []
setting_names = [settings_name_global_normal, settings_name_global_shuffled]
for shuffled_confound, settings_name_global in zip([False, True], setting_names):

    df_all = []

    for mode in modes:
        settings_name = '{}_{}'.format(settings_name_global, mode)


        dfs = []
        pars_list = []
        data = {}
        output_folder = os.path.join(DATA_PATH, 'simulations_highdim_PR_results',
                                     'SIMULATION_{}'.format(settings_name_global))
        files = glob.glob('{}/results_{}*'.format(output_folder, settings_name))

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
        pars = pars_list[0]
        print(pars)

        simulation_types = np.sort(df['simulation_type'].unique())
        pars = pars_list[0]
        deconf_methods = pars['deconf_methods']

        n_repeats = pars['n_repeats']
        n_permutations = pars['n_permutations']
        parameter_to_vary = pars['parameter_to_vary']
        parameter_values = np.sort(pars['parameter_values'])

        pred_methods_lp = deconf_methods


    dm = pd.concat(df_all)

    dmp = pd.DataFrame(columns=['mode', 'simulation_type', 'decoder', 'deconf_method', 'shuffled_confound',
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

                            # this works even if you have just one repeat/pvalue
                            if n_repeats > 1:
                                res = combine_pvalues(pvals, method='pearson')
                                pvalue = res.pvalue
                            else:
                                pvalue = pvals[0]

                            dmp.loc[dmp.shape[0], :] = [mode, simulation_type, decoder,
                                                      deconf_method, shuffled_confound, parameter_value,
                                                      pvalue]
    dms.append(dmp)

dz = pd.concat(dms)

dl = pd.DataFrame(columns=['simulation_type', 'deconf_method', 'mode', 'TPR_TNR_av', 'TPR', 'TNR', 'FPR',
                           'balanced_acc', 'f1_score'])
for i, simulation_type in enumerate(simulation_types):
    for j, deconf_method in enumerate(method_set):
        for mode in modes:
            dzel = dz[(dz['simulation_type'] == simulation_type) &
                          (dz['deconf_method'] == deconf_method) &
                          (dz['mode'] == mode)]
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

            dl.loc[dl.shape[0], :] = [simulation_type, deconf_method, mode, TPR_TNR, TPR, TNR, FPR, balanced_acc, f1]


supplementary_methods = ['predictability_nonlin_svc']

main_methods = ['predictability_lin',
                   'predictability_nonlin_rf']
labels = ['main', 'supplementary']

for method_set, set_label in zip([main_methods, supplementary_methods], labels):

    for score in ['TPR_TNR_av', 'FPR', 'TNR', 'TPR']:
        figsize = [3.25*len(method_set), 3.5]
        if len(method_set) == 1:
            figsize = [3.5, 3.5]

        f, ax = plt.subplots(1, len(method_set), sharey=True,
                             sharex=False, figsize=figsize)
        if len(method_set) == 1:
            ax = [ax]

        for i, deconf_method in enumerate(method_set):

            sns.barplot(data=dl[dl['deconf_method'] == deconf_method],
                        x='simulation_type', order=simulation_types,
                        hue='mode', hue_order=modes, y=score, errorbar=None, dodge=True,
                        color=method_palette[deconf_method], ax=ax[i], gap=0.15)
            ax[i].set_xlabel('')
            xlim = ax[i].get_xlim()
            #ax[i].set_xticklabels([highdim_simulation_labels[s] for s in simulation_types], rotation=90)
            ax[i].set_title(pr_labels_detailed[deconf_method], fontsize=10)
            if i == 0:
                handles_lin, labels_lin = ax[i].get_legend_handles_labels()
            if i == 1:
                handles_nonlin, labels_nonlin = ax[i].get_legend_handles_labels()
            ax[i].legend().remove()
            x_bars = []
            for patch in ax[i].patches:
                x_bars.append(patch.get_x() + patch._width / 2)
            x_bars = np.array(x_bars)[0: len(modes) * len(simulation_types)]
            ax[i].set_xticks(x_bars)
            ax[i].set_xticklabels([calibration_modes_labels[d] for d in np.repeat(modes, len(simulation_types))], rotation=90)
            # ax.set_xticks(x_bars)
            central_bars_x = x_bars[len(modes)+1:2*len(modes)+1]
            sim_labels = [highdim_simulation_labels[s] for s in simulation_types]
            for x, lab in zip(central_bars_x, sim_labels):
                ax[i].text(x, 1.17, lab, ha='center', va='top');
        for axx in ax:
            axx.set_xlim(xlim)
            axx.set_ylim([0, 1.18])
            # Move the legend to ax2
        ax[0].set_ylabel(score_labels[score])
        plt.show()
        sns.despine()
        plt.tight_layout()

        if save_plots:
            plot_name = '{}_barplot_singlemodes_by_simulation_{}_{}.{}'.format(set_label, settings_name_global_normal, score, plot_format)
            f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                      bbox_inches="tight")





