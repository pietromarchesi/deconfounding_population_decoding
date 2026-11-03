import os
import pandas as pd
from constants import DATA_PATH
import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
from plotting_style  import *
from scipy.stats import wilcoxon, mannwhitneyu

"""
Loading results for CR and PR applied to V1 data, and plot together.
"""

# 0-400 ms time bin (supplementary fig. 8)
PR_settings_name_global = 'OCT9_NEWDATA_0_400_ALLCOMP'
CR_settings_name_global = 'OCT9_NEWDATA_0_400_ALLCOMP'

# 100-500 ms time bin
PR_settings_name_global = 'OCT9_NEWDATA'
CR_settings_name_global = 'OCT9_NEWDATA'


CR_mode = 'all_sessions_significance_n_components_notune_cr_notune_dec'

deconf_methods_plot = ['linear_confound_regression',
                       'nonlinear_confound_regression_rf',
                       'predictability_lin',
                       'predictability_nonlin_rf']

save_plots = True
alpha_level_permutations = 0.05
alpha_level = 0.05
plot_format = 'png'
n_components_main = 30

plots_folder = os.path.join(DATA_PATH, 'plots', 'EPHYS_PLOTS', 'ephys_data_CR_PR_combined',
                            '{}_{}_{}'.format(PR_settings_name_global, CR_settings_name_global, CR_mode))
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

output_folder = os.path.join(DATA_PATH, 'ephys_data_CR_PR_combined')
PR_file = os.path.join(output_folder, 'PR_dfscore_{}.pkl'.format(PR_settings_name_global))
CR_file = os.path.join(output_folder, 'CR_dfscore_{}_{}.pkl'.format(CR_settings_name_global, CR_mode))

PR_df = pickle.load(open(PR_file, 'rb'))
CR_df = pickle.load(open(CR_file, 'rb'))

PR_df['new_mode'] = ['calibration_{}'.format(d) for d in PR_df['calibration_method']]
CR_df['new_mode'] = ['decoder_{}'.format(d) for d in CR_df['decoder']]
PR_df = PR_df.drop(['mode','calibration_method'], axis=1)
CR_df = CR_df.drop(['decoder'], axis=1)
print(PR_df.columns)

dl = pd.concat([PR_df, CR_df]).reset_index(drop=True)


for col in ['n_components', 'TPR', 'FPR', 'TPR_TNR_av']:
    dl[col] = pd.to_numeric(dl[col])

dl30 = dl[dl['n_components'] == n_components_main]

# PLOT SCORE AVERAGED ACROSS DECODERS MODES BY SIMULATION TYPE
for score in ['TPR_TNR_av', 'FPR', 'TPR']:
    f, ax = plt.subplots(1, 1, figsize=[2,3])
    sns.barplot(data=dl30, x='deconf_method', order=deconf_methods_plot,
                y=score, palette=method_palette, ax=ax)
    ax.set_xlabel('')
    ax.set_ylim([0, 1.15])
    ax.set_xticklabels([method_labels[d] for d in deconf_methods_plot], rotation=90)
    #dl['new_mode'] = dl['new_mode'].astype(str)

    gap = 0
    from math import comb
    n_comparisons = comb(len(deconf_methods_plot), 2)
    for ii, method1 in enumerate(deconf_methods_plot):
        for jj, method2 in enumerate(deconf_methods_plot):
            if method1 != method2 and jj > ii:
                m1 = dl30[(dl30['deconf_method'] == method1)].sort_values(by='new_mode')[score].values.astype(float)

                m2 = dl30[(dl30['deconf_method'] == method2)].sort_values(by='new_mode')[score].values.astype(float)

                if 'regression' in method1 and 'regression' in method2:
                    same_family = True
                elif 'predictability' in method1 and 'predictability' in method2:
                    same_family = True
                else:
                    same_family = False

                if same_family:
                    stat, p = wilcoxon(m1, m2)
                else:
                    stat, p = mannwhitneyu(m1, m2)

                # (p_left, p_rope, p_right) = two_on_multiple(x=m1, y=m2, runs=1)
                # is_sig = p_right >= (1 - alpha_level_tests)
                is_sig = p <= (alpha_level_permutations / n_comparisons)
                print(method1, method2, p)
                if is_sig:
                    ax.hlines(y=1.02+gap, xmin=ii, xmax=jj, color='k', lw=1)
                    gap = gap + 0.025

    ax.set_ylabel(score_labels[score])
    plt.show()
    sns.despine()
    plt.tight_layout()

    if save_plots:
        plot_name = 'barplot_by_simulation_{}_{}_{}.{}'.format(CR_settings_name_global, PR_settings_name_global, score, plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=400,
                  bbox_inches="tight")


# LINEPLOT
for score in ['TPR_TNR_av', 'FPR', 'TPR']:
    f, ax = plt.subplots(1, 1, figsize=[2.5, 3])
    dl['dummy'] = 'dummy'
    arts = sns.lineplot(data=dl, x='n_components', y=score,
                 hue='deconf_method', hue_order=deconf_methods_plot,
                 palette=method_palette, ax=ax, style='dummy',
                 markers='o')
    for coll in arts.collections:
        coll.set_linewidth(0)
    x = np.sort(dl['n_components'].unique())
    ax.set_ylim([0, 1.01])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(x)
    ax.set_xlim([x[0] - 3, x[-1] + 3])
    ax.set_xlabel('# movement PCs')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    ax.set_ylabel(score_labels[score])
    sns.despine()
    plt.tight_layout()
    plt.show()
    plot_name = '{}_vs_n_components_averaged.{}'.format(score, plot_format)
    f.savefig(os.path.join(plots_folder, plot_name), dpi=400,
              bbox_inches="tight")



f, ax = plt.subplots(1, 1)
labels_mod = [method_labels[l] for l in labels if ~np.isin(l, ['deconf_method', 'dummy'])]
ax.legend(handles[1:], labels_mod, frameon=False)
ax.axis('off')
plt.show()
plot_name = 'lineplot_legend.{}'.format(plot_format)
f.savefig(os.path.join(plots_folder, plot_name), dpi=400,
          bbox_inches="tight")

