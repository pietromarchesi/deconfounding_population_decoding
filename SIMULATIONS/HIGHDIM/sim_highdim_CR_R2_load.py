import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotting_style import *
from constants import *
import pickle

"""
Load regression scores (R2) for CR applied to high-dimensional simulations
"""


settings_name_global = 'OCT14_R2_FINAL'

modes = ['tune_cr_notune_dec',
         'notune_cr_notune_dec']

dpi = 400
plot_format = 'png'

plots_folder = os.path.join(DATA_PATH, 'plots', 'simulations_HIGHDIM_CR_R2', 'DECODE_SETTINGS_{}'.format(settings_name_global))
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

dfs = []
for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)


    output_folder = os.path.join(DATA_PATH, 'simulations_highdim_CR_R2_results',
                                 'SIMULATION_{}'.format(settings_name_global))
    #file = glob.glob('{}/*'.format(output_folder, settings_name))
    file = os.path.join(output_folder, 'results_r2_{}_{}.pkl'.format(settings_name_global, mode))

    # TODO to load all files from the same global settings:
    #files = glob.glob('{}/*'.format(output_folder, settings_name))

    res = pickle.load(open(file, 'rb'))
    df = res['ds']
    df['mode'] = mode
    for col in ['R2', 'EV', 'MSE', 'parameter_value']:
        df[col] = pd.to_numeric(df[col])

    dfs.append(df)
    print(df[df['method'] == 'nonlinear_confound_regression_svc'].head())

    pars = res['pars']
    simulation_types = res['pars']['simulation_types']
    deconf_methods = res['pars']['deconf_methods']

    simulation_types = np.sort(df['simulation_type'].unique())

    # AVERAGE FOLDS
    # we have one row per fold score, so we first averaged to get cross-validated scores

    excude_cols = ['mean_score', 'fold']
    df = df.groupby([c for c in df.columns if ~np.isin(c, excude_cols)]).mean().reset_index()

    df['dummy'] = 'dummy'

    score_labels = {'R2' : 'R2 score',
              'MSE' : 'Mean square error',
              'EV' : 'Explained variance'}

    for score in ['R2', 'MSE', 'EV']:

        f, ax = plt.subplots(1, len(simulation_types), sharey=True, figsize=[7.5,3])
        if len(simulation_types) == 1:
            ax = np.array([ax])
        for i, simulation_type in enumerate(simulation_types):
            arts = sns.lineplot(data=df[df['simulation_type'] == simulation_type],
                                x='parameter_value',
                                y=score,
                                hue='method',
                                palette=method_palette,
                                ax=ax[i],
                                style='method',
                                markers=cr_markers,
                                markersize=8)
            for line in arts.lines:
                line.set_linestyle('-')

            ax[i].set_title(highdim_simulation_labels[simulation_type], fontsize=10)
            for coll in arts.collections:
                coll.set_linewidth(0)

            x = np.sort(df['parameter_value'].unique())
            ax[i].set_xticks(x)
            ax[i].set_xlabel(pars['parameter_to_vary'])
            handles, labels = ax[i].get_legend_handles_labels()
            ax[i].legend().remove()
            ax[i].set_xlabel('# features')
            ax[i].axhline(0, ls=':', c='grey', zorder=-10)

            #ax[i].legend().remove()
        ax[0].set_ylabel(score_labels[score])


        sns.despine()
        plt.tight_layout()
        plt.show()

        plot_name = '{}_score_{}.{}'.format(score, mode, plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")


    f, ax = plt.subplots(1, 1)
    labels_mod = [cr_labels_detailed[l] for l in labels]
    ax.legend(handles, labels_mod, frameon=False)
    ax.axis('off')
    plt.show()
    plot_name = 'R2_score_legend.{}'.format(plot_format)
    f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
              bbox_inches="tight")


import copy
f, ax = plt.subplots(1, 1)
labels_mod = [cr_labels_detailed[l] for l in labels]
handles1 = handles[1:]
labels_mod = labels_mod[1:]
handles2 = copy.deepcopy(handles1)
for h in handles2:
    h.set_linestyle(':')
labels_mod2 = ['{}, Tuned HP'.format(m) for m in labels_mod]
ax.legend(np.hstack([handles1, handles2]), np.hstack([labels_mod, labels_mod2]), frameon=False)
ax.axis('off')
plt.show()
plot_name = 'R2_score_tuned_legend.{}'.format(plot_format)
f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
          bbox_inches="tight")



dz = pd.concat(dfs)
deconf_methods_tuning = [d for d in deconf_methods if d != 'linear_confound_regression']

for score in ['R2', 'EV']:

    f, ax = plt.subplots(1, len(simulation_types), sharey=True, figsize=[7.5, 3])

    for i, simulation_type in enumerate(simulation_types):
        if len(simulation_types) == 1:
            ax = np.array([ax])
        for k, deconf_method in enumerate(deconf_methods_tuning):
            dashes = {modes[0] : 'dotted', modes[1] : 'solid'}
            for j, mode in enumerate(modes):
                arts = sns.lineplot(data=dz[(dz['method'] == deconf_method) &
                                            (dz['simulation_type'] == simulation_type) &
                                            (dz['mode'] == mode)],
                                    x='parameter_value',
                                    y=score,
                                    color=method_palette[deconf_method],
                                    ax=ax[i],
                                    marker=cr_markers[deconf_method],
                                    linestyle=dashes[mode],
                                    markersize=8)
                handles, labels = ax[i].get_legend_handles_labels()

                ax[i].set_title(highdim_simulation_labels[simulation_type], fontsize=10)
                for coll in arts.collections:
                    coll.set_linewidth(0)
                ax[i].axhline(0, ls=':', c='grey', zorder=-10)

            x = np.sort(df['parameter_value'].unique())
            ax[i].set_xticks(x)
            #ax[i].set_xlabel(pars['parameter_to_vary'])
            ax[i].set_xlabel('# features')

            #ax[i].legend().remove()

            #ax[i].legend().remove()
        ax[0].set_ylabel(score_labels[score])

    sns.despine()
    plt.tight_layout()
    plt.show()

    plot_name = '{}_score_by_mode.{}'.format(score, plot_format)
    f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
              bbox_inches="tight")