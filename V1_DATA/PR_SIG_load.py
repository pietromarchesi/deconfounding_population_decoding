import os
import glob
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from constants import DATA_PATH
import pickle
from session_selection import animal_id_per_session
from plotting_style  import *
from baycomp import two_on_single, two_on_multiple
from sklearn.metrics import f1_score, precision_score, recall_score
import sklearn as skl

"""
Loading results and generating figures for PR applied to V1 data.
"""

# Results for the 100-500 ms time bin (main)
settings_name_global = 'OCT9_NEWDATA'


# Results in 0-400 ms time bin (supplementary fig. 8)
settings_name_global = 'OCT9_NEWDATA_0_400_ALLCOMP'


# select modes to be plotted
modes = ['all_sessions_significance_nocalib_notune',
         'all_sessions_significance_nocalib_tune',
         'all_sessions_significance_calib_sigmoid',
         'all_sessions_significance_calib_isotonic',
         'all_sessions_significance_calib_vennabers']

# all available modes
all_modes = ['all_sessions_significance_nocalib_notune',
             'all_sessions_significance_nocalib_tune',
             'all_sessions_significance_calib_sigmoid',
             'all_sessions_significance_calib_isotonic',
             'all_sessions_significance_calib_vennabers']

# order for scatter plot
modes_order_scatter = ['all_sessions_significance_nocalib_notune',
                     'all_sessions_significance_calib_sigmoid',
                     'all_sessions_significance_calib_isotonic',
                     'all_sessions_significance_calib_vennabers',
                       'all_sessions_significance_nocalib_tune']

dpi = 400
plot_format = 'png'

if settings_name_global == 'OCT9_NEWDATA':
    plot_dots = True
    figure_modes = ['MAIN', 'SUPPLEMENTARY']
    deconf_methods_main = ['predictability_lin',
                               'predictability_nonlin_rf']
    deconf_methods_supplementary = ['predictability_nonlin_gb']
    deconf_methods_supplementary_comparison = ['predictability_nonlin_rf',
                                               'predictability_nonlin_gb']
    deconf_methods_supplementary_dots = ['predictability_lin',
                                         'predictability_nonlin_rf',
                                         'predictability_nonlin_gb']

elif settings_name_global == 'OCT9_NEWDATA_0_400_ALLCOMP':
    plot_dots = False
    figure_modes = ['MAIN']
    deconf_methods_main = ['predictability_lin',
                               'predictability_nonlin_rf']


save_dfscore = True
alpha_level_permutations = 0.05
alpha_level = 0.05
n_components_mainfig = 30


plots_folder_general = os.path.join(DATA_PATH, 'plots', 'EPHYS_PLOTS', 'PR_CAL', settings_name_global)
if not os.path.isdir(plots_folder_general):
    os.makedirs(plots_folder_general)

df_scores = []
for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)

    plots_folder = os.path.join(DATA_PATH, 'plots', 'EPHYS_PLOTS', 'PR_CAL', settings_name)
    if not os.path.isdir(plots_folder):
        os.makedirs(plots_folder)

    # LOAD DATA
    output_folder = os.path.join(DATA_PATH, 'ephys_data_PR_results',
                                 'SIMULATION_{}'.format(settings_name))
    files = glob.glob('{}/*'.format(output_folder))

    dfs, pars = [], []
    for file in files:
        res = pickle.load(open(file, 'rb'))
        dfs.append(res['df'])
        pars.append(res['pars'])
    pars = pars[0]
    df = pd.concat(dfs)

    mode = pars['mode']
    calibration_method = pars['calibration_method']

    n_permutations = pars['n_permutations']
    time_bin_indexes = pars['time_bin_indexes']
    if len(time_bin_indexes) == 1:
        time_bin_index_dots = time_bin_indexes[0]
        response_time_bin_index = time_bin_indexes[0]
    else:
        raise NotImplementedError
    time_bin_times = pars['time_bin_times']
    time_bins = pars['time_bins']
    shuffle_components_list = pars['shuffle_components_list']
    session_ids = pars['session_ids']
    deconf_methods = np.sort(df['deconf_method'].unique())

    n_pca_components = pars['n_pca_components']
    try:
        n_pca_components[0]
    except TypeError:
        n_pca_components = [n_pca_components]

    # folds are averaged in computation loop, so no need to groupby here
    num_col = ['predictability', 'accuracy_full', 'accuracy_naive',
               'time', 't0', 't1', 'time_bin']
    for col in num_col:
        df[col] = pd.to_numeric(df[col])
    df = df.drop(['decoder'], axis=1)
    df['animal_id'] = [animal_id_per_session[s] for s in df['session_id'].values]


    # MAKE SIGNIFICANCE DATAFRAMES
    if np.isin(mode, all_modes):

        dps = []
        dss = []
        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for shuffle_components in shuffle_components_list:
                    rs = df[df['shuffle_confound'] == shuffle_components]

                    dp = pd.DataFrame(columns=['session_id',
                                               'animal_id',
                                               'deconf_method',
                                               'n_components',
                                               'time_bin_index',
                                               'time',
                                               'pvalue',
                                               'mean_score',
                                               'mean_permuted_score',
                                               'is_sig'])

                    for session_id in session_ids:
                        for kk, deconf_method in enumerate(deconf_methods):
                            df_sel = rs[(rs['session_id'] == session_id) &
                                        (rs['deconf_method'] == deconf_method) &
                                        (rs['time_bin'] == time_bin_index) &
                                        (rs['n_pca_components'] == n_components)]

                            obs = df_sel[df_sel['permuted'] == False]
                            obs_val = obs['predictability'].iloc[0]
                            animal_id = obs['animal_id'].iloc[0]


                            if n_permutations > 0:
                                perms = df_sel[(df_sel['permuted'] == True)]
                                perms_val = perms['predictability'].__array__()
                                p_val = (perms_val > obs_val).sum() / (n_permutations + 1)
                                is_sig = p_val <= alpha_level_permutations
                                perms_mean_val = perms_val.mean()
                            else:
                                perms_mean_val = np.nan
                                p_val = np.nan
                                is_sig = False

                            dp.loc[dp.shape[0], :] = [session_id, animal_id, deconf_method, n_components,
                                                      time_bin_index, time,
                                                      p_val, obs_val, perms_mean_val, is_sig]

                    dp['shuffledC'] = shuffle_components
                    dps.append(dp)


        dp = pd.concat(dps)
        dp['mean_score'] = pd.to_numeric(dp['mean_score'])

        ds = pd.DataFrame(columns=['deconf_method',
                                   'n_components',
                                   'shuffledC',
                                   'time_bin_index',
                                   'time',
                                   'perc_sig',
                                   'n_sig'])
        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for kk, deconf_method in enumerate(deconf_methods):
                    for shuffle_components in [True, False]:
                        dpsel = dp[(dp['deconf_method'] == deconf_method) &
                                   (dp['n_components'] == n_components) &
                                   (dp['shuffledC'] == shuffle_components) &
                                   (dp['time_bin_index'] == time_bin_index)]
                        n_sess = dpsel['session_id'].unique().shape[0]
                        n_sig = dpsel['is_sig'].astype(bool).sum()
                        perc_sig = 100 * n_sig / n_sess
                        ds.loc[ds.shape[0], :] = [deconf_method, n_components, shuffle_components,
                                                  time_bin_index, time, perc_sig, n_sig]
            ds['perc_sig'] = pd.to_numeric(ds['perc_sig'])

    # MAKE F1-SCORE DATAFRAME
    if np.isin(mode, all_modes):

        dfscore = pd.DataFrame(columns=['mode',
                                        'deconf_method',
                                        'n_components',
                                        'time_bin_index',
                                        'time',
                                        'f1_score',
                                        'precision',
                                        'recall',
                                        'TPR',
                                        'TNR',
                                        'TPR_TNR_av',
                                        'FPR',
                                        'bal_accuracy'])

        for n_components in pars['n_pca_components']:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for kk, deconf_method in enumerate(deconf_methods):
                    dpsel1 = dp[(dp['deconf_method'] == deconf_method) &
                                (dp['n_components'] == n_components) &
                               (dp['shuffledC'] == True) &
                               (dp['time_bin_index'] == time_bin_index)]
                    n_sess = dpsel1['session_id'].unique().shape[0]
                    # if C is shuffled, significant sessions are true positives
                    tp = dpsel1['is_sig'].astype(bool).sum()
                    fn = n_sess - tp
                    ypred1 = dpsel1['is_sig'].astype(int).__array__()
                    ytest1 = np.ones_like(ypred1)
                    TPR = tp / n_sess

                    dpsel2 = dp[(dp['deconf_method'] == deconf_method) &
                                (dp['n_components'] == n_components) &
                                (dp['shuffledC'] == False) &
                               (dp['time_bin_index'] == time_bin_index)]
                    n_sess = dpsel2['session_id'].unique().shape[0]
                    # if C is shuffled, significant sessions are true positives
                    fp = dpsel2['is_sig'].astype(bool).sum()
                    tn = np.where(dpsel2['is_sig'].astype(bool)==False)[0].shape[0]
                    ypred2 = dpsel2['is_sig'].astype(int).__array__()
                    ytest2 = np.zeros_like(ypred2)
                    TNR = tn / n_sess
                    FPR = fp / n_sess

                    ytrue = np.hstack([ytest1, ytest2])
                    ypred = np.hstack([ypred1, ypred2])
                    f1 = f1_score(y_true=ytrue, y_pred=ypred)
                    precision = precision_score(y_true=ytrue, y_pred=ypred)
                    recall = recall_score(y_true=ytrue, y_pred=ypred)
                    balacc = balanced_accuracy_score(y_true=ytrue, y_pred=ypred)
                    print(deconf_method, tp, fn, fp)
                    f1_manual = 2*tp / (2*tp+fp+fn)
                    assert f1-f1_manual < 1e-8
                    TPR_TNR = (TPR+TNR) / 2

                    dfscore.loc[dfscore.shape[0], :] = [mode, deconf_method, n_components,
                                                        time_bin_index, time, f1, precision, recall,
                                                        TPR, TNR, TPR_TNR, FPR, balacc]

    numeric_cols = ['f1_score', 'precision', 'recall', 'TPR', 'TNR', 'TPR_TNR_av', 'bal_accuracy', 'FPR']
    for col in numeric_cols:
        dfscore[col] = pd.to_numeric(dfscore[col])
    if calibration_method is None:
        if pars['tune_predictability']:
            dfscore['calibration_method'] = 'no_calib_tune'
        else:
            dfscore['calibration_method'] = 'no_calib_no_tune'
    else:
        dfscore['calibration_method'] = calibration_method
    dfscore = dfscore.round(4)
    df_scores.append(dfscore)


    # PLOT DOTS
    if plot_dots:
        time = time_bin_times[time_bin_index_dots]
        ytop = 0.6
        ylims2 = [-0.2, ytop]
        # yticks2 = [0.5, 0.6, 0.7, 0.8]

        f, ax = plt.subplots(1, len(deconf_methods_supplementary_dots), figsize=[4.2, 3.2], sharey=True)

        ytops, siglines = [], []
        for i, deconf_method in enumerate(deconf_methods_supplementary_dots):

            dfsel = dp[(dp['deconf_method'] == deconf_method) &
                       (dp['time'] == time) &
                       (dp['n_components'] == n_components_mainfig)].copy()

            n_sessions = dfsel['session_id'].unique().shape[0]

            y0 = dfsel[dfsel['shuffledC'] == False]['mean_score'].__array__(float)
            y1 = dfsel[dfsel['shuffledC'] == True]['mean_score'].__array__(float)
            y0_sig = dfsel[dfsel['shuffledC'] == False]['is_sig'].__array__()
            y1_sig = dfsel[dfsel['shuffledC'] == True]['is_sig'].__array__()

            if len(y0) == 0:
                continue

            ytops.append(np.hstack([y0, y1]).max())

            # for the statistics, treat animal as unit of inference
            if skl.__version__ == '1.7.0':
                y0_animal = dfsel[dfsel['shuffledC'] == False].groupby('animal_id').mean(numeric_only=True).reset_index()['mean_score'].__array__(float)
                y1_animal = dfsel[dfsel['shuffledC'] == True].groupby('animal_id').mean(numeric_only=True).reset_index()['mean_score'].__array__(float)

            else:
                y0_animal = dfsel[dfsel['shuffledC'] == False].groupby('animal_id').mean().reset_index()['mean_score'].__array__(float)
                y1_animal = dfsel[dfsel['shuffledC'] == True].groupby('animal_id').mean().reset_index()['mean_score'].__array__(float)

            x0 = [0 + np.random.uniform(-0.05, 0.05) for i in range(y0.shape[0])]
            x1 = [1 + np.random.normal(-0.05, 0.05) for i in range(y1.shape[0])]

            facecolor_y0 = [method_palette[deconf_method] if a else 'w' for a in y0_sig]
            facecolor_y1 = [method_palette[deconf_method] if a else 'w' for a in y1_sig]
            edgecolors_y0 = ['w' if a else method_palette[deconf_method] for a in y0_sig]
            edgecolors_y1 = ['w' if a else method_palette[deconf_method] for a in y1_sig]
            sizes_y0 = [30 if a else 25 for a in y0_sig]
            sizes_y1 = [30 if a else 25 for a in y1_sig]

            from scipy.stats import mannwhitneyu

            stat, p = mannwhitneyu(y0, y1)
            print('Method {}, p-value = {}'.format(deconf_method, p))

            ax[i].scatter(x0, y0, s=sizes_y0, edgecolor=edgecolors_y0, facecolor=facecolor_y0)
            ax[i].scatter(x1, y1, s=sizes_y1, edgecolor=edgecolors_y1, facecolor=facecolor_y1)
            for k in range(y0.shape[0]):
                ax[i].plot([x0[k], x1[k]], [y0[k], y1[k]], ls='-', linewidth=0.8,
                        color=method_palette[deconf_method], zorder=-10)
            (p_left, p_rope, p_right) = two_on_multiple(x=y0_animal, y=y1_animal, rope=0, runs=1)
            print(deconf_method, p_right)
            if p_right > 1 - alpha_level:
                siglines.append(True)
            else:
                siglines.append(False)

            ax[i].legend().remove()
            ax[i].set_xlim([-0.2, 1.2])
            ax[i].set_xticks([0, 1])
            #ax[i].set_ylim(ylims2)
            # ax.set_yticks(yticks2)
            ax[i].set_xticklabels(['True C', 'Shuffled C'], rotation=45, ha='center')
            # ax.set_title(method_labels[deconf_method].replace('\n', ' '))
            ax[i].axhline(0.0, ls=':', c='grey', zorder=-20)
            # ax.text(x=0.5, y=0.43, text='p={:.5f}'.format(p), s=8)
            ax[i].set_title(pr_labels_detailed[deconf_method], fontsize=10)

        for axx, sigline in zip(ax, siglines):
            if sigline:
                axx.plot([0, 1], [np.max(ytops)+0.05, np.max(ytops)+0.05], ls='-', c='k', lw=1)

        ax[0].set_ylabel('Predictability [bits]')
        f.suptitle(calibration_modes_labels_detailed[mode], fontsize=10, x=0.6)
        plt.tight_layout()
        sns.despine()
        plt.show()

        plot_name = 'dotsss_t{}_{}.{}'.format(time_bin_index_dots, mode, plot_format)
        f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                  bbox_inches="tight")



ds = pd.concat(df_scores)

for figure_mode in figure_modes:
    if figure_mode == 'MAIN':
        if save_dfscore:
            output_folder = os.path.join(DATA_PATH, 'ephys_data_CR_PR_combined')
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            dssave = ds[np.isin(ds['deconf_method'], deconf_methods_main)]
            output_full_path = os.path.join(output_folder, 'PR_dfscore_{}.pkl'.format(settings_name_global))
            print('Saving scores to {}'.format(output_full_path))
            pickle.dump(dssave, open(output_full_path, 'wb'))

        ds30 = ds[ds['n_components'] == n_components_mainfig]
        for score in ['TPR_TNR_av', 'TPR', 'FPR']:
            if len(deconf_methods_main) == 1:
                figsize = [1.8, 3]
            else:
                figsize = [1.5 * len(deconf_methods_main), 3]
            f, ax = plt.subplots(1, len(deconf_methods_main), sharey=True,
                                 sharex=True, figsize=figsize)
            if len(deconf_methods_main) == 1:
                ax = [ax]
            for i, deconf_method in enumerate(deconf_methods_main):
                palette = sns.dark_palette(method_palette[deconf_method], n_colors=len(modes))
                mode_palette = {mode: palette[i] for i, mode in enumerate(modes)}
                sns.barplot(data=ds30[ds30['deconf_method'] == deconf_method],
                            x='mode', order=modes, y=score, errorbar=None, dodge=True,
                            color=method_palette[deconf_method], ax=ax[i])
                for bars in ax[i].containers:
                    for bar, mode in zip(bars, modes):
                        bar.set_facecolor(mode_palette[mode])
                ax[i].set_xlabel('')
                ax[i].set_xticklabels([calibration_modes_labels[s] for s in modes], rotation=90)
                ax[i].set_title(pr_labels_detailed[deconf_method], fontsize=10)
                if i == 0:
                    handles_lin, labels_lin = ax[i].get_legend_handles_labels()
                if i == 1:
                    handles_nonlin, labels_nonlin = ax[i].get_legend_handles_labels()
                ax[i].legend().remove()
                ax[i].set_ylim([0, 1])
                # Move the legend to ax2
            ax[0].set_ylabel(score_labels[score])
            plt.show()
            sns.despine()
            plt.tight_layout()

            plot_name = 'barplot_single_modes_{}_{}.{}'.format(settings_name_global, score, plot_format)
            f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                      bbox_inches="tight")

        for score in ['TPR_TNR_av', 'TPR', 'FPR']:

            f, ax = plt.subplots(1, len(deconf_methods_main), sharey=True,
                                 sharex=True, figsize=[2*len(deconf_methods_main),3])

            if len(deconf_methods_main) == 1:
                ax = [ax]
            for kk, deconf_method in enumerate(deconf_methods_main):
                palette = sns.dark_palette(method_palette[deconf_method], n_colors=len(modes))
                mode_palette = {mode: palette[i] for i, mode in enumerate(modes)}
                for mi, mode in enumerate(modes_order_scatter):
                    fsc = ds[(ds['deconf_method'] == deconf_method) &
                             (ds['mode'] == mode)]
                    n_comps =  fsc['n_components'].values.astype(float)
                    x = fsc['n_components'].values.astype(float)+(mi-len(modes)/2)/1.3
                    vals = fsc[score].values.astype(float)

                    ax[kk].plot(x, vals, c=mode_palette[mode],
                                   marker=mode_markers[mode],
                                   markerfacecolor='w', markeredgecolor=mode_palette[mode],
                                   label=calibration_modes_labels[mode])

                ax[kk].axhline(0.5, ls=':', c='grey', zorder=-10)
                ax[kk].set_ylim([0, 1.03])
                ax[kk].set_yticks(np.arange(0, 1.1, 0.2))
                ax[kk].set_xticks(n_comps)
                ax[kk].set_xlim([n_comps[0] - 3, n_comps[-1] + 3])
                ax[kk].set_xlabel('# movement PCs')
                ax[kk].set_title(pr_labels_detailed[deconf_method], fontsize=10)

                sns.despine()
            ax[0].legend()
            ax[1].legend()
            handles_lin, labels_lin = ax[0].get_legend_handles_labels()
            handles_nonlin, labels_nonlin = ax[1].get_legend_handles_labels()
            ax[0].legend().remove()
            ax[1].legend().remove()
            ax[0].set_ylabel(score_labels[score])
            plt.tight_layout()
            plt.show()
            plot_name = '{}_vs_n_components.{}'.format(score, plot_format)
            f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                      bbox_inches="tight")

        f, ax = plt.subplots(1, 2)
        ax[0].legend(handles_lin, labels_lin, frameon=False)
        ax[0].axis('off')
        ax[1].legend(handles_nonlin, labels_nonlin, frameon=False)
        ax[1].axis('off')
        plt.show()
        plot_name = 'legend_modes_n_comp_{}.{}'.format(settings_name_global, plot_format)
        f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                  bbox_inches="tight")

    if figure_mode == 'SUPPLEMENTARY':

        ds30 = ds[ds['n_components'] == n_components_mainfig]
        for score in ['TPR_TNR_av', 'TPR', 'FPR']:
            if len(deconf_methods_supplementary) == 1:
                figsize = [1.8, 3]
            else:
                figsize = [1.5 * len(deconf_methods_supplementary), 3]
            f, ax = plt.subplots(1, len(deconf_methods_supplementary), sharey=True,
                                 sharex=True, figsize=figsize)
            if len(deconf_methods_supplementary) == 1:
                ax = [ax]
            for i, deconf_method in enumerate(deconf_methods_supplementary):
                palette = sns.dark_palette(method_palette[deconf_method], n_colors=len(modes))
                mode_palette = {mode: palette[i] for i, mode in enumerate(modes)}
                sns.barplot(data=ds30[ds30['deconf_method'] == deconf_method],
                            x='mode', order=modes, y=score, errorbar=None, dodge=True,
                            color=method_palette[deconf_method], ax=ax[i])
                for bars in ax[i].containers:
                    for bar, mode in zip(bars, modes):
                        bar.set_facecolor(mode_palette[mode])
                plt.show()
                ax[i].set_xlabel('')
                ax[i].set_xticklabels([calibration_modes_labels[s] for s in modes], rotation=90)
                ax[i].set_title(pr_labels_detailed[deconf_method], fontsize=10)
                if i == 0:
                    handles_lin, labels_lin = ax[i].get_legend_handles_labels()
                if i == 1:
                    handles_nonlin, labels_nonlin = ax[i].get_legend_handles_labels()
                ax[i].legend().remove()
                ax[i].set_ylim([0, 1])
                # Move the legend to ax2
            ax[0].set_ylabel(score_labels[score])
            plt.show()
            sns.despine()
            plt.tight_layout()

            plot_name = 'SUPPLEMENTARY_barplot_single_modes_{}_{}.{}'.format(settings_name_global, score, plot_format)
            f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                      bbox_inches="tight")

        for score in ['TPR_TNR_av', 'TPR', 'FPR']:

            f, ax = plt.subplots(1, len(deconf_methods_supplementary), sharey=True,
                                 sharex=True, figsize=[2 * len(deconf_methods_supplementary), 3])
            if len(deconf_methods_supplementary) == 1:
                ax = [ax]
            for kk, deconf_method in enumerate(deconf_methods_supplementary):
                palette = sns.dark_palette(method_palette[deconf_method], n_colors=len(modes))
                mode_palette = {mode: palette[i] for i, mode in enumerate(modes)}
                for mode in modes:
                    fsc = ds[(ds['deconf_method'] == deconf_method) &
                             (ds['mode'] == mode)]
                    x = fsc['n_components'].values.astype(float)
                    vals = fsc[score].values.astype(float)

                    ax[kk].plot(x, vals, c=mode_palette[mode],
                                marker=mode_markers[mode],
                                markerfacecolor='w', markeredgecolor=mode_palette[mode],
                                label=calibration_modes_labels[mode])

                ax[kk].axhline(0.5, ls=':', c='grey', zorder=-10)
                ax[kk].set_ylim([0, 1.03])
                ax[kk].set_yticks(np.arange(0, 1.1, 0.2))
                ax[kk].set_xticks(x)
                ax[kk].set_xlim([x[0] - 3, x[-1] + 3])
                ax[kk].set_xlabel('# movement PCs')
                ax[kk].set_title(pr_labels_detailed[deconf_method], fontsize=10)

                sns.despine()
            ax[0].legend()
            handles_lin, labels_lin = ax[0].get_legend_handles_labels()
            ax[0].legend().remove()
            ax[0].set_ylabel(score_labels[score])
            plt.tight_layout()
            plt.show()
            plot_name = 'SUPPLEMENTARY_scatterplot_{}_vs_n_components.{}'.format(score, plot_format)
            f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                      bbox_inches="tight")


        for score in ['TPR_TNR_av', 'FPR', 'TPR']:

            f, ax = plt.subplots(1, 1, figsize=[2.5, 3])
            ds['dummy'] = 'dummy'
            arts = sns.lineplot(data=ds, x='n_components', y=score,
                                hue='deconf_method', hue_order=deconf_methods_supplementary_comparison,
                                palette=nonlinear_pr_method_palette, ax=ax,
                                markers='o', style='dummy')
            for coll in arts.collections:
                coll.set_linewidth(0)
            x = np.sort(ds['n_components'].unique().astype(int))
            ax.set_ylim([0, 1.01])
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_xticks(x)
            ax.set_xlim([x[0] - 3, x[-1] + 3])
            ax.set_xlabel('# movement PCs')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()
            ax.set_ylabel(score_labels[score])
            sns.despine()
            plt.show()
            plt.tight_layout()
            plot_name = 'SUPPLEMENTARY_comparison_{}_vs_n_components_averaged.{}'.format(score, plot_format)
            f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                      bbox_inches="tight")

        f, ax = plt.subplots(1, 1, figsize=[2, 2])
        legend_elements = [plt.Line2D([0], [0], marker='o', color=nonlinear_pr_method_palette[m],
                                      label=pr_labels_detailed[m], lw=2,
                                      markerfacecolor=nonlinear_pr_method_palette[m],
                                      markeredgecolor='w',
                                      markersize=10) for m in deconf_methods_supplementary_comparison]
        ax.legend(handles=legend_elements, loc='center', frameon=False,
                  ncol=1)
        sns.despine(left=True, bottom=True)
        ax.axis('off')
        plt.show()
        plot_name = 'SUPPLEMENTARY_legend_predictability.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder_general, plot_name), dpi=dpi,
                  bbox_inches="tight")













