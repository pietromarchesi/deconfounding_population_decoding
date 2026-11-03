import glob
import os
import pickle
import numpy as np
import pandas as pd
from constants import DATA_PATH
import matplotlib.pyplot as plt
from plotting_style import *
from session_selection import animal_id_per_session
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

"""
Loading results and generating figures for CR applied to V1 data, 
combining results with and without hyperparameter optimization. 
"""

# SUPPLEMENTARY FIG. 9
settings_name_global = 'OCT9_NEWDATA'

modes = ['all_sessions_significance_n_components_notune_cr_notune_dec',
         'all_sessions_significance_n_components_notune_cr_tune_dec']

# PLOTTING PARAMETERS
dpi = 400
save_dfscore = False
alpha_level_permutations = 0.05
plot_format = 'png'

selected_methods = ['linear_confound_regression',
                    'linear_confound_regression_glm',
                     'nonlinear_confound_regression_rf']

deconf_methods = selected_methods

selected_decoders = ['logistic_regression',
                    'logistic_regression_tuned',
                    'SVC',
                    'SVC_tuned',
                    'random_forest',
                    'random_forest_tuned']

plots_folder = os.path.join(DATA_PATH, 'plots', 'EPHYS_PLOTS', 'CR_combined', settings_name_global)
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

dfscores = []
for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)
    # LOAD DATA

    output_folder = os.path.join(DATA_PATH, 'ephys_data_CR_results',
                                 'SIMULATION_{}'.format(settings_name))
    files = glob.glob('{}/*'.format(output_folder))

    cr_rs, dec_rs, pars = [], [], []
    for file in files:
        try:
            res = pickle.load(open(file, 'rb'))
        except:
            res = pd.read_pickle(file)
        cr_rs.append(res['cr_rs'])
        dec_rs.append(res['dec_rs'])
        pars.append(res['pars'])
    pars = pars[0]
    cr_rs = pd.concat(cr_rs)
    dec_rs = pd.concat(dec_rs)

    decoders = pars['decoders']
    shuffle_components_list = pars['shuffle_components_list']
    shuffle_firing_rates_list = pars['shuffle_firing_rates_list']
    n_permutations = pars['n_permutations']
    session_ids = dec_rs['session_id'].unique()
    time_bin_indexes = pars['time_bin_indexes']
    time_bin_times = pars['time_bin_times']
    time_bins = pars['time_bins']
    response_time_bin_index = pars['time_bin_index']
    decode_from_confound_predictions = pars['decode_from_confound_predictions']
    mode = pars['mode']

    # PREPARE REGRESSION SCORES

    # PREPARE DECODING SCORES
    dec_rs['mean_score'] = pd.to_numeric(dec_rs['mean_score'])

    n_pca_components = [30]
    dec_rs = dec_rs[dec_rs['n_pca_components'] == 30]

    excude_cols = ['mean_score', 'fold']
    decav = dec_rs.groupby([c for c in dec_rs.columns if ~np.isin(c, excude_cols)]).mean().reset_index()
    decav['animal_id'] = [animal_id_per_session[s] for s in decav['session_id'].values]

    dt = decav[decav['time'] == 0.2]
    dt = dt[np.isin(dt['deconf_method'], ['raw', 'linear_confound_regression_glm_tuned',
                                     'linear_confound_regression_ridge'])]
    dt = dt[dt['decoder'] == 'logistic_regression_tuned']

    if 'all_sessions_significance' in mode:

        # SIGNIFICANCE PER SESSION
        dps = []
        dss = []

        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for shuffle_components in shuffle_components_list:
                    if shuffle_components:
                        extra_string = 'shuffledC'
                    else:
                        extra_string = 'trueC'

                    dp = pd.DataFrame(columns=['session_id',
                                               'animal_id',
                                               'n_components',
                                               'deconf_method',
                                               'decoder',
                                               'time_bin_index',
                                               'time',
                                               'pvalue',
                                               'mean_score',
                                               'mean_permuted_score',
                                               'is_sig'])

                    for session_id in session_ids:
                        for kk, deconf_method in enumerate(deconf_methods):
                            for decoder in decoders:
                                df_sel = decav[(decav['session_id'] == session_id) &
                                               (decav['deconf_method'] == deconf_method) &
                                               (decav['n_pca_components'] == n_components) &
                                               (decav['decoder'] == decoder) &
                                               (decav['shuffle_confound'] == shuffle_components) &
                                               (decav['time_bin'] == time_bin_index)]

                                obs = df_sel[df_sel['permuted'] == False]
                                obs_val = obs['mean_score'].iloc[0]
                                animal_id = obs['animal_id'].iloc[0]

                                if n_permutations > 0:
                                    perms = df_sel[(df_sel['permuted'] == True)]
                                    perms_val = perms['mean_score'].__array__()
                                    p_val = (perms_val > obs_val).sum() / (n_permutations + 1)
                                    is_sig = p_val <= alpha_level_permutations
                                    perms_mean_val = perms_val.mean()
                                else:
                                    perms_mean_val = np.nan
                                    p_val = np.nan
                                    is_sig = False

                                dp.loc[dp.shape[0], :] = [session_id, animal_id, n_components, deconf_method, decoder,
                                                          time_bin_index, time,
                                                          p_val, obs_val, perms_mean_val, is_sig]

                    dp['shuffledC'] = shuffle_components
                    dps.append(dp)

            dp = pd.concat(dps)
            dp['mean_score'] = pd.to_numeric(dp['mean_score'])


        # PERCENTAGE AND ABSOLUTE NUMBER OF SIGNIFICANT RECORDINGS
        ds = pd.DataFrame(columns=['deconf_method',
                                   'n_components',
                                   'decoder',
                                   'shuffledC',
                                   'time_bin_index',
                                   'time',
                                   'perc_sig',
                                   'N_sig'])
        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for decoder in decoders:
                    for kk, deconf_method in enumerate(deconf_methods):
                        for shuffle_components in [True, False]:
                            dpsel = dp[(dp['deconf_method'] == deconf_method) &
                                       (dp['n_components'] == n_components) &
                                       (dp['shuffledC'] == shuffle_components) &
                                       (dp['decoder'] == decoder) &
                                       (dp['time_bin_index'] == time_bin_index)]
                            n_sess = dpsel['session_id'].unique().shape[0]
                            n_sig = dpsel['is_sig'].astype(bool).sum()
                            perc_sig = 100 * n_sig / dpsel.shape[0]
                            ds.loc[ds.shape[0], :] = [deconf_method, n_components, decoder, shuffle_components,
                                                      time_bin_index, time, perc_sig, n_sig]
        ds['perc_sig'] = pd.to_numeric(ds['perc_sig'])

        # FPR, TPR, BAL. ACC. SCORES
        dfscore = pd.DataFrame(columns=['deconf_method',
                                        'decoder',
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

        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for decoder in decoders:
                    for kk, deconf_method in enumerate(deconf_methods):
                        dpsel1 = dp[(dp['deconf_method'] == deconf_method) &
                                    (dp['n_components'] == n_components) &
                                    (dp['decoder'] == decoder) &
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
                                    (dp['decoder'] == decoder) &
                                    (dp['shuffledC'] == False) &
                                    (dp['time_bin_index'] == time_bin_index)]
                        n_sess = dpsel2['session_id'].unique().shape[0]
                        # if C is shuffled, significant sessions are true positives
                        fp = dpsel2['is_sig'].astype(bool).sum()
                        tn = np.where(dpsel2['is_sig'].astype(bool) == False)[0].shape[0]
                        ypred2 = dpsel2['is_sig'].astype(int).__array__()
                        ytest2 = np.zeros_like(ypred2)
                        TNR = tn / n_sess
                        FPR = fp / n_sess

                        ytrue = np.hstack([ytest1, ytest2])
                        ypred = np.hstack([ypred1, ypred2])
                        f1 = f1_score(y_true=ytrue, y_pred=ypred)
                        precision = precision_score(y_true=ytrue, y_pred=ypred)
                        recall =  recall_score(y_true=ytrue, y_pred=ypred)
                        balacc = balanced_accuracy_score(y_true=ytrue, y_pred=ypred)
                        print(deconf_method, tp, fn, fp)
                        f1_manual = 2 * tp / (2 * tp + fp + fn)
                        assert f1 - f1_manual < 1e-8
                        TPR_TNR = (TPR + TNR) / 2

                        dfscore.loc[dfscore.shape[0], :] = [deconf_method, decoder, n_components,
                                                            time_bin_index, time, f1, precision, recall,
                                                            TPR, TNR, TPR_TNR, FPR, balacc]

        numeric_cols = ['f1_score', 'precision', 'recall', 'TPR', 'TNR', 'TPR_TNR_av', 'bal_accuracy']
        for col in numeric_cols:
            dfscore[col] = pd.to_numeric(dfscore[col])
        dfscore = dfscore.round(4)
        #df_scores.append(dfscore)

        # SAVE FPR/TPR/BAL.ACC. SCORES TO PLOT WITH PR RESULTS
        if save_dfscore:
            output_folder = os.path.join(DATA_PATH, 'ephys_data_CR_PR_combined')
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            output_full_path = os.path.join(output_folder, 'CR_dfscore_{}_{}.pkl'.format(settings_name_global, mode))
            print('Saving scores to {}'.format(output_full_path))
            pickle.dump(dfscore, open(output_full_path, 'wb'))

        dps = []
        dss = []

        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for shuffle_components in shuffle_components_list:
                    if shuffle_components:
                        extra_string = 'shuffledC'
                    else:
                        extra_string = 'trueC'

                    dp = pd.DataFrame(columns=['session_id',
                                               'animal_id',
                                               'n_components',
                                               'deconf_method',
                                               'decoder',
                                               'time_bin_index',
                                               'time',
                                               'pvalue',
                                               'mean_score',
                                               'mean_permuted_score',
                                               'is_sig'])

                    for session_id in session_ids:
                        for kk, deconf_method in enumerate(deconf_methods):
                            for decoder in decoders:
                                df_sel = decav[(decav['session_id'] == session_id) &
                                               (decav['deconf_method'] == deconf_method) &
                                               (decav['n_pca_components'] == n_components) &
                                               (decav['decoder'] == decoder) &
                                               (decav['shuffle_confound'] == shuffle_components) &
                                               (decav['time_bin'] == time_bin_index)]

                                obs = df_sel[df_sel['permuted'] == False]
                                obs_val = obs['mean_score'].iloc[0]
                                animal_id = obs['animal_id'].iloc[0]

                                if n_permutations > 0:
                                    perms = df_sel[(df_sel['permuted'] == True)]
                                    perms_val = perms['mean_score'].__array__()
                                    p_val = (perms_val > obs_val).sum() / (n_permutations + 1)
                                    is_sig = p_val <= alpha_level_permutations
                                    perms_mean_val = perms_val.mean()
                                else:
                                    perms_mean_val = np.nan
                                    p_val = np.nan
                                    is_sig = False

                                dp.loc[dp.shape[0], :] = [session_id, animal_id, n_components, deconf_method, decoder,
                                                          time_bin_index, time,
                                                          p_val, obs_val, perms_mean_val, is_sig]

                    dp['shuffledC'] = shuffle_components
                    dps.append(dp)

            dp = pd.concat(dps)
            dp['mean_score'] = pd.to_numeric(dp['mean_score'])

    # PROCESS RESULTS
    if 'all_sessions_significance' in mode:
        # PERCENTAGE AND ABSOLUTE NUMBER OF SIGNIFICANT RECORDINGS
        ds = pd.DataFrame(columns=['deconf_method',
                                   'n_components',
                                   'decoder',
                                   'shuffledC',
                                   'time_bin_index',
                                   'time',
                                   'perc_sig',
                                   'N_sig'])
        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for decoder in decoders:
                    for kk, deconf_method in enumerate(deconf_methods):
                        for shuffle_components in [True, False]:
                            dpsel = dp[(dp['deconf_method'] == deconf_method) &
                                       (dp['n_components'] == n_components) &
                                       (dp['shuffledC'] == shuffle_components) &
                                       (dp['decoder'] == decoder) &
                                       (dp['time_bin_index'] == time_bin_index)]
                            n_sess = dpsel['session_id'].unique().shape[0]
                            n_sig = dpsel['is_sig'].astype(bool).sum()
                            perc_sig = 100 * n_sig / dpsel.shape[0]
                            ds.loc[ds.shape[0], :] = [deconf_method, n_components, decoder, shuffle_components,
                                                      time_bin_index, time, perc_sig, n_sig]
        ds['perc_sig'] = pd.to_numeric(ds['perc_sig'])
        print(ds)


        dfscore = pd.DataFrame(columns=['deconf_method',
                                        'decoder',
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

        for n_components in n_pca_components:
            for time_bin_index in time_bin_indexes:
                time = time_bin_times[time_bin_index]
                for decoder in decoders:
                    for kk, deconf_method in enumerate(deconf_methods):
                        dpsel1 = dp[(dp['deconf_method'] == deconf_method) &
                                    (dp['n_components'] == n_components) &
                                    (dp['decoder'] == decoder) &
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
                                    (dp['decoder'] == decoder) &
                                    (dp['shuffledC'] == False) &
                                    (dp['time_bin_index'] == time_bin_index)]
                        n_sess = dpsel2['session_id'].unique().shape[0]
                        # if C is shuffled, significant sessions are true positives
                        fp = dpsel2['is_sig'].astype(bool).sum()
                        tn = np.where(dpsel2['is_sig'].astype(bool) == False)[0].shape[0]
                        ypred2 = dpsel2['is_sig'].astype(int).__array__()
                        ytest2 = np.zeros_like(ypred2)
                        TNR = tn / n_sess
                        FPR = fp / n_sess

                        ytrue = np.hstack([ytest1, ytest2])
                        ypred = np.hstack([ypred1, ypred2])
                        f1 = f1_score(y_true=ytrue, y_pred=ypred)
                        precision = precision_score(y_true=ytrue, y_pred=ypred)
                        recall =  recall_score(y_true=ytrue, y_pred=ypred)
                        balacc = balanced_accuracy_score(y_true=ytrue, y_pred=ypred)
                        print(deconf_method, tp, fn, fp)
                        f1_manual = 2 * tp / (2 * tp + fp + fn)
                        assert f1 - f1_manual < 1e-8
                        TPR_TNR = (TPR + TNR) / 2

                        dfscore.loc[dfscore.shape[0], :] = [deconf_method, decoder, n_components,
                                                            time_bin_index, time, f1, precision, recall,
                                                            TPR, TNR, TPR_TNR, FPR, balacc]

        numeric_cols = ['f1_score', 'precision', 'recall', 'TPR', 'TNR', 'TPR_TNR_av', 'bal_accuracy']
        for col in numeric_cols:
            dfscore[col] = pd.to_numeric(dfscore[col])
        dfscore = dfscore.round(4)

        if mode == 'all_sessions_significance_n_components_notune_cr_tune_dec':
            dfscore['decoder'] = ['{}_tuned'.format(i) for i in dfscore['decoder']]

        #df_scores.append(dfscore)
        dfscores.append(dfscore)



ds = pd.concat(dfscores)

for score in ['TPR', 'FPR']:
    f, ax = plt.subplots(1, len(selected_methods), sharey=True,
                         sharex=True, figsize=[2.5 * len(selected_methods), 3])
    if len(selected_methods) == 1:
        ax = np.array([ax])
    for i, deconf_method in enumerate(selected_methods):
        palette = sns.dark_palette(method_palette[deconf_method], n_colors=len(selected_decoders)/2)
        palette = [p for p in palette for _ in (0, 1)]
        decoder_palette = {decoder: palette[i] for i, decoder in enumerate(selected_decoders)}
        sns.barplot(data=ds[ds['deconf_method'] == deconf_method],
                    x='decoder', order=selected_decoders, y=score, errorbar=None, dodge=True,
                    color=method_palette[deconf_method], ax=ax[i], gap=0.05)
        for bars in ax[i].containers:
            for bar, decoder in zip(bars, selected_decoders):
                bar.set_facecolor(decoder_palette[decoder])
        ax[i].set_xlabel('')
        ax[i].set_xticklabels([decoder_labels[s] for s in selected_decoders], rotation=90)
        ax[i].set_title(cr_labels_detailed[deconf_method], fontsize=10)
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

    plot_name = 'barplot_single_decoders_30_comp_{}_{}.{}'.format(settings_name_global, score, plot_format)
    f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
              bbox_inches="tight")