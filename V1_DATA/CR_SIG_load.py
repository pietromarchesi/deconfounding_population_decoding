import os
import glob
import pickle
from constants import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from plotting_style import *
from session_selection import animal_id_per_session
from scipy.stats import mannwhitneyu, bootstrap
from baycomp import two_on_single, two_on_multiple
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

"""
Loading results and generating figures for CR applied to V1 data.
"""

# ----------------------------------------------------------
# select one of the options

# MAIN FIG. 5 and 6
settings_name_global = 'OCT9_NEWDATA'
modes = ['all_sessions_significance_n_components_notune_cr_notune_dec']
deconf_methods = ['raw', 'linear_confound_regression',
                  'nonlinear_confound_regression_rf']
save_dfscore = True
n_pcs_plot = 'all'
plot_dots = True
plot_violins = True
plot_barplots = True
plot_score_barplots = True
plot_score_lineplots = True


# # CONFOUND LEAKAGE FIG. 6
# settings_name_global = 'OCT9_NEWDATA'
# modes = ['all_sessions_leakage',
#          'all_sessions_leakage_shuffle_features']
# deconf_methods = ['raw', 'linear_confound_regression',
#                   'nonlinear_confound_regression_rf']
# save_dfscore = False
# n_pcs_plot = 'all'
# plot_dots = False
# plot_violins = False
# plot_barplots = False
# plot_score_barplots = False
# plot_score_lineplots = False


# # CONFOUND LEAKAGE FIG. 6
# settings_name_global = 'OCT9_NEWDATA'
# modes = ['all_sessions_leakage_supplementary',
#          'all_sessions_leakage_shuffle_features_supplementary']
# deconf_methods = ['raw', 'linear_confound_regression_glm']
# save_dfscore = False
# n_pcs_plot = 'all'
# plot_dots = False
# plot_violins = False
# plot_barplots = False
# plot_score_barplots = False
# plot_score_lineplots = False

# 0-400 MS TIME BIN (SUPPLEMENTARY FIG. 8)
# settings_name_global = 'OCT9_NEWDATA_0_400_ALLCOMP'
# modes = ['all_sessions_significance_n_components_notune_cr_notune_dec']
# deconf_methods = ['linear_confound_regression',
#                   'nonlinear_confound_regression_rf']
# save_dfscore = False
# n_pcs_plot = 'all'
# plot_dots = False
# plot_violins = False
# plot_barplots = False
# plot_score_barplots = True
# plot_score_lineplots = True

# ----------------------------------------------------------

# PLOTTING PARAMETERS
dpi = 400
plot_format = 'png'
n_pca_components_main_plot = 30
alpha_level_permutations = 0.05
alpha_level = 0.05
leakage_in_individual_sessions = False


for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)
    # LOAD DATA
    plots_folder = os.path.join(DATA_PATH, 'plots', 'EPHYS_PLOTS', 'CR', settings_name)
    if not os.path.isdir(plots_folder) :
        os.makedirs(plots_folder)

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
    n_pca_components = pars['n_pca_components']
    if n_pcs_plot == 'all':
        try:
            n_pca_components[0]
        except TypeError:
            n_pca_components = [n_pca_components]
    else:
        n_pca_components = [n_pcs_plot]
    mode = pars['mode']

    #deconf_methods = pars['deconf_methods']
    #deconf_methods = np.sort(dec_rs['deconf_method'].unique())
    #deconf_methods = main_methods
    # PREPARE REGRESSION SCORES

    # PREPARE DECODING SCORES
    dec_rs['mean_score'] = pd.to_numeric(dec_rs['mean_score'])
    excude_cols = ['mean_score', 'fold']
    decav = dec_rs.groupby([c for c in dec_rs.columns if ~np.isin(c, excude_cols)]).mean().reset_index()
    decav['animal_id'] = [animal_id_per_session[s] for s in decav['session_id'].values]

    dt = decav[decav['time'] == 0.2]
    dt = dt[np.isin(dt['deconf_method'], ['raw', 'linear_confound_regression_glm_tuned',
                                     'linear_confound_regression_ridge'])]
    dt = dt[dt['decoder'] == 'logistic_regression_tuned']


    # PROCESS RESULTS
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


    if np.isin(mode, ['all_sessions_leakage', 'all_sessions_leakage_supplementary', 'all_sessions_leakage_shuffle_features', 'all_sessions_leakage_shuffle_features_supplementary']):
        # p_right indicates probability of improvement vs raw
        # testing on all sessions combined
        # deconf_methods = pars['deconf_methods']

        ls = pd.DataFrame(columns=['decoder', 'shuffle_firing_rates', 'deconf_method',
                                   'p_right_bayes', 'p_right_bayes_uncorrected',
                                   'p_val_wilcoxon', 'p_val_wilcoxon_uncorrected'])
        for decoder in decoders:
            for shuffle_firing_rates in shuffle_firing_rates_list:
                for deconf_method in [i for i in deconf_methods if i!='raw']:
                    ddeconf = dec_rs[(dec_rs['decoder'] == decoder) &
                               (dec_rs['shuffle_confound'] == False) &
                               (dec_rs['shuffle_firing_rates'] == shuffle_firing_rates) &
                               (dec_rs['time_bin'] == response_time_bin_index) &
                               (dec_rs['deconf_method'] == deconf_method)]

                    draw = dec_rs[(dec_rs['decoder'] == decoder) &
                               (dec_rs['shuffle_confound'] == False) &
                               (dec_rs['shuffle_firing_rates'] == shuffle_firing_rates) &
                               (dec_rs['time_bin'] == response_time_bin_index) &
                               (dec_rs['deconf_method'] == 'raw')]

                    scores_raw, scores_cr = [], []
                    for session_id in ddeconf['session_id'].unique():
                        dxs = draw[draw['session_id'] == session_id]
                        scores_raw.append(dxs['mean_score'].__array__().reshape(-1, 1))
                        dys = ddeconf[ddeconf['session_id'] == session_id]
                        scores_cr.append(dys['mean_score'].__array__().reshape(-1, 1))
                    # session as the unit of inference
                    scores_raw = np.hstack(scores_raw).mean(axis=0)
                    scores_cr = np.hstack(scores_cr).mean(axis=0)
                    (p_left,p_rope, p_right_uncorrected) = two_on_multiple(x=scores_raw, y=scores_cr, rope=0, runs=1)
                    stat, p_w_uncorrected = wilcoxon(scores_raw, scores_cr, alternative='two-sided')

                    # using the animal as the unit of inference
                    scores = pd.DataFrame(columns=['raw', 'cr', 'session_id', 'animal_id'])
                    scores['raw'] = scores_raw
                    scores['cr'] = scores_cr
                    scores['session_id'] = ddeconf['session_id'].unique()
                    scores['animal_id'] = [animal_id_per_session[s] for s in scores['session_id'].values]
                    scores = scores.groupby('animal_id').mean(numeric_only=True).reset_index()

                    (p_left,p_rope, p_right) = two_on_multiple(x=scores['raw'], y=scores['cr'], rope=0, runs=1)
                    stat, p_w = wilcoxon(scores['raw'], y=scores['cr'], alternative='two-sided')

                    ls.loc[ls.shape[0], :] = [decoder, shuffle_firing_rates, deconf_method,
                                              p_right, p_right_uncorrected, p_w, p_w_uncorrected]
        print(ls)


    # MAKE FIGURES
    if np.isin(mode, ['all_sessions_decode_from_confound_notune_dec',  'all_sessions_decode_from_confound_tune_dec']):
        # PLOT DECODE FROM CONFOUND
        palette = sns.dark_palette(sns.xkcd_rgb['grey'], n_colors=len(decoders))
        decoder_palette = {decoder: palette[i] for i, decoder in enumerate(decoders)}

        hue_order = ['decode_from_confound']
        f, axx = plt.subplots(1, figsize=[3, 3], sharey=True)
        dx = decav[(decav['shuffle_confound'] == False) &
                   (decav['shuffle_firing_rates'] == False)]
        n_pcs = np.sort(dx['n_pca_components'].unique().astype(int))
        arts = sns.lineplot(data=dx[dx['deconf_method']=='decode_from_confound'],
                     x='n_pca_components', y='mean_score', hue='decoder',
                     errorbar=('ci', 95), ax=axx, style='decoder', palette=decoder_palette,
                     markers=decoder_markers, markersize=8)
        for coll in arts.collections:
            coll.set_linewidth(0)
        for line in arts.lines:
            line.set_linestyle('-')
        axx.legend()
        handles, labels = axx.get_legend_handles_labels()
        axx.legend().remove()
        sns.despine()
        axx.set_ylabel('Mean accuracy\nDecoding from $C$')
        axx.set_xlabel('# movement PCs')
        axx.axhline(0.5, ls=':', c='grey')
        axx.set_ylim([0.4, 0.9])
        axx.set_xticks(n_pcs)

        plt.tight_layout()
        plt.show()
        plot_name = 'decode_from_C.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")

        f, ax = plt.subplots(1, 1)
        labels_mod = [decoder_labels[l] for l in labels]
        ax.legend(handles, labels_mod, frameon=False)
        ax.axis('off')
        plt.show()
        plot_name = 'dec_from_conf_legend.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")


    elif 'all_sessions_significance' in mode:
        dpsel = dp[dp['n_components'] == n_pca_components_main_plot]
        selected_methods = [d for d in deconf_methods if d != 'raw']

        if plot_dots:
            # PLOT DOTS for the deconf methods
            #for time_bin_index in time_bin_indexes:
            time = time_bin_times[response_time_bin_index]
            ytop = 0.95
            ylims2 = [0.4, ytop]
            yticks2 = [0.4, 0.5, 0.6, 0.7, 0.8]
            for decoder in decoders:
                for deconf_method in selected_methods:
                    f, ax = plt.subplots(1, 1, figsize=[2.8, 3], sharey=True)

                    dfsel_raw = dpsel[(dpsel['deconf_method'] == 'raw') &
                                (dpsel['decoder'] == decoder) &
                                (dpsel['time'] == time)].copy()

                    dfsel = dpsel[(dpsel['deconf_method'] == deconf_method) &
                                (dpsel['decoder'] == decoder) &
                                (dpsel['time'] == time)].copy()

                    dfsel_raw = dfsel_raw.sort_values(by='animal_id')
                    dfsel_raw = dfsel_raw.drop(['mean_permuted_score'],axis=1)

                    dfsel = dfsel.sort_values(by='animal_id')
                    dfsel = dfsel.drop(['mean_permuted_score'],axis=1)

                    n_sessions = dfsel['session_id'].unique().shape[0]

                    y0 = dfsel_raw[dfsel_raw['shuffledC'] == False]['mean_score'].__array__(float)
                    y1 = dfsel[dfsel['shuffledC'] == True]['mean_score'].__array__(float)
                    y2 = dfsel[dfsel['shuffledC'] == False]['mean_score'].__array__(float)

                    y0_sig = dfsel_raw[dfsel_raw['shuffledC'] == False]['is_sig'].__array__()
                    y1_sig = dfsel[dfsel['shuffledC'] == True]['is_sig'].__array__()
                    y2_sig = dfsel[dfsel['shuffledC'] == False]['is_sig'].__array__()

                    # for the statistics, treat the animal as the unit of inference
                    y0_animal = dfsel_raw[dfsel_raw['shuffledC'] == False].groupby('animal_id').mean(numeric_only=True).sort_values(by='animal_id').reset_index()['mean_score'].__array__(float)
                    y1_animal = dfsel[dfsel['shuffledC'] == True].groupby('animal_id').mean(numeric_only=True).sort_values(by='animal_id').reset_index()['mean_score'].__array__(float)
                    y2_animal = dfsel[dfsel['shuffledC'] == False].groupby('animal_id').mean(numeric_only=True).sort_values(by='animal_id').reset_index()['mean_score'].__array__(float)

                    x0 = [0 + np.random.uniform(-0.05, 0.05) for i in range(y0.shape[0])]
                    x1 = [1 + np.random.normal(-0.05, 0.05) for i in range(y1.shape[0])]
                    x2 = [2 + np.random.normal(-0.05, 0.05) for i in range(y2.shape[0])]

                    edgecolors_y0 = ['w' if a else method_palette['raw'] for a in y0_sig]
                    edgecolors_y1 = ['w' if a else method_palette[deconf_method] for a in y1_sig]
                    edgecolors_y2 = ['w' if a else method_palette[deconf_method] for a in y2_sig]

                    facecolor_y0 = [method_palette['raw'] if a else 'w' for a in y0_sig]
                    facecolor_y1 = [method_palette[deconf_method] if a else 'w' for a in y1_sig]
                    facecolor_y2 = [method_palette[deconf_method] if a else 'w' for a in y2_sig]

                    size_y0 = [55 if a else 35 for a in y0_sig]
                    size_y1 = [55 if a else 35 for a in y1_sig]
                    size_y2 = [55 if a else 35 for a in y2_sig]

                    ax.scatter(x0, y0, s=size_y0, edgecolor=edgecolors_y0, facecolor=facecolor_y0)
                    ax.scatter(x1, y1, s=size_y1, edgecolor=edgecolors_y1, facecolor=facecolor_y1)
                    ax.scatter(x2, y2, s=size_y2, edgecolor=edgecolors_y2, facecolor=facecolor_y2)

                    for i in range(y0.shape[0]):
                        ax.plot([x0[i], x1[i]], [y0[i], y1[i]], ls='-', linewidth=1,
                                color=method_palette[deconf_method], zorder=-10)
                        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], ls='-', linewidth=1,
                                color=method_palette[deconf_method], zorder=-10)

                    (p_left, p_rope, p_right) = two_on_multiple(x=y1_animal, y=y0_animal, rope=0, runs=1)
                    print(decoder, deconf_method, p_right)
                    # BONFERRONI CORRECTION *2
                    if p_right > 1 - alpha_level/2:
                        ax.plot([0, 1], [ytop-0.02, ytop-0.02], ls='-', c='k', lw=1)

                    (p_left2, p_rope2, p_right2) = two_on_multiple(x=y2_animal, y=y1_animal, rope=0, runs=1)
                    print(decoder, deconf_method, p_right)
                    if p_right2*2 > 1 - alpha_level/2:
                        ax.plot([1, 2], [ytop-0.01, ytop-0.01], ls='-', c='k', lw=1)

                    ax.legend().remove()
                    sns.despine()
                    ax.set_xlim([-0.2, 2.2])
                    ax.set_xticks([0, 1, 2])
                    ax.set_ylim(ylims2)
                    ax.set_yticks(yticks2)
                    ax.set_title(decoder_labels[decoder], fontsize=10)
                    label0 = 'Raw'
                    label1 = 'CR\nShuffled $C$'
                    label2 = 'CR\nTrue $C$'
                    ax.set_xticklabels([label0, label1, label2], rotation=0)
                    #ax.set_title(method_labels[deconf_method].replace('\n', ' '))
                    ax.set_ylabel('Mean accuracy')
                    ax.axhline(0.5, ls=':', c='grey', zorder=-20)
                    #ax.text(x=0.5, y=0.43, text='p={:.5f}'.format(p), s=8)
                    plt.tight_layout()
                    plt.show()
                    plot_name = 'new_dotsss_{}_{}_t{}.{}'.format(deconf_method, decoder, response_time_bin_index, plot_format)
                    f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                              bbox_inches="tight")

        if plot_violins:
            # PLOT VIOLINS oF ACCURACY SCORES (FIG. 6)
            for deconf_method in selected_methods:
                f, ax = plt.subplots(1, 1, figsize=[3.5, 3], sharey=True)

                dfsel_raw = dpsel[(dpsel['deconf_method'] == 'raw') &
                            (dpsel['time'] == time)].copy()
                dfsel_raw = dfsel_raw[dfsel_raw['shuffledC'] == False]

                dfsel = dpsel[(dpsel['deconf_method'] == deconf_method) &
                            (dpsel['time'] == time)].copy()
                dfsel = dfsel[dfsel['shuffledC'] == True]

                dfvplot = pd.concat([dfsel_raw, dfsel])

                sns.violinplot(data=dfvplot, hue='deconf_method',
                               palette=method_palette, split=True,
                               y='mean_score', x='decoder', order=decoders,
                               fill=False,
                               inner='quart', cut=2)

                for kk, decoder in enumerate(decoders):
                    d1 = dfsel_raw[dfsel_raw['decoder'] == decoder]
                    y1_animal = d1.groupby('animal_id').mean(numeric_only=True).sort_values(
                                            by='animal_id').reset_index()['mean_score'].__array__(float)

                    d2 = dfsel[dfsel['decoder'] == decoder]
                    y2_animal = d2.groupby('animal_id').mean(numeric_only=True).sort_values(
                                            by='animal_id').reset_index()['mean_score'].__array__(float)

                    (p_left2, p_rope2, p_right2) = two_on_multiple(x=y2_animal, y=y1_animal, rope=0, runs=1)
                    print(decoder, deconf_method, p_right2)
                    if p_right2 > 1 - alpha_level/len(decoders):
                        print('sig')
                        ax.plot([kk-0.2, kk+0.2], [0.98, 0.98], ls='-', c='k', lw=1)

                ax.set_xticklabels([decoder_labels[d] for d in decoders])
                ax.set_xlabel('')
                ax.set_ylabel('Mean accuracy')
                ax.axhline(0.5, ls=':', c='grey', zorder=-20)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend().remove()
                ax.set_ylim([0.3, 1])
                sns.despine()
                plt.tight_layout()
                plt.show()
                plot_name = 'new_violins_{}_t{}.{}'.format(deconf_method, response_time_bin_index, plot_format)
                f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                          bbox_inches="tight")

                f, ax = plt.subplots(1, 1)
                labels_mod = [cr_labels_detailed[labels[0]], '{}\nShuffled $C$'.format(cr_labels_detailed[labels[1]])]
                ax.legend(handles, labels_mod, frameon=False)
                ax.axis('off')
                plt.show()
                plot_name = 'new_violins_legend_{}.{}'.format(deconf_method, plot_format)
                f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                          bbox_inches="tight")

        if plot_barplots:
            # PLOT BARPLOTS (FIG. 6)
            dssel = ds[ds['n_components'] == n_pca_components_main_plot]

            for deconf_method in selected_methods:

                f, ax = plt.subplots(1, 1, figsize=[3.5, 3], sharey=True)
                dssel_raw = dssel[(dssel['deconf_method'] == 'raw') &
                                  (dssel['time'] == time) &
                                  (dssel['shuffledC'] == False)].copy()

                dssel_met = dssel[(dssel['deconf_method'] == deconf_method) &
                              (dssel['time'] == time) &
                              (dssel['shuffledC'] == True)].copy()

                dsvplot = pd.concat([dssel_raw, dssel_met])

                sns.barplot(data=dsvplot, x='decoder',
                            palette=method_palette,
                            y='perc_sig', hue='deconf_method',
                            hue_order=['raw', deconf_method], ax=ax, fill=False)
                ax.set_xticklabels([decoder_labels[d] for d in decoders])

                nraw = []
                nmet = []
                for kk, decoder in enumerate(decoders):
                    dssel_raw = dssel[(dssel['deconf_method'] == 'raw') &
                                      (dssel['time'] == time) &
                                      (dssel['shuffledC'] == False) &
                                      (dssel['decoder'] == decoder)].copy()
                    dssel_met = dssel[(dssel['deconf_method'] == deconf_method) &
                                      (dssel['time'] == time) &
                                      (dssel['shuffledC'] == True) &
                                      (dssel['decoder'] == decoder)].copy()
                    assert dssel_raw.shape[0] == 1
                    assert dssel_met.shape[0] == 1

                    nsig_raw = dssel_raw['N_sig'].iloc[0]
                    nsig_met = dssel_met['N_sig'].iloc[0]
                    print(nsig_raw, nsig_met)
                    nraw.append(nsig_raw)
                    nmet.append(nsig_met)

                nsig = nraw+nmet
                #indx = np.argsort([p.xy[0] for p in ax.patches[0:len(ns)]])
                for ii, bar in enumerate(ax.patches[0:len(nsig)]):
                    ax.text(bar.get_x()+bar._width/2, 20, '$N={}$'.format(nsig[ii]), color='k',
                            ha='center', va='center',
                            rotation='vertical')
                ax.set_ylabel('% significant')
                ax.set_xlabel('')
                sns.despine()
                handles, labels = ax.get_legend_handles_labels()
                ax.legend().remove()
                ax.set_ylim([0, 100])
                plt.tight_layout()
                plt.show()
                plot_name = 'new_barplots_{}_t{}.{}'.format(deconf_method, response_time_bin_index, plot_format)
                f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                          bbox_inches="tight")

                f, ax = plt.subplots(1, 1)
                labels_mod = [cr_labels_detailed[labels[0]], '{}\nShuffled $C$'.format(cr_labels_detailed[labels[1]])]
                ax.legend(handles, labels_mod, frameon=False)
                ax.axis('off')
                plt.show()
                plot_name = 'new_barplots_legend_{}.{}'.format(deconf_method, plot_format)
                f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                          bbox_inches="tight")


        deconf_methods_leg = [d for d in deconf_methods if d!='decode_from_confound']
        # f, ax = plt.subplots(1, 1, figsize=[2, 2])
        # legend_elements = [plt.Line2D([0], [0], marker='o', color=method_palette[m],
        #                               label=cr_labels_main[m], lw=0,
        #                               markerfacecolor=method_palette[m],
        #                               markeredgecolor='w',
        #                               markersize=10) for m in deconf_methods_leg]
        # ax.legend(handles=legend_elements, loc='center', frameon=False,
        #           ncol=1)
        # sns.despine(left=True, bottom=True)
        # ax.axis('off')
        # plt.show()
        # plot_name = 'CR_legend_dots.{}'.format(plot_format)
        # f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
        #           bbox_inches="tight")

        # f, ax = plt.subplots(1, 1, figsize=[2, 2])
        # legend_elements = [plt.Line2D([0], [0], marker='o', color=method_palette[m],
        #                               label=cr_labels_main[m], lw=8,
        #                               markerfacecolor=method_palette[m],
        #                               markeredgecolor='w',
        #                               markersize=0) for m in deconf_methods_leg]
        # ax.legend(handles=legend_elements, loc='center', frameon=False,
        #           ncol=1)
        # sns.despine(left=True, bottom=True)
        # ax.axis('off')
        # plt.show()
        # plot_name = 'CR_legend_barplot.{}'.format(plot_format)
        # f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
        #           bbox_inches="tight")
        #
        # f, ax = plt.subplots(1, 1, figsize=[2, 2])
        # legend_elements = [plt.Line2D([0], [0], marker='o', color='black',
        #                               label='$p<{}$'.format(alpha_level_permutations), lw=0,
        #                               markerfacecolor='black', markersize=10, markeredgewidth=2),
        #                    plt.Line2D([0], [0], marker='o', color='black',
        #                               label="Not sig.".format(alpha_level_permutations), lw=0,
        #                               markerfacecolor='white', markersize=10, markeredgewidth=2)]
        # ax.legend(handles=legend_elements, loc='center', frameon=False,
        #           ncol=1)
        # sns.despine(left=True, bottom=True)
        # ax.axis('off')
        # plt.show()
        # plot_name = 'significant_legend.{}'.format(plot_format)
        # f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
        #           bbox_inches="tight")


        if plot_score_barplots:
            # PLOT BARPLOTS (FIG. 5)
            ds = dfscore.copy()
            ds = ds[ds['n_components'] == 30]

            for score in ['TPR_TNR_av', 'TPR', 'TNR', 'FPR']:
                f, ax = plt.subplots(1, len(selected_methods), sharey=True,
                                     sharex=True, figsize=[1.5*len(selected_methods), 3])
                if len(selected_methods) == 1:
                    ax = np.array([ax])
                for i, deconf_method in enumerate(selected_methods):
                    palette = sns.dark_palette(method_palette[deconf_method], n_colors=len(decoders))
                    decoder_palette = {decoder: palette[i] for i, decoder in enumerate(decoders)}
                    sns.barplot(data=ds[ds['deconf_method'] == deconf_method],
                                x='decoder', order=decoders, y=score, errorbar=None, dodge=True,
                                color=method_palette[deconf_method], ax=ax[i], gap=0.05)
                    for bars in ax[i].containers:
                        for bar, decoder in zip(bars, decoders):
                            bar.set_facecolor(decoder_palette[decoder])

                    ax[i].set_xlabel('')
                    ax[i].set_xticklabels([decoder_labels[s] for s in decoders], rotation=90)
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

        if plot_score_lineplots:
            # PLOT FPR/TPR/BAL.ACC. VS. # PCs
            if dfscore['n_components'].unique().shape[0] > 1:
                dn = dfscore.copy()
                for score in ['TPR_TNR_av', 'FPR', 'TPR']:
                    f, ax = plt.subplots(1, len(selected_methods), sharey=True,
                                         sharex=True, figsize=[2*len(selected_methods), 3])
                    if len(selected_methods) == 1:
                        ax = [ax]
                    for kk, deconf_method in enumerate(selected_methods):
                        palette = sns.dark_palette(method_palette[deconf_method], n_colors=len(decoders))
                        decoder_palette = {decoder: palette[i] for i, decoder in enumerate(decoders)}
                        for decoder in decoders:
                            fsc = dn[(dn['deconf_method'] == deconf_method) &
                                     (dn['decoder'] == decoder)]
                            x = fsc['n_components'].values.astype(float)
                            vals = fsc[score].values.astype(float)

                            ax[kk].plot(x, vals, c=decoder_palette[decoder],
                                        marker=decoder_markers[decoder],
                                        markerfacecolor='w', markeredgecolor=decoder_palette[decoder],
                                        label=decoder_labels[decoder])
                            # ax[kk].scatter(x, vals,
                            #                marker=mode_markers[mode],
                            #                facecolor='w', edgecolor=mode_palette[mode],
                            #                label=calibration_modes_labels[mode])
                        ax[kk].axhline(0.5, ls=':', c='grey', zorder=-10)
                        ax[kk].set_ylim([0, 1.01])
                        ax[kk].set_yticks(np.arange(0, 1.1, 0.2))
                        ax[kk].set_xticks(x)
                        ax[kk].set_xlim([x[0] - 3, x[-1] + 3])
                        ax[kk].set_xlabel('# movement PCs')
                        ax[kk].set_title(cr_labels_detailed[deconf_method], fontsize=10)
                        sns.despine()

                    ax[0].legend()
                    handles_lin, labels_lin = ax[0].get_legend_handles_labels()
                    ax[0].legend().remove()
                    if len(ax) > 1:
                        ax[1].legend()
                        handles_nonlin, labels_nonlin = ax[1].get_legend_handles_labels()
                        ax[1].legend().remove()
                    ax[0].set_ylabel(score_labels[score])
                    plt.tight_layout()
                    plt.show()
                    plot_name = '{}_vs_n_components.{}'.format(score, plot_format)
                    f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                              bbox_inches="tight")

                f, ax = plt.subplots(1, 2)
                ax[0].legend(handles_lin, labels_lin, frameon=False)
                ax[0].axis('off')
                ax[1].legend(handles_nonlin, labels_nonlin, frameon=False)
                ax[1].axis('off')
                plt.show()
                plot_name = 'legend_modes_n_comp_{}.{}'.format(settings_name_global, plot_format)
                f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                          bbox_inches="tight")

    elif mode == 'all_sessions_leakage' or mode == 'all_sessions_leakage_supplementary':
        # p_right indicates probability of improvement vs raw
        # testing on all sessions combined
        selected_methods = deconf_methods

        # COMPUTE LEAKAGE SINGLE PLOTS
        for decoder in decoders:
            dx = decav[(decav['decoder'] == decoder) &
                       (decav['shuffle_confound'] == False) &
                       (decav['shuffle_firing_rates'] == False) &
                       (decav['time_bin'] == response_time_bin_index)]
            f, ax = plt.subplots(1, 1, figsize=[0.7*len(deconf_methods), 3])

            if decode_from_confound_predictions:
                ax.set_ylabel('Mean accuracy\nDecoding from $X_C$')
            else:
                raise ValueError
            ax.axhline(0.5, ls=':', c='grey', zorder=-20)
            ax.set_ylim([0.45, 1])
            #ax.set_xlim([-0.4, 2.4])
            for kk, deconf_method in enumerate([d for d in deconf_methods if d!= 'raw']):
                row = ls[(ls['decoder']==decoder) & (ls['deconf_method'] == deconf_method)]
                if row['p_right_bayes'].iloc[0] >= (1-alpha_level):
                    pass
                    ax.scatter(kk+1, 1-0.018, marker='*', c='k', s=40)

            sns.swarmplot(data=dx, x='deconf_method', y='mean_score', ax=ax,
                          order=deconf_methods, size=6, palette=method_palette)
            ax.axvline(0.5, c='k', zorder=-5, linewidth=0.75)
            ax.set_xlabel('')
            if mode == 'all_sessions_leakage':
                ax.set_xticklabels([cr_labels_main[l._text] for l in ax.get_xticklabels()], rotation=90)
            elif mode == 'all_sessions_leakage_supplementary':
                ax.set_xticklabels([cr_labels_detailed[l._text] for l in ax.get_xticklabels()], rotation=90)
            sns.despine()
            plt.tight_layout()

            plot_name = 'decoding_scores_SWARM_{}.{}'.format(decoder, plot_format)
            f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                      bbox_inches="tight")


        # PLOT LEAKAGE COMBINED

        f, ax = plt.subplots(1, len(decoders), sharey=True,
                             figsize=[0.55 * len(selected_methods)*len(decoders), 3])

        for ij, decoder in enumerate(decoders):
            dx = decav[(decav['decoder'] == decoder) &
                       (decav['shuffle_confound'] == False) &
                       (decav['shuffle_firing_rates'] == False) &
                       (decav['time_bin'] == response_time_bin_index)]

            if decode_from_confound_predictions:
                ax[0].set_ylabel('Mean accuracy\nDecoding from $X_C$')
            else:
                raise ValueError
            ax[ij].axhline(0.5, ls=':', c='grey', zorder=-20)
            ax[ij].set_ylim([0.4, 1])
            #ax.set_xlim([-0.4, 2.4])
            for kk, deconf_method in enumerate([d for d in deconf_methods if d!= 'raw']):
                row = ls[(ls['decoder']==decoder) & (ls['deconf_method'] == deconf_method)]
                if row['p_right_bayes'].iloc[0] >= (1-alpha_level):
                    pass
                    ax[ij].scatter(kk+1, 1-0.018, marker='*', c='k', s=40)

            sns.swarmplot(data=dx, x='deconf_method', y='mean_score', ax=ax[ij],
                          order=selected_methods, size=6, palette=method_palette)
            ax[ij].axvline(0.5, c='k', zorder=-5, linewidth=0.75)
            ax[ij].set_xlabel('')
            ax[ij].set_xticklabels([cr_labels_detailed_nextline[l._text] for l in ax[ij].get_xticklabels()], rotation=90)
            ax[ij].set_title(decoder_labels[decoder], fontsize=10)
        sns.despine()
        plt.tight_layout()
        plt.show()

        plot_name = 'decoding_scores_SWARM_COMBINED.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")


        # p_right indicates probability of improvement vs raw
        # testing on individual sessions (scores for each model are the cross-validation scores)
        if leakage_in_individual_sessions:
            ls2 = pd.DataFrame(columns=['session_id', 'decoder', 'shuffle_firing_rates', 'deconf_method', 'p_right_bayes', 'is_sig', 'p_val_wilcoxon'])
            for decoder in decoders:
                for shuffle_firing_rates in shuffle_firing_rates_list:
                    for deconf_method in [i for i in deconf_methods if i!='raw']:
                        for session_id in session_ids:
                            ddeconf = dec_rs[(dec_rs['decoder'] == decoder) &
                                       (dec_rs['shuffle_confound'] == False) &
                                       (dec_rs['shuffle_firing_rates'] == shuffle_firing_rates) &
                                       (dec_rs['time_bin'] == response_time_bin_index) &
                                       (dec_rs['session_id'] == session_id) &
                                       (dec_rs['deconf_method'] == deconf_method)]

                            draw = dec_rs[(dec_rs['decoder'] == decoder) &
                                       (dec_rs['shuffle_confound'] == False) &
                                       (dec_rs['shuffle_firing_rates'] == shuffle_firing_rates) &
                                       (dec_rs['time_bin'] == response_time_bin_index) &
                                       (dec_rs['session_id'] == session_id) &
                                       (dec_rs['deconf_method'] == 'raw')]

                            scores_cr = ddeconf['mean_score']
                            scores_raw = draw['mean_score']
                            (p_left, p_right) = two_on_single(x=scores_raw, y=scores_cr, runs=1)
                            stat, p_w = wilcoxon(scores_raw, scores_cr, alternative='two-sided')
                            is_sig = p_right >= (1-alpha_level)
                            ls2.loc[ls2.shape[0], :] = [session_id, decoder, shuffle_firing_rates, deconf_method, p_right, is_sig, p_w]
            print(ls2)

            # COMPUTE LEAKAGE COMBINED ()
            for decoder in decoders:
                dx = decav[(decav['decoder'] == decoder) &
                           (decav['shuffle_confound'] == False) &
                           (decav['shuffle_firing_rates'] == False) &
                           (decav['time_bin'] == response_time_bin_index)]
                lx = ls2[(ls2['decoder'] == decoder) &
                           (ls2['shuffle_firing_rates'] == False)]

                dm = pd.merge(dx, lx)

                f, ax = plt.subplots(1, 1, figsize=[6, 4])
                sns.swarmplot(data=dx[dx['deconf_method'] == 'raw'], x='deconf_method', y='mean_score', ax=ax,
                              order=deconf_methods, size=10, palette=method_palette)
                sns.swarmplot(data=dm[dm['is_sig']], x='deconf_method', y='mean_score', ax=ax,
                              order=deconf_methods, size=10, palette=method_palette,
                              edgecolor='k', linewidth=2)
                sns.swarmplot(data=dm[~dm['is_sig']], x='deconf_method', y='mean_score', ax=ax,
                              order=deconf_methods, size=8, palette=method_palette)
                ax.set_xlabel('')
                if decode_from_confound_predictions:
                    ax.set_ylabel('Mean accuracy\nDecoding from $X_C$')
                else:
                    raise ValueError
                ax.axhline(0.5, ls=':', c='grey', zorder=-20)
                ax.set_xticklabels([cr_labels_detailed[l._text] for l in ax.get_xticklabels()], rotation=90)
                sns.despine()
                plt.tight_layout()
                plot_name = 'decoding_scores_individual_significance_SWARM_{}.{}'.format(decoder, plot_format)
                f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                          bbox_inches="tight")

    elif mode == 'all_sessions_leakage_shuffle_features' or mode == 'all_sessions_leakage_shuffle_features_supplementary':

        selected_methods = deconf_methods

        if mode == 'all_sessions_leakage_shuffle_features':
            ylim = [0.35, 0.75]
        elif mode == 'all_sessions_leakage_shuffle_features_supplementary':
            ylim = [0.35, 0.85]

        # COMPUTE LEAKAGE COMBINED
        for decoder in decoders:
            dx = decav[(decav['decoder'] == decoder) &
                       (decav['shuffle_confound'] == False) &
                       (decav['time_bin'] == response_time_bin_index)]
            f, ax = plt.subplots(1, 1, figsize=[0.7*len(deconf_methods), 3])

            ax.set_ylabel('Mean accuracy\nDecoding from $X_{shuffled}$')

            ax.axhline(0.5, ls=':', c='grey', zorder=-20)
            ax.set_ylim(ylim)
            ax.set_xlim([-0.4, 2.4])
            for kk, deconf_method in enumerate([d for d in deconf_methods if d != 'raw']):
                row = ls[(ls['decoder'] == decoder) & (ls['deconf_method'] == deconf_method)]
                if row['p_right_bayes'].iloc[0] >= (1 - alpha_level):
                    pass
                    ax.scatter(kk + 1, ylim[1] - 0.018, marker='*', c='k', s=40)

            sns.swarmplot(data=dx, x='deconf_method', y='mean_score', ax=ax,
                          order=deconf_methods, size=6, palette=method_palette)
            ax.set_xlabel('')
            ax.axvline(0.5, c='k', zorder=-5, linewidth=0.75)
            if mode == 'all_sessions_leakage_shuffle_features':
                ax.set_xticklabels([cr_labels_main[l._text] for l in ax.get_xticklabels()], rotation=90)
            elif mode == 'all_sessions_leakage_shuffle_features_supplementary':
                ax.set_xticklabels([cr_labels_detailed[l._text] for l in ax.get_xticklabels()], rotation=90)
            sns.despine()
            plt.tight_layout()
            plt.show()

            plot_name = 'decoding_scores_SWARM_{}.{}'.format(decoder, plot_format)
            f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                      bbox_inches="tight")



        # PLOT LEAKAGE COMBINED
        f, ax = plt.subplots(1, len(decoders), sharey=True,
                             figsize=[0.55 * len(selected_methods)*len(decoders), 3])

        for ij, decoder in enumerate(decoders):
            dx = decav[(decav['decoder'] == decoder) &
                       (decav['shuffle_confound'] == False) &
                       (decav['time_bin'] == response_time_bin_index)]

            ax[ij].axhline(0.5, ls=':', c='grey', zorder=-20)
            ax[ij].set_ylim(ylim)
            #ax.set_xlim([-0.4, 2.4])
            for kk, deconf_method in enumerate([d for d in deconf_methods if d!= 'raw']):
                row = ls[(ls['decoder']==decoder) & (ls['deconf_method'] == deconf_method)]
                if row['p_right_bayes'].iloc[0] >= (1-alpha_level):
                    pass
                    ax[ij].scatter(kk+1,  ylim[1]-0.021, marker='*', c='k', s=40)

            sns.swarmplot(data=dx, x='deconf_method', y='mean_score', ax=ax[ij],
                          order=selected_methods, size=6, palette=method_palette)
            ax[ij].axvline(0.5, c='k', zorder=-5, linewidth=0.75)
            ax[ij].set_xlabel('')
            ax[ij].set_xticklabels([cr_labels_detailed_nextline[l._text] for l in ax[ij].get_xticklabels()], rotation=90)
            ax[ij].set_title(decoder_labels[decoder], fontsize=10)
        ax[0].set_ylabel('Mean accuracy\nDecoding from $X_{shuffled}$')
        sns.despine()
        plt.tight_layout()
        plt.show()

        plot_name = 'decoding_scores_SWARM_COMBINED.{}'.format(plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")

