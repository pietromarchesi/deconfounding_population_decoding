import os
import pickle
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.metrics import balanced_accuracy_score, mean_poisson_deviance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDClassifier, PoissonRegressor
from constants import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_poisson_deviance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from decoding_utils import *
from plotting_style import *
from session_selection import selected_sessions_auditory
from sklearn.svm import SVR
import warnings

"""
Compute confound regression on V1 data
"""

settings_name_global = 'OCT_9_DECODE_FROM_CONF_NOTUNE'

# V1 dataset parameters
experiment = 'audiofreq_in_big_change'
data_settings_name = 'sep13_2026'
ephys_dataset = 'modid'
area_spikes_dataset = 'V1'

# ------------------------------------------------------------------------

# all available modes of running CR
all_modes = ['all_sessions_significance_n_components_notune_cr_notune_dec',
             'all_sessions_significance_n_components_tune_cr_notune_dec',
             'all_sessions_significance_n_components_notune_cr_tune_dec',
             'all_sessions_leakage',
             'all_sessions_leakage_shuffle_features',
             'all_sessions_leakage_supplementary',
             'all_sessions_leakage_shuffle_features_supplementary',
             'all_sessions_decode_from_confound_notune_dec',
             'all_sessions_decode_from_confound_tune_dec']

# -------------------------------------------------------------------------

# selected mode
modes = ['all_sessions_decode_from_confound_notune_dec']

# CR variants
deconf_methods_main = ['linear_confound_regression',
                       'nonlinear_confound_regression_rf']

# CR variants for supplementary figures
# applies only to
# -'all_sessions_leakage_supplementary'
# -'all_sessions_leakage_shuffle_features_supplementary'
deconf_methods_supplementary = ['linear_confound_regression_glm',
                               'nonlinear_confound_regression_svc']

# -------------------------------------------------------------------------

for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)

    if data_settings_name == 'aug22':
        time_bin_index = 3
    elif data_settings_name == 'sep13_2026':
        time_bin_index = 13

    if mode == 'all_sessions_significance_n_components_notune_cr_notune_dec':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        shuffle_components_list = [False, True]
        shuffle_firing_rates_list = [False]
        decode_from_confound_predictions = False
        deconf_methods = deconf_methods_main #+ ['raw'] # TODO commented out, arleady ran
        if data_settings_name == 'aug22':
            n_pca_components = [10, 20, 30]
        elif data_settings_name == 'sep13_2026':
            n_pca_components = [10, 20, 30, 40, 50]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = False

    elif mode == 'all_sessions_significance_n_components_tune_cr_notune_dec':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        shuffle_components_list = [False, True]
        shuffle_firing_rates_list = [False]
        decode_from_confound_predictions = False
        deconf_methods = ['raw'] + deconf_methods_main
        if data_settings_name == 'aug22':
            n_pca_components = [10, 20, 30]
        elif data_settings_name == 'sep13_2026':
            n_pca_components = [10, 20, 30, 40, 50]
        tune_hyperparameters_cr = True
        tune_hyperparameters_dec = False

    elif mode == 'all_sessions_significance_n_components_notune_cr_tune_dec':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        shuffle_components_list = [False, True]
        shuffle_firing_rates_list = [False]
        decode_from_confound_predictions = False
        deconf_methods = ['raw'] + deconf_methods_main
        if data_settings_name == 'aug22':
            n_pca_components = [10, 20, 30]
        elif data_settings_name == 'sep13_2026':
            n_pca_components = [10, 20, 30, 40, 50]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = True

    elif mode == 'all_sessions_decode_from_confound_notune_dec':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 0
        shuffle_components_list = [False]
        shuffle_firing_rates_list = [False]
        decode_from_confound_predictions = False
        deconf_methods = ['decode_from_confound']
        n_pca_components = [10, 20, 30, 40, 50]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = False

    elif mode == 'all_sessions_decode_from_confound_tune_dec':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 0
        shuffle_components_list = [False]
        shuffle_firing_rates_list = [False]
        decode_from_confound_predictions = False
        deconf_methods = ['decode_from_confound']
        n_pca_components = [10, 20, 30, 40, 50]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = True

    elif mode == 'all_sessions_leakage':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 0
        shuffle_components_list = [False, True]
        shuffle_firing_rates_list = [False]
        decode_from_confound_predictions = True
        deconf_methods = ['raw'] + deconf_methods_main
        n_pca_components = [30]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = False

    elif mode == 'all_sessions_leakage_supplementary':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 0
        shuffle_components_list = [False, True]
        shuffle_firing_rates_list = [False]
        decode_from_confound_predictions = True
        deconf_methods = ['raw'] + deconf_methods_supplementary
        n_pca_components = [30]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = False

    elif mode == 'all_sessions_leakage_shuffle_features':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 0
        shuffle_components_list = [False]
        shuffle_firing_rates_list = [True]
        decode_from_confound_predictions = False
        deconf_methods = ['raw'] + deconf_methods_main
        n_pca_components = [30]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = False

    elif mode == 'all_sessions_leakage_shuffle_features_supplementary':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 0
        shuffle_components_list = [False]
        shuffle_firing_rates_list = [True]
        decode_from_confound_predictions = False
        deconf_methods = ['raw'] + deconf_methods_supplementary
        n_pca_components = [30]
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = False

    remove_low_firing = False
    shuffle_target = False
    random_seed = 0
    compute_r2 = False

    n_kfold_splits = 5
    standardize_firing_rates = True
    standardize_confound = True

    # CONFOUND REGRESSION PARAMETERS
    n_fold_cr_hyperpar_tuning = 3

    cr_fixed_params = {'rf_max_depth': 4,
                       'rf_n_estimators': 50,
                       'ridge_alpha': 1,
                       'glm_max_iter' : 4000,
                       'svr_c': 1,
                       'glm_alpha': 0.1}

    cr_hyperpars_grids = {'ridge': [0.001, 0.01, 0.1, 1, 10],
                          'glm': [0.001, 0.01, 0.1, 1, 10],
                          'svr': [0.001, 0.01, 0.1, 1, 10],
                          'rf': [3, 4, 5, 6, 7]}

    # DECODER PARAMETERS
    n_fold_decoder_tuning = 5

    decoders_fixed_params = {'lr_C': 1,
                             'lr_max_iter' : 3000,
                             'svc_C': 1,
                             'rf_max_depth': 4,
                             'rf_n_estimators': 50}

    decoders_hyperpars_grids = {'lr': [0.001, 0.01, 0.1, 1, 10],
                                'svc': [0.001, 0.01, 0.1, 1, 10],
                                'rf_max_depth': [4, 6, 8],
                                'rf_n_estimators' : [50, 100]}

    decoders = ['logistic_regression',
                'SVC',
                'random_forest']

    data = load_decoding_data(settings_name=data_settings_name,
                              experiment_name=experiment,
                              dataset=ephys_dataset,
                              area_spikes=area_spikes_dataset)


    for deconf_method in deconf_methods:
        assert np.isin(deconf_method, cr_methods)

        cr_rs = pd.DataFrame(columns=['session_id', 'deconf_method',
                                      'n_pca_components', 'n_trials', 'time', 'time_bin', 't0',
                                      't1', 'shuffle_confound', 'shuffle_firing_rates',
                                      'fold', 'neuron', 'permuted', 'perm_index', 'mean_score',
                                      'poisson_regressor_score'])

        dec_rs = pd.DataFrame(columns=['session_id', 'deconf_method', 'decoder',
                                       'n_pca_components', 'n_trials', 'time', 'time_bin', 't0', 't1',
                                       'shuffle_confound', 'shuffle_firing_rates',
                                       'permuted', 'perm_index', 'mean_score', 'fold'])

        for shuffle_components in shuffle_components_list:

            for shuffle_firing_rates in shuffle_firing_rates_list:

                n_trials_dict = {}
                for session_id in session_ids:

                    time_bin_times = data['data'][session_id]['time_bin_times']
                    time_bins = data['data'][session_id]['time_bins']
                    n_time_bins_per_trial = len(time_bin_times)

                    if compute_all_time_bins:
                        time_bin_indexes = np.arange(n_time_bins_per_trial)
                    else:
                        time_bin_indexes = [time_bin_index]

                    for n_components in n_pca_components:
                        for time_bin_indx in time_bin_indexes:
                            time = time_bin_times[time_bin_indx]
                            t0, t1 = time_bins[time_bin_indx]

                            X_full = data['data'][session_id]['binned_spikes'][time_bin_indx]
                            C_full = data['data'][session_id]['binned_movement'][time_bin_indx]
                            y_full = data['data'][session_id]['target']
                            total_n_trials = X_full.shape[0]

                            print('Session {}, t={}, {} - shuffle C: {}, shuffle X: {}, #PCA: {}'.format(session_id, time, cr_labels_detailed[deconf_method],
                                                                                               shuffle_components, shuffle_firing_rates, n_components))
                            X = X_full.copy()
                            C = C_full.copy()
                            y = y_full.copy()

                            if shuffle_components:
                                indx = np.random.default_rng(seed=random_seed).permutation(C.shape[0])
                                C = C[indx, 0:n_components]
                            else:
                                C = C[:, 0:n_components]

                            if shuffle_firing_rates:
                                for ix in range(X.shape[1]):
                                    indx = np.random.default_rng(seed=ix).permutation(X.shape[0])
                                    X[:, ix] = X[indx, ix]

                            if shuffle_target:
                                assert deconf_method == 'raw'
                                assert not shuffle_components
                                y = np.random.default_rng(seed=random_seed).permutation(y)

                            for nperm in range(n_permutations+1):
                                if nperm == 0:
                                    C_dec = np.copy(C)
                                    y_dec = np.copy(y)
                                    permuted = False
                                elif nperm > 0:
                                    random_index = np.random.default_rng(seed=nperm).permutation(np.arange(X.shape[0]))
                                    # only if method is predictability we permute C together with y
                                    C_dec = np.copy(C)
                                    y_dec = y[random_index]
                                    permuted = True

                                if nperm % 50 == 0 and nperm > 0:
                                    print('--- n_perms = {}'.format(nperm))

                                # run the decoding
                                kfold = StratifiedKFold(n_splits=n_kfold_splits, shuffle=True,
                                                        random_state=random_seed)
                                iterable = kfold.split(X, y_dec)

                                kfold_scores = []
                                y_test_all, y_pred_all, c_all = [], [], []

                                for fold, (training_ind, testing_ind) in enumerate(iterable):

                                    # decoder = SGDClassifier(random_state=92)
                                    X_train = X[training_ind, :].copy()
                                    X_test = X[testing_ind, :].copy()
                                    y_train = y_dec[training_ind]
                                    y_test = y_dec[testing_ind]
                                    C_train = C_dec[training_ind, :]
                                    C_test = C_dec[testing_ind, :]

                                    if standardize_firing_rates:
                                        if np.isin(deconf_method, ['linear_confound_regression_glm', 'linear_confound_regression_glm_tuned']):
                                            warnings.warn('Skipping standardization of firing rates because '
                                                          'the select method is the GLM')
                                        else:
                                            ss = StandardScaler()
                                            X_train = ss.fit_transform(X_train)
                                            X_test = ss.transform(X_test)

                                    if standardize_confound:
                                        ss = StandardScaler()
                                        C_train = ss.fit_transform(C_train)
                                        C_test = ss.transform(C_test)

                                    if remove_low_firing and np.isin(deconf_method, ['linear_confound_regression_glm',
                                                                                     'linear_confound_regression_glm_tuned']):
                                        min_nonzero = 20 * total_n_trials /100
                                        indx_remove = np.where(np.count_nonzero(X, axis=0)<min_nonzero)[0]
                                        indx_keep = [ix for ix in np.arange(X.shape[1]) if ~np.isin(ix, indx_remove)]
                                        X_train = X_train[:, indx_keep]
                                        X_test = X_test[:, indx_keep]

                                        # ss = MinMaxScaler(feature_range=(0, 1))
                                        # X_train = ss.fit_transform(X_train)
                                        # X_test = ss.transform(X_test)
                                    if not standardize_firing_rates:
                                        assert (X_train<0).sum() == 0
                                        assert (X_test<0).sum() == 0

                                    # Apply confound regression (linear regression)
                                        # for high-dimensional confounds, avoid overfitting on the
                                        # train set
                                    new_X_train = []
                                    new_X_test = []
                                    for ix in range(X_train.shape[1]):

                                        # PREPARE THE MODELS
                                        if deconf_method == 'linear_confound_regression':
                                            model = LinearRegression()

                                        elif deconf_method == 'linear_confound_regression_ridge':
                                            parameters_cr = {'alpha': cr_hyperpars_grids['ridge']}
                                            model = Ridge(alpha=cr_fixed_params['ridge_alpha'], fit_intercept=True)

                                        elif deconf_method == 'linear_confound_regression_glm':
                                            parameters_cr = {'alpha': cr_hyperpars_grids['glm']}
                                            model = PoissonRegressor(max_iter=cr_fixed_params['glm_max_iter'],
                                                                     alpha=cr_fixed_params['glm_alpha'])

                                        elif deconf_method == 'nonlinear_confound_regression_svc':
                                            parameters_cr = {'C': cr_hyperpars_grids['svr']}
                                            model = SVR(C=cr_fixed_params['svr_c'], kernel='rbf')

                                        elif deconf_method == 'nonlinear_confound_regression_rf':
                                            parameters_cr = {'max_depth': cr_hyperpars_grids['rf']}
                                            model = RandomForestRegressor(n_estimators=cr_fixed_params['rf_n_estimators'],
                                                                            random_state=fold,
                                                                            max_depth=cr_fixed_params['rf_max_depth'])

                                        if tune_hyperparameters_cr:
                                            if deconf_method == 'linear_confound_regression':
                                                # no hyperparameters to tune
                                                pass
                                            else:
                                                # print('Tuning hyperparameters of CR: {}'.format(deconf_method))
                                                inner_cv_1 = KFold(n_splits=n_fold_cr_hyperpar_tuning, shuffle=True,
                                                                   random_state=fold)
                                                model = GridSearchCV(model, parameters_cr, refit=True, scoring='r2',
                                                                     cv=inner_cv_1)

                                        if deconf_method == 'raw':
                                            pass
                                        elif deconf_method == 'decode_from_confound':
                                            pass
                                        else:
                                            # PREDICTIONS TO REGRESS OUT
                                            model = model.fit(X=C_train, y=X_train[:, ix])

                                            new_X_train.append(model.predict(C_train).reshape(-1, 1))
                                            new_X_test.append(model.predict(C_test).reshape(-1, 1))

                                            # EVALUATE R2
                                            if compute_r2:
                                                y_pred = model.predict(C_test).reshape(-1, 1)
                                                y_null = np.tile(X_train[:, ix].mean(), X_test.shape[0])
                                                mask = (y_pred > 0).flatten()
                                                if mask.sum() == 0:
                                                    # if there are no positive values in y_pred, the score cannot
                                                    # be computed
                                                    y_pred = y_pred + 1e-6
                                                    y_null = y_null + 1e-6
                                                    mask = (y_pred > 0).flatten()
                                                if mask.sum() == 0:
                                                    score = 0
                                                else:
                                                    D_pred = mean_poisson_deviance(y_true=X_test[mask, ix], y_pred=y_pred[mask])
                                                    D_null = mean_poisson_deviance(y_true=X_test[mask, ix], y_pred=y_null[mask])
                                                    score = 1 - D_pred / D_null
                                                poisson_regressor_score = np.nan
                                            else:
                                                score = np.nan
                                                poisson_regressor_score = np.nan

                                            cr_rs.loc[cr_rs.shape[0], :] = [session_id, deconf_method, n_components,
                                                                            total_n_trials, time, time_bin_indx, t0, t1,
                                                                            shuffle_components, shuffle_firing_rates, fold, ix,
                                                                            permuted, nperm, score, poisson_regressor_score]

                                    # REGRESS OUT
                                    if deconf_method == 'raw':
                                        # if 'raw', decode_from_confound_predictions is ignored
                                        pass

                                    elif deconf_method == 'decode_from_confound':
                                        X_train = C_train
                                        X_test = C_test

                                    else:
                                        if decode_from_confound_predictions:
                                            X_train = np.hstack(new_X_train)
                                            X_test = np.hstack(new_X_test)
                                        else:
                                            X_train = X_train - np.hstack(new_X_train)
                                            X_test = X_test - np.hstack(new_X_test)

                                    # DECODE
                                    for decoder in decoders:
                                        if decoder == 'random_forest':
                                            parameters_dec = {'max_depth': decoders_hyperpars_grids['rf_max_depth'],
                                                              'n_estimators' : decoders_hyperpars_grids['rf_n_estimators']}
                                            model = RandomForestClassifier(n_estimators=decoders_fixed_params['rf_n_estimators'],
                                                                            random_state=fold,
                                                                            max_depth=decoders_fixed_params['rf_max_depth'])

                                        elif decoder == 'logistic_regression':
                                            parameters_dec = {'C': decoders_hyperpars_grids['lr']}
                                            model = LogisticRegression(C=decoders_fixed_params['lr_C'], penalty='l2',
                                                                       max_iter=decoders_fixed_params['lr_max_iter'],
                                                                       random_state=fold)

                                        elif decoder == 'SVC':
                                            parameters_dec = {'C': decoders_hyperpars_grids['svc']}
                                            model = SVC(C=decoders_fixed_params['svc_C'], probability=True,
                                                        gamma='scale', kernel='rbf', random_state=fold)

                                        if tune_hyperparameters_dec:
                                            inner_cv = StratifiedKFold(n_splits=n_fold_decoder_tuning, shuffle=True,
                                                                       random_state=fold)
                                            model = GridSearchCV(model, parameters_dec, refit=True,
                                                                 scoring=None, cv=inner_cv)

                                        model.fit(X_train, y_train)
                                        y_pred = model.predict(X_test)

                                        acc_score = balanced_accuracy_score(y_test, y_pred)

                                        dec_rs.loc[dec_rs.shape[0], :] = [session_id, deconf_method, decoder, n_components,
                                                                          total_n_trials, time, time_bin_indx, t0, t1,
                                                                          shuffle_components, shuffle_firing_rates,
                                                                          permuted, nperm, acc_score, fold]



        pars = {'experiment' : experiment,
                'data_settings_name' : data_settings_name,
                'ephys_dataset' : ephys_dataset,
                'area_spikes_dataset' : area_spikes_dataset,
                'compute_all_time_bins' : compute_all_time_bins,
                'n_permutations' : n_permutations,
                'time_bin_index' : time_bin_index,
                'shuffle_components_list' : shuffle_components_list,
                'n_pca_components' : n_pca_components,
                'remove_low_firing' : remove_low_firing,
                'shuffle_target' : shuffle_target,
                'n_kfold_splits' : n_kfold_splits,
                'standardize_firing_rates' : standardize_firing_rates,
                'standardize_confound' : standardize_confound,
                'random_seed' : random_seed,
                'deconf_methods' : deconf_methods,
                'shuffle_firing_rates_list' : shuffle_firing_rates_list,
                'decode_from_confound_predictions' : decode_from_confound_predictions,

                'n_fold_cr_tuning' : n_fold_cr_hyperpar_tuning,
                'cr_fixed_params': cr_fixed_params,
                'cr_hyperpars_grids': cr_hyperpars_grids,

                'decoders': decoders,
                'n_fold_decoder_tuning': n_fold_decoder_tuning,
                'decoders_fixed_params': decoders_fixed_params,
                'decoders_hyperpars_grids': decoders_hyperpars_grids,

                'tune_hyperparameters_cr': tune_hyperparameters_cr,
                'tune_hyperparameters_dec': tune_hyperparameters_dec,

                'time_bin_times': time_bin_times,
                'time_bin_indexes' : time_bin_indexes,
                'time_bins': time_bins,
                'n_time_bins_per_trial': n_time_bins_per_trial,
                'mode' : mode,
                'compute_r2' : compute_r2}


        out = {'pars' : pars,
               'cr_rs' : cr_rs,
               'dec_rs' : dec_rs}

        output_folder = os.path.join(DATA_PATH, 'ephys_data_CR_results',
                                     'SIMULATION_{}'.format(settings_name))

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        output_full_path = os.path.join(output_folder, 'ephys_data_CR_results_{}_{}.pkl'.format(settings_name, deconf_method))
        print('Saving simulation to {}'.format(output_full_path))
        pickle.dump(out, open(output_full_path, 'wb'))



