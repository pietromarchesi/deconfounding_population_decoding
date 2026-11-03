import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer, log_loss
from decoding_utils import *
from plotting_style import *
from session_selection import selected_sessions_auditory
from sklearn.calibration import CalibratedClassifierCV
from venn_abers import VennAbersCalibrator
import sklearn as skl

"""
Compute predictability on V1 data
"""

settings_name_global = 'OCT9_NEWDATA'

modes = ['all_sessions_significance_nocalib_notune',
         'all_sessions_significance_calib_isotonic',
         'all_sessions_significance_calib_vennabers',
         'all_sessions_significance_calib_sigmoid',
         'all_sessions_significance_nocalib_tune']

deconf_methods = ['predictability_lin',
                  'predictability_nonlin_svc',
                  'predictability_nonlin_rf',
                  'predictability_nonlin_gb']

# V1 dataset parameters
data_settings_name = 'sep13_2026'
experiment = 'audiofreq_in_big_change'
ephys_dataset = 'modid'
area_spikes_dataset = 'V1'


# CALIBRATION PARS
inner_cv_calibration_nsplits = 5

# DECODING PARS
n_kfold_splits = 5
standardize_features = True
standardize_confound = True

# HYPERPARAMETERS
random_seed = 0
shuffle_components_list = [False, True]
inner_cv_hyperpar_tune_nsplits = 5

pr_fixed_pars = {'lr_penalty' : 'l2',
                 'lr_solver' : 'saga',
                 'lr_max_iter' : 6000,
                 'lr_C_notune' : 1,
                 'svc_C_notune' : 0.1,
                 'svc_kernel' : 'rbf',
                 'rf_n_estimators' : 50,
                 'rf_max_depth' : 4,
                 'gb_n_estimators': 50,
                 'gb_max_depth': 4}

hyperpars_grids = {'predictability_lin' : [0.001, 0.01, 0.1, 1, 10],
                  'predictability_nonlin_svc' : [0.001, 0.01, 0.1, 1, 10],
                   'predictability_nonlin_rf' : [3, 4, 5, 6, 7],
                   'predictability_nonlin_gb' : [3, 4, 5, 6, 7]}

data = load_decoding_data(settings_name=data_settings_name,
                          experiment_name=experiment,
                          dataset=ephys_dataset,
                          area_spikes=area_spikes_dataset)

if skl.__version__ == '1.7.0':
    log_loss_scorer = make_scorer(log_loss, response_method='predict_proba', greater_is_better=False)
else:
    log_loss_scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)

for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)

    if data_settings_name == 'aug22':
        time_bin_index = 3
        n_pca_components = [10, 20, 30]

    elif data_settings_name == 'sep13_2026':
        time_bin_index = 13
        n_pca_components = [10, 20, 30, 40, 50]

    if mode == 'all_sessions_significance_nocalib_notune':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        tune_predictability = False
        calibrate = False
        calibrate_ensemble = False
        calibration_method = None

    elif mode == 'all_sessions_significance_nocalib_tune':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        tune_predictability = True
        calibrate = False
        calibrate_ensemble = False
        calibration_method = None

    elif mode == 'all_sessions_significance_calib_sigmoid':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        tune_predictability = False
        calibrate = True
        calibrate_ensemble = True
        calibration_method = 'sigmoid'

    elif mode == 'all_sessions_significance_calib_isotonic':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        tune_predictability = False
        calibrate = True
        calibrate_ensemble = True
        calibration_method = 'isotonic'

    elif mode == 'all_sessions_significance_calib_vennabers':
        compute_all_time_bins = False
        session_ids = selected_sessions_auditory
        n_permutations = 200
        tune_predictability = False
        calibrate = True
        calibrate_ensemble = True
        calibration_method = 'venn-abers'

    for deconf_method in deconf_methods:

        rs = pd.DataFrame(columns=['session_id', 'deconf_method', 'decoder',
                                   'n_pca_components', 'n_trials',
                                   'time', 'time_bin', 't0', 't1',
                                   'shuffle_confound', 'permuted', 'perm_index','predictability',
                                   'accuracy_full', 'accuracy_naive'])

        for shuffle_components in shuffle_components_list:

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

                        print('Mode: {} - Session {}, t={}, {} - shuffle C: {} - # components: {}'.format(mode, session_id, time,
                                                                            pr_labels_detailed[deconf_method],
                                                                            shuffle_components, n_components))
                        # get decoding data
                        X_full = data['data'][session_id]['binned_spikes'][time_bin_indx]
                        C_full = data['data'][session_id]['binned_movement'][time_bin_indx]
                        y_full = data['data'][session_id]['target']
                        total_n_trials = X_full.shape[0]
                        print(data['data'][session_id]['time_bins'][time_bin_indx,:])

                        X = X_full.copy()
                        C = C_full.copy()
                        y = y_full.copy()

                        if shuffle_components:
                            if n_components == 1:
                                C = np.random.default_rng(seed=random_seed).permutation(C[:, 0]).reshape(-1, 1)
                            else:
                                indx = np.random.default_rng(seed=random_seed).permutation(C.shape[0])
                                C = C[indx, 0:n_components]

                        else:
                            if n_pca_components == 1:
                                C = C[:, 0].reshape(-1, 1)
                            else:
                                C = C[:, 0:n_components]

                        for nperm in range(n_permutations+1):
                            if nperm == 0:
                                C_dec = np.copy(C)
                                y_dec = np.copy(y)
                                permuted = False
                            elif nperm > 0:
                                random_index =  np.random.default_rng(seed=nperm).permutation(np.arange(X.shape[0]))
                                # permute C together with y
                                if np.isin(deconf_method, predictability_methods):
                                    C_dec = C[random_index, :]
                                else:
                                    C_dec = np.copy(C)

                                y_dec = y[random_index]
                                permuted = True

                            if nperm % 10 == 0:
                                print('--- n_perms = {}'.format(nperm))

                            kfold = StratifiedKFold(n_splits=n_kfold_splits, shuffle=True,
                                                    random_state=10)
                            iterable = kfold.split(X, y_dec)

                            llratios = []
                            kfold_accuracy_full, kfold_accuracy_naive = [], []
                            y_pred_proba_full_list, y_pred_proba_naive_list = [], []

                            for fold, (training_ind, testing_ind) in enumerate(iterable):

                                X_train = X[training_ind, :]
                                X_test = X[testing_ind, :]
                                y_train = y_dec[training_ind]
                                y_test = y_dec[testing_ind]
                                C_train = C_dec[training_ind, :]
                                C_test = C_dec[testing_ind, :]

                                if standardize_features:
                                    ss = StandardScaler()
                                    X_train = ss.fit_transform(X_train)
                                    X_test = ss.transform(X_test)

                                if standardize_confound:
                                    ss = StandardScaler()
                                    C_train = ss.fit_transform(C_train)
                                    C_test = ss.transform(C_test)

                                features_full_train = np.hstack([X_train, C_train])
                                features_full_test = np.hstack([X_test, C_test])
                                features_naive_train = C_train
                                features_naive_test = C_test


                                if deconf_method == 'predictability_lin':
                                    parameters = {'C': hyperpars_grids['predictability_lin']}
                                    full_model = LogisticRegression(penalty=pr_fixed_pars['lr_penalty'],
                                                                    solver=pr_fixed_pars['lr_solver'],
                                                                    C=pr_fixed_pars['lr_C_notune'], random_state=fold,
                                                                    max_iter=pr_fixed_pars['lr_max_iter'])
                                    naive_model = LogisticRegression(penalty=pr_fixed_pars['lr_penalty'],
                                                                    solver=pr_fixed_pars['lr_solver'],
                                                                    C=pr_fixed_pars['lr_C_notune'], random_state=fold,
                                                                    max_iter=pr_fixed_pars['lr_max_iter'])

                                elif deconf_method == 'predictability_nonlin_svc':
                                    parameters = {'C': hyperpars_grids['predictability_nonlin_svc']}
                                    full_model = SVC(probability=True, C=pr_fixed_pars['svc_C_notune'],
                                                     kernel=pr_fixed_pars['svc_kernel'],
                                                     gamma='scale', random_state=fold)
                                    naive_model = SVC(probability=True, C=pr_fixed_pars['svc_C_notune'],
                                                      kernel=pr_fixed_pars['svc_kernel'],
                                                      gamma='scale', random_state=fold)

                                elif deconf_method == 'predictability_nonlin_rf':
                                    parameters = {'max_depth': hyperpars_grids['predictability_nonlin_rf']}
                                    full_model = RandomForestClassifier(n_estimators=pr_fixed_pars['rf_n_estimators'],
                                                                        random_state=fold,
                                                                        max_depth=pr_fixed_pars['rf_max_depth'])
                                    naive_model = RandomForestClassifier(n_estimators=pr_fixed_pars['rf_n_estimators'],
                                                                         random_state=fold,
                                                                         max_depth=pr_fixed_pars['rf_max_depth'])

                                elif deconf_method == 'predictability_nonlin_gb':
                                    parameters = {'max_depth': hyperpars_grids['predictability_nonlin_gb']}
                                    full_model = GradientBoostingClassifier(n_estimators=pr_fixed_pars['gb_n_estimators'],
                                                                            random_state=fold,
                                                                            max_depth=pr_fixed_pars['gb_max_depth'])
                                    naive_model = GradientBoostingClassifier(n_estimators=pr_fixed_pars['gb_n_estimators'],
                                                                             random_state=fold,
                                                                             max_depth=pr_fixed_pars['gb_max_depth'])

                                if tune_predictability:
                                    if calibrate:
                                        raise NotImplementedError('Tuning hyperparameters AND calibrating is not supported')
                                    inner_cv = StratifiedKFold(n_splits=inner_cv_hyperpar_tune_nsplits, shuffle=True, random_state=fold)
                                    full_model = GridSearchCV(full_model, parameters, refit=True, scoring=log_loss_scorer, cv=inner_cv)

                                    inner_cv = StratifiedKFold(n_splits=inner_cv_hyperpar_tune_nsplits, shuffle=True, random_state=fold)
                                    naive_model = GridSearchCV(naive_model, parameters, refit=True, scoring=log_loss_scorer, cv=inner_cv)

                                if calibrate:
                                    if calibration_method == 'venn-abers':
                                        full_model = VennAbersCalibrator(estimator=full_model, inductive=False,
                                                                         n_splits=inner_cv_calibration_nsplits,
                                                                         random_state=fold)
                                        naive_model = VennAbersCalibrator(estimator=naive_model, inductive=False,
                                                                          n_splits=inner_cv_calibration_nsplits,
                                                                          random_state=fold)

                                    else:
                                        calib_inner_cv = StratifiedKFold(n_splits=inner_cv_calibration_nsplits,
                                                                         shuffle=True,
                                                                         random_state=fold)
                                        full_model = CalibratedClassifierCV(full_model, cv=calib_inner_cv,
                                                                            ensemble=calibrate_ensemble,
                                                                            method=calibration_method)

                                        naive_model = CalibratedClassifierCV(naive_model, cv=calib_inner_cv,
                                                                            ensemble=calibrate_ensemble,
                                                                            method=calibration_method)


                                full_model.fit(features_full_train, y_train)
                                naive_model.fit(features_naive_train, y_train)

                                y_pred_proba_full = full_model.predict_proba(features_full_test)
                                y_pred_proba_naive = naive_model.predict_proba(features_naive_test)
                                y_pred_proba_full_list.append(y_pred_proba_full)
                                y_pred_proba_naive_list.append(y_pred_proba_naive)

                                if calibrate and calibration_method == 'venn-abers':
                                    y_pred_full = full_model.predict(features_full_test)[:, 1].astype(int)
                                    y_pred_naive = naive_model.predict(features_naive_test)[:, 1].astype(int)
                                else:
                                    y_pred_full = full_model.predict(features_full_test)
                                    y_pred_naive = naive_model.predict(features_naive_test)

                                fold_accuracy_full = balanced_accuracy_score(y_test, y_pred_full)
                                fold_accuracy_naive = balanced_accuracy_score(y_test, y_pred_naive)

                                kfold_accuracy_full.append(fold_accuracy_full)
                                kfold_accuracy_naive.append(fold_accuracy_naive)

                                log_likelihood_full = log2_likelihood(y_test, y_pred_proba_full)
                                log_likelihood_naive = log2_likelihood(y_test, y_pred_proba_naive)
                                llratio = log_likelihood_full - log_likelihood_naive
                                llratios.append(llratio)

                            accuracy_full = np.mean(kfold_accuracy_full)
                            accuracy_naive = np.mean(kfold_accuracy_naive)
                            predictability = np.mean(llratios)

                            y_pred_proba_full = np.vstack(y_pred_proba_full_list)
                            y_pred_proba_naive = np.vstack(y_pred_proba_naive_list)

                            rs.loc[rs.shape[0], :] = [session_id, deconf_method, 'none', n_components,
                                                      total_n_trials, time, time_bin_indx,  t0,  t1,
                                                      shuffle_components, permuted, nperm, predictability,
                                                      accuracy_full, accuracy_naive]

        pars = {'session_ids': session_ids,
                'data_settings_name': data_settings_name,
                'ephys_dataset': ephys_dataset,
                'area_spikes_dataset': area_spikes_dataset,
                'compute_all_time_bins': compute_all_time_bins,
                'shuffle_components_list' : shuffle_components_list,
                'time_bin_index': time_bin_index,
                'n_time_bins_per_trial': n_time_bins_per_trial,
                'time_bin_times': time_bin_times,
                'time_bins': time_bins,
                'time_bin_indexes': time_bin_indexes,
                'n_permutations': n_permutations,
                'n_kfold_splits': n_kfold_splits,
                'n_pca_components': n_pca_components,
                'deconf_methods': deconf_methods,
                'standardize_features': standardize_features,
                'standardize_confound': standardize_confound,

                'tune_predictability': tune_predictability,
                'inner_cv_hyperpar_tune_nsplits': inner_cv_hyperpar_tune_nsplits,
                'pr_fixed_pars': pr_fixed_pars,
                'hyperpars_grids' : hyperpars_grids,

                'mode' : mode,
                'calibrate': calibrate,
                'inner_cv_calibration_nsplits': inner_cv_calibration_nsplits,
                'calibrate_ensemble': calibrate_ensemble,
                'calibration_method' : calibration_method}

        out = {'pars' : pars,
               'df' : rs}

        output_folder = os.path.join(DATA_PATH, 'ephys_data_PR_results',
                                     'SIMULATION_{}'.format(settings_name))

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        output_full_path = os.path.join(output_folder, 'test_SIG_results_{}_{}.pkl'.format(settings_name, deconf_method))
        print('Saving simulation to {}'.format(output_full_path))
        pickle.dump(out, open(output_full_path, 'wb'))


