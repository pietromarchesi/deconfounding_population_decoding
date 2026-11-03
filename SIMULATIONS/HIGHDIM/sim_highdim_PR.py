import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotting_style import *
from constants import *
from scipy.stats import mannwhitneyu, bootstrap
import pickle
from SIMULATIONS.simulation_utils import make_simulated_data, make_simulated_data_highdim
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from decoding_utils import log2_likelihood
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, make_scorer, accuracy_score
from venn_abers import VennAbersCalibrator
from sklearn.metrics import balanced_accuracy_score


"""
PR estimation on high-dimensional simulations
"""

settings_name_global = 'OCT7_TEST_LIN_SHUFFLETOGETHER'

modes = ['nocalib_notune',
         'calib_sigmoid',
         'calib_isotonic',
         'calib_vennabers',
         'nocalib_tune']

set_seed = True

deconf_methods = ['predictability_lin',
                  'predictability_nonlin_svc',
                  'predictability_nonlin_rf']

# DATA PARS
simulation_types = ['linear_highdim', 'nonlinear_highdim', 'nonlinear_highdim_noise']
n_repeats = 1
n_permutations = 100
shuffle_confounds = True
shuffle_confounds_individually = False

parameter_to_vary = 'n_features'
# parameter_to_vary = 'n_samples'

if parameter_to_vary == 'n_features':
    n_samples = 200
    parameter_values = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

elif parameter_to_vary == 'n_samples':
    n_features = 30
    parameter_values = [50, 100, 150, 200, 300, 500]

# PREDICTABILITY PARS
lr_penalty = 'l2'
lr_solver = 'saga'
lr_max_iter = 6000
lr_C_notune = 1
svc_kernel = 'rbf'
svc_C_notune = 0.1
rf_n_estimators = 50
rf_max_depth = 4

hyperpars_grids = {'predictability_lin': [0.001, 0.01, 0.1, 1, 10],
                   'predictability_nonlin_svc': [0.001, 0.01, 0.1, 1, 10],
                   'predictability_nonlin_rf': [3, 4, 5, 6, 7],
                   'predictability_nonlin_gb': [3, 4, 5, 6, 7]}

n_kfold_splits = 5
inner_cv_calibration_nsplits = 5
inner_cv_hyperpar_tune_nsplits = 5
standardize_features = True
standardize_confound = True


output_folder = os.path.join(DATA_PATH, 'simulations_highdim_PR_results',
                             'SIMULATION_{}'.format(settings_name_global))

plots_folder = os.path.join(DATA_PATH, 'plots', 'simulations',
                            'DECODE_SETTINGS_{}'.format(settings_name_global))

if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)

    if mode == 'nocalib_notune':
        tune_hyperparameters = False
        calibrate = False
        calibration_ensemble = True
        calibration_method = None

    if mode == 'nocalib_tune':
        tune_hyperparameters = True
        calibrate = False
        calibration_ensemble = True
        calibration_method = None

    if mode == 'calib_sigmoid':
        tune_hyperparameters = False
        calibrate = True
        calibration_ensemble = True
        calibration_method = 'sigmoid'

    elif mode == 'calib_isotonic':
        tune_hyperparameters = False
        calibrate = True
        calibration_ensemble = True
        calibration_method = 'isotonic'

    elif mode == 'calib_vennabers':
        tune_hyperparameters = False
        calibrate = True
        calibration_ensemble = True
        calibration_method = 'venn-abers'


    for simulation_type in simulation_types:

        df = pd.DataFrame(columns=['simulation_type',
                                   'parameter_to_vary',
                                   'parameter_value',
                                   'repeat',
                                   'permuted',
                                   'permutation',
                                   'calibrated',
                                   'method',
                                   'decoder',
                                   'mean_score',
                                   'test_acc_full',
                                   'test_acc_naive'])

        data = {'data' : {n : {} for n in range(n_repeats)}}

        for nrep in range(n_repeats):

            for k, parameter_value in enumerate(parameter_values):

                if parameter_to_vary == 'n_features':
                    n_features = parameter_value
                elif parameter_to_vary == 'n_samples':
                    n_samples = parameter_value
                else:
                    raise NotImplementedError

                # seed setting ensures that datasets for different parameter values are
                # different, and that datasets for every repeat are different,
                # but that for every mode we get the same datasets for proper comparisons
                if set_seed:
                    seed = nrep+k

                noise_levels_linear_features = np.random.default_rng(seed=seed).uniform(low=1, high=4, size=n_features)
                noise_levels_confounds = np.random.default_rng(seed=seed).uniform(low=1, high=4, size=n_features)
                noise_levels_nonlinear_features = np.random.default_rng(seed=seed).uniform(low=0.1, high=2, size=n_features)
                noise_levels_nonlinear_noise_features = np.random.default_rng(seed=seed).uniform(low=0.6, high=2, size=n_features)

                if simulation_type == 'linear_highdim':
                    X_all = []
                    C_all = []
                    for ifeat, nfeat in enumerate(range(n_features)):
                        X, y, C = make_simulated_data_highdim(simulation_type='linear',
                                                      n_samples=n_samples,
                                                      seed=seed,
                                                      n_feature=ifeat,
                                                      noise_features=noise_levels_linear_features[ifeat],
                                                      noise_confound=noise_levels_confounds[ifeat])
                        X_all.append(X)
                        C_all.append(C)
                    X = np.hstack(X_all)
                    C = np.hstack(C_all)

                elif simulation_type == 'nonlinear_highdim':
                    X_all = []
                    C_all = []
                    for ifeat, nfeat in enumerate(range(n_features)):
                        X, y, C = make_simulated_data_highdim(simulation_type='nonlinear',
                                                      n_samples=n_samples,
                                                      seed=seed,
                                                      n_feature=ifeat,
                                                      noise_features=noise_levels_nonlinear_features[ifeat],
                                                      noise_confound=None)
                        X_all.append(X)
                        C_all.append(C)
                    X = np.hstack(X_all)
                    C = np.hstack(C_all)

                elif simulation_type == 'nonlinear_highdim_noise':
                    X_all = []
                    C_all = []
                    for ifeat, nfeat in enumerate(range(n_features)):
                        X, y, C = make_simulated_data_highdim(simulation_type='nonlinear_noise',
                                                      n_samples=n_samples,
                                                      seed=seed,
                                                      n_feature=ifeat,
                                                      noise_features=noise_levels_nonlinear_noise_features[ifeat],
                                                      noise_confound=None)
                        X_all.append(X)
                        C_all.append(C)
                    X = np.hstack(X_all)
                    C = np.hstack(C_all)

                if shuffle_confounds:
                    if shuffle_confounds_individually:
                        for col in range(C.shape[1]):
                            C[:, col] = np.random.default_rng(seed=col).permutation(C[:, col])
                    else:
                        indx = np.random.default_rng(seed=seed).permutation(C.shape[0])
                        C = C[indx, :]


                # generate permutations
                # permute C and y in the same way
                C_permuted_list, y_permuted_list = [], []
                for pix in range(n_permutations + 1):
                    # print(s)
                    if pix == 0:
                        C_permuted = np.copy(C)
                        y_permuted = np.copy(y)
                    else:
                        random_index = np.random.permutation(np.arange(X.shape[0]))
                        C_permuted = C[random_index, :]
                        y_permuted = y[random_index]
                    C_permuted_list.append(C_permuted)
                    y_permuted_list.append(y_permuted)

                data_key = 'data'
                data[data_key][nrep]['X'] = X
                data[data_key][nrep]['y'] = y
                data[data_key][nrep]['C'] = C

                for method_idx, deconf_method in enumerate(deconf_methods):

                    print('\nMode: {}\nDeconf method: {}, simulation: {}, n_features: {}'.format(mode, deconf_method, simulation_type, n_features))

                    for nperm in range(n_permutations + 1):
                        if nperm == 0:
                            permuted = False
                        elif nperm > 0:
                            permuted = True
                        if nperm % 10 == 0:
                            print('--- n_perms = {}'.format(nperm))

                        y_dec = y_permuted_list[nperm]
                        C_dec = C_permuted_list[nperm]

                        # print('Decoding with {}, deconf method: {}'.format(decoder, deconf_method))
                        # print('--- repeat {:02d} of {:02d} --- permutation {:02d} of {:02d}'.format(nrep, n_repeats, nperm, n_permutations))
                        kfold_accuracy_full, kfold_accuracy_naive = [], []
                        llratios = []

                        kfold = StratifiedKFold(n_splits=n_kfold_splits, shuffle=True,
                                                random_state=92)
                        iterable = kfold.split(X, y)

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

                            log_loss_scorer = make_scorer(log_loss, response_method='predict_proba', greater_is_better=False)

                            if tune_hyperparameters:
                                if calibrate:
                                    raise ValueError('Cannot tune hyperparameters AND calibrate')

                                if deconf_method == 'predictability_lin':
                                    parameters = {'C': hyperpars_grids['predictability_lin']}
                                    full_model = LogisticRegression(penalty=lr_penalty, solver=lr_solver, random_state=fold, max_iter=lr_max_iter)
                                    naive_model = LogisticRegression(penalty=lr_penalty, solver=lr_solver, random_state=fold, max_iter=lr_max_iter)

                                elif deconf_method == 'predictability_nonlin_svc':
                                    parameters = {'C': hyperpars_grids['predictability_nonlin_svc']}
                                    full_model = SVC(probability=True, kernel=svc_kernel, random_state=fold)
                                    naive_model = SVC(probability=True, kernel=svc_kernel, random_state=fold)

                                elif deconf_method == 'predictability_nonlin_rf':
                                    parameters = {'max_depth': hyperpars_grids['predictability_nonlin_rf']}
                                    full_model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=fold)
                                    naive_model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=fold)

                                inner_cv = StratifiedKFold(n_splits=inner_cv_hyperpar_tune_nsplits, shuffle=True, random_state=fold)
                                full_model = GridSearchCV(full_model, param_grid=parameters, refit=True, scoring=log_loss_scorer, cv=inner_cv)

                                inner_cv = StratifiedKFold(n_splits=inner_cv_hyperpar_tune_nsplits, shuffle=True, random_state=fold)
                                naive_model = GridSearchCV(naive_model, param_grid=parameters, refit=True, scoring=log_loss_scorer, cv=inner_cv)

                            else:
                                if deconf_method == 'predictability_lin':
                                    full_model = LogisticRegression(penalty=lr_penalty, C=lr_C_notune, solver=lr_solver, random_state=fold, max_iter=lr_max_iter)
                                    naive_model = LogisticRegression(penalty=lr_penalty, C=lr_C_notune, solver=lr_solver, random_state=fold, max_iter=lr_max_iter)

                                elif deconf_method == 'predictability_nonlin_svc':
                                    full_model = SVC(probability=True, C=svc_C_notune, kernel=svc_kernel, random_state=fold)
                                    naive_model = SVC(probability=True, C=svc_C_notune, kernel=svc_kernel, random_state=fold)

                                elif deconf_method == 'predictability_nonlin_rf':
                                    full_model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=fold, max_depth=rf_max_depth)
                                    naive_model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=fold, max_depth=rf_max_depth)

                            if calibrate:
                                if calibration_method == 'venn-abers':
                                    full_model = VennAbersCalibrator(estimator=full_model, inductive=False,
                                                                     n_splits=inner_cv_calibration_nsplits, random_state=fold)
                                    naive_model = VennAbersCalibrator(estimator=naive_model, inductive=False,
                                                                      n_splits=inner_cv_calibration_nsplits,
                                                                      random_state=fold)

                                else:
                                    calib_inner_cv = StratifiedKFold(n_splits=inner_cv_calibration_nsplits, shuffle=True,
                                                                     random_state=fold)
                                    full_model = CalibratedClassifierCV(full_model, cv=calib_inner_cv,
                                                                        ensemble=calibration_ensemble,
                                                                        method=calibration_method)

                                    naive_model = CalibratedClassifierCV(naive_model, cv=calib_inner_cv,
                                                                         ensemble=calibration_ensemble,
                                                                         method=calibration_method)

                            full_model.fit(features_full_train, y_train)
                            naive_model.fit(features_naive_train, y_train)

                            y_pred_proba_full = full_model.predict_proba(features_full_test)
                            y_pred_proba_naive = naive_model.predict_proba(features_naive_test)

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

                        df.loc[df.shape[0], :] = [simulation_type,
                                                  parameter_to_vary,
                                                  parameter_value,
                                                  nrep,
                                                  permuted,
                                                  nperm,
                                                  calibrate,
                                                  deconf_method,
                                                  'none',
                                                  predictability,
                                                  accuracy_full,
                                                  accuracy_naive]

                df['mean_score'] = pd.to_numeric(df['mean_score'])

        predictability_pars = {'tune_hyperpars_nfolds': inner_cv_calibration_nsplits,
                               'lr_penalty': lr_penalty,
                               'lr_solver': lr_solver,
                               'lr_max_iter': lr_max_iter,
                               'lr_C_notune': lr_C_notune,
                               'svc_kernel': svc_kernel,
                               'svc_C_notune': svc_C_notune,
                               'rf_n_estimators': rf_n_estimators,
                               'rf_max_depth': rf_max_depth,
                               'hyperpars_grids': hyperpars_grids}

        pars = {'settings_name' : settings_name,
                'n_permutations' : n_permutations,
                'n_repeats' : n_repeats,
                'n_samples' : n_samples,
                'set_seed' : set_seed,
                'simulation_types' : simulation_types,
                'simulation_type' : simulation_type,
                'deconf_methods' : deconf_methods,
                'calibrate' : calibrate,
                'n_kfold_splits' : n_kfold_splits,
                'parameter_to_vary' : parameter_to_vary,
                'parameter_values' : parameter_values,
                'standardize_features' : standardize_features,
                'standardize_confound' : standardize_confound,
                'calibration_ensemble' : calibration_ensemble,
                'calibration_method' : calibration_method,
                'inner_cv_hyperpar_tune_nsplits' : inner_cv_hyperpar_tune_nsplits,
                'inner_cv_calibration_nsplits' : inner_cv_calibration_nsplits,
                'predictability_pars' : predictability_pars,
                'tune_predictability' : tune_hyperparameters,
                'shuffle_confounds' : shuffle_confounds,
                'shuffle_confounds_individually' : shuffle_confounds_individually}

        out = {'pars' : pars,
               'df' : df,
               'data' : data}

        output_full_path = os.path.join(output_folder, 'results_{}_{}.pkl'.format(settings_name, simulation_type))
        print('Saving simulation to {}'.format(output_full_path))
        pickle.dump(out, open(output_full_path, 'wb'))



