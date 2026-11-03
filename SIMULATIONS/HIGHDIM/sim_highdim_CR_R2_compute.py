import os
import numpy as np
import pandas as pd
from constants import *
import pickle
from SIMULATIONS.simulation_utils import make_simulated_data_highdim
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

"""
Compute regression scores (R2) for CR applied to high-dimensional simulations
"""

settings_name_global = 'OCT14_R2_FINAL'

modes = ['notune_cr_notune_dec',
         'notune_cr_tune_dec']

set_seed = True
shuffle_confounds = False
shuffle_confounds_individually = True

all_deconf_methods = ['raw',
                      'decode_from_confound',
                      'linear_confound_regression',
                      'linear_confound_regression_ridge',
                      'nonlinear_confound_regression_svc',
                      'nonlinear_confound_regression_svc_tuned',
                      'linear_confound_regression_glm',
                      'linear_confound_regression_glm_tuned',
                      'nonlinear_confound_regression_rf',
                      'nonlinear_confound_regression_rf_tuned']

deconf_methods = ['linear_confound_regression',
                  'nonlinear_confound_regression_svc',
                  'nonlinear_confound_regression_rf']

# DATA PARS
simulation_types = ['linear_highdim',
                    'nonlinear_highdim',
                    'nonlinear_highdim_noise']
n_repeats = 1
n_permutations = 0
parameter_to_vary = 'n_features'

if parameter_to_vary == 'n_features':
    n_samples = 400
    parameter_values = [5, 15, 25, 35, 45, 55]#[10, 15, 20, 25, 30, 35, 40, 45, 50, 55]#np.arange(5, 51, 5) #[5, 10, 20, 40, 80]

elif parameter_to_vary == 'n_samples':
    n_features = 30
    parameter_values = [50, 100, 150, 200, 300, 500]


# CR PARAMETERS
n_fold_cr_hyperpar_tuning = 3
cr_fixed_params = {'rf_max_depth'    : 4,
                   'rf_n_estimators' : 50,
                   'ridge_alpha'     : 1,
                   'svr_c'           : 1,
                   'poisson_alpha'   : 0.1}

cr_hyperpars_grids = {'ridge' : [0.001, 0.01, 0.1, 1, 10],
                      'glm' :   [0.001, 0.01, 0.1, 1, 10],
                      'svr' :   [0.001, 0.01, 0.1, 1, 10],
                      'rf_max_depth' : [4, 6, 8],
                      'rf_n_estimators' : [50, 100]}

# DECODER PARAMETERS
n_fold_decoder_tuning = 5
decoders_fixed_params = {'lr_C' : 1,
                         'svc_C' : 1,
                         'rf_max_depth' : 4,
                         'rf_n_estimators' : 50}

decoders_hyperpars_grids  = {'lr' : [0.001, 0.01, 0.1, 1, 10],
                             'svc' : [0.001, 0.01, 0.1, 1, 10],
                             'rf' : [3, 4, 5, 6, 7]}

n_kfold_splits = 5
standardize_features = True
standardize_confound = True
compute_r2 = False

output_folder = os.path.join(DATA_PATH, 'simulations_highdim_CR_R2_results',
                             'SIMULATION_{}'.format(settings_name_global))

plots_folder = os.path.join(DATA_PATH, 'plots', 'simulation_HIGHDIM_CR_R2_results',
                            'DECODE_SETTINGS_{}'.format(settings_name_global))

if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


for mode in modes:
    settings_name = '{}_{}'.format(settings_name_global, mode)

    if mode == 'notune_cr_notune_dec':
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = False

    elif mode == 'notune_cr_tune_dec':
        tune_hyperparameters_cr = False
        tune_hyperparameters_dec = True

    elif mode == 'tune_cr_tune_dec':
        tune_hyperparameters_cr = True
        tune_hyperparameters_dec = True

    elif mode == 'tune_cr_notune_dec':
        tune_hyperparameters_cr = True
        tune_hyperparameters_dec = False


    ds = pd.DataFrame(columns=['simulation_type',
                               'parameter_to_vary',
                               'parameter_value',
                               'repeat',
                               'permuted',
                               'permutation',
                               'fold',
                               'method',
                               'feature',
                               'R2',
                               'MSE',
                               'EV'])

    for simulation_type in simulation_types:

        df = pd.DataFrame(columns=['simulation_type',
                                   'parameter_to_vary',
                                   'parameter_value',
                                   'repeat',
                                   'permuted',
                                   'permutation',
                                   'fold',
                                   'method',
                                   'decoder',
                                   'mean_score'])

        data = {'data' : {n : {} for n in range(n_repeats)}}

        for nrep in range(n_repeats):

            for k, parameter_value in enumerate(parameter_values):

                if parameter_to_vary == 'n_features':
                    n_features = parameter_value
                else:
                    raise NotImplementedError

                if set_seed:
                    seed = nrep + k

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
                # permute C and y in the same way (although C permutations not necessary for CR)
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

                    print('Deconf method: {}, simulation: {}, n_features: {}'.format(deconf_method, simulation_type, n_features))

                    for nperm in range(n_permutations + 1):
                        if nperm == 0:
                            permuted = False
                        elif nperm > 0:
                            permuted = True
                        if nperm % 10 == 0:
                            print('--- n_perms = {}'.format(nperm))

                        y_dec = y_permuted_list[nperm]
                        C_dec = np.copy(C)

                        kfold_accuracy_full, kfold_accuracy_naive = [], []

                        kfold = StratifiedKFold(n_splits=n_kfold_splits, shuffle=True,
                                                random_state=92)
                        iterable = kfold.split(X, y)

                        kfold_accuracy_full = []
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

                            # Apply confound regression
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
                                    model = PoissonRegressor(max_iter=3000, alpha=cr_fixed_params['poisson_alpha'])

                                elif deconf_method == 'nonlinear_confound_regression_svc':
                                    parameters_cr = {'C': cr_hyperpars_grids['svr']}
                                    model = SVR(C=cr_fixed_params['svr_c'], kernel='rbf')

                                elif deconf_method == 'nonlinear_confound_regression_rf':
                                    parameters_cr = {'max_depth': cr_hyperpars_grids['rf_max_depth'],
                                                     'n_estimators' : cr_hyperpars_grids['rf_n_estimators']}
                                    model = RandomForestRegressor(n_estimators=cr_fixed_params['rf_n_estimators'],
                                                                  random_state=fold,
                                                                  max_depth=cr_fixed_params['rf_max_depth'])

                                if tune_hyperparameters_cr:
                                    if deconf_method == 'linear_confound_regression':
                                        # no hyperparameters to tune
                                        pass
                                    else:
                                        inner_cv_1 = KFold(n_splits=n_fold_cr_hyperpar_tuning, shuffle=True,
                                                                     random_state=fold)
                                        model = GridSearchCV(model, parameters_cr, refit=True, scoring='r2',
                                                             cv=inner_cv_1)

                                model = model.fit(X=C_train, y=X_train[:, ix])
                                new_X_train.append(model.predict(C_train).reshape(-1, 1))
                                new_X_test.append(model.predict(C_test).reshape(-1, 1))

                                # EVALUATE R2
                                feat_pred = model.predict(C_test)
                                feat_true = X_test[:, ix]
                                r2 = r2_score(y_pred=feat_pred, y_true=feat_true)
                                mse = mean_squared_error(y_pred=feat_pred, y_true=feat_true)
                                ev = explained_variance_score(y_pred=feat_pred, y_true=feat_true)

                                ds.loc[ds.shape[0], :] = [simulation_type,
                                                          parameter_to_vary,
                                                          parameter_value,
                                                          nrep,
                                                          permuted,
                                                          nperm,
                                                          fold,
                                                          deconf_method,
                                                          ix,
                                                          r2,
                                                          mse,
                                                          ev]


    pars = {'settings_name': settings_name,
            'n_permutations': n_permutations,
            'n_repeats': n_repeats,
            'n_samples': n_samples,
            'set_seed': set_seed,
            'simulation_types': simulation_types,
            'deconf_methods': deconf_methods,
            'n_kfold_splits': n_kfold_splits,
            'parameter_to_vary': parameter_to_vary,
            'parameter_values': parameter_values,
            'standardize_features': standardize_features,
            'standardize_confound': standardize_confound,
            'compute_r2': compute_r2,

            'n_fold_cr_hyperpar_tuning': n_fold_cr_hyperpar_tuning,
            'cr_fixed_params': cr_fixed_params,
            'cr_hyperpars_grids': cr_hyperpars_grids,

            'shuffle_confounds' : shuffle_confounds,
            'shuffle_confounds_individually' : shuffle_confounds_individually,

            'n_fold_decoder_tuning': n_fold_decoder_tuning,
            'decoders_fixed_params': decoders_fixed_params,
            'decoders_hyperpars_grids': decoders_hyperpars_grids,

            'tune_hyperparameters_cr': tune_hyperparameters_cr,
            'tune_hyperparameters_dec': tune_hyperparameters_dec}

    out = {'pars' : pars,
           'ds' : ds,
           'data' : data}

    output_full_path = os.path.join(output_folder, 'results_r2_{}.pkl'.format(settings_name))
    print('Saving simulation to {}'.format(output_full_path))
    pickle.dump(out, open(output_full_path, 'wb'))



