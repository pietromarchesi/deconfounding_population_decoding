import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotting_style import *
from constants import *
from SIMULATIONS.simulation_utils import make_simulated_data_highdim
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import r2_score

"""
CR on high-dimensional simulations where we iterate over the range of hyperparameter values
"""

settings_name_global = 'OCT14_R2_TEST'

modes = ['notune_cr_notune_dec']

plot_format = 'png'
dpi = 400
set_seed = True
shuffle_confounds = False
shuffle_confounds_individually = True

deconf_methods = ['nonlinear_confound_regression_svc',
                  'nonlinear_confound_regression_rf']

# DATA PARS
simulation_types = ['linear_highdim', 'nonlinear_highdim', 'nonlinear_highdim_noise']
n_repeats = 1
n_permutations = 0
parameter_to_vary = 'max_depth'
n_samples = 400
n_features = 30

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
                      'rf' : [3, 4, 5, 6, 7]}

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

plots_folder = os.path.join(DATA_PATH, 'plots', 'simulation_HIGHDIM_CR_R2',
                            'DECODE_SETTINGS_{}'.format(settings_name_global))

if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)

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
                               'deconf_method',
                               'parameter_to_vary',
                               'parameter_value',
                               'repeat',
                               'permuted',
                               'permutation',
                               'fold',
                               'method',
                               'feature',
                               'r2_score'])

    for simulation_type in simulation_types:

        data = {'data' : {n : {} for n in range(n_repeats)}}

        for nrep in range(n_repeats):

            for method_idx, deconf_method in enumerate(deconf_methods):

                if deconf_method == 'nonlinear_confound_regression_svc':
                    parameter_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
                    parameter_label = 'Regularization parameter'
                    hyperpar_labels = ['Regularization parameter', 'Maximum depth']
                elif deconf_method == 'nonlinear_confound_regression_rf':
                    parameter_values = [3, 4, 5, 6, 7, 8]
                    parameter_label = 'Maximum depth'
                    hyperpar_labels = ['Regularization parameter', 'Maximum depth']

                for k, parameter_value in enumerate(parameter_values):
                    if deconf_method == 'nonlinear_confound_regression_svc':
                        svr_c = parameter_value
                    elif deconf_method == 'nonlinear_confound_regression_rf':
                        max_depth = parameter_value

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

                    print('Deconf method: {}, simulation: {}, Parameter: {} = {}'.format(deconf_method, simulation_type, parameter_label, parameter_value))

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
                                    parameters_cr = {'C': svr_c}
                                    model = SVR(C=svr_c, kernel='rbf')
                                    print('SVC - C={}'.format(svr_c))

                                elif deconf_method == 'nonlinear_confound_regression_rf':
                                    parameters_cr = {'max_depth': cr_hyperpars_grids['rf']}
                                    model = RandomForestRegressor(n_estimators=cr_fixed_params['rf_n_estimators'],
                                                                  random_state=fold,
                                                                  max_depth=max_depth)
                                    print('RF - MD={}'.format(max_depth))


                                if tune_hyperparameters_cr:
                                    if deconf_method == 'linear_confound_regression':
                                        # no hyperparameters to tune
                                        pass
                                    else:
                                        #print('Tuning hyperparameters of CR: {}'.format(deconf_method))
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

                                ds.loc[ds.shape[0], :] = [simulation_type,
                                                          deconf_method,
                                                          parameter_to_vary,
                                                          parameter_value,
                                                          nrep,
                                                          permuted,
                                                          nperm,
                                                          fold,
                                                          deconf_method,
                                                          ix,
                                                          r2]


    ds['r2_score'] = pd.to_numeric(ds['r2_score'])
    excude_cols = ['mean_score', 'fold']
    ds = ds.groupby([c for c in ds.columns if ~np.isin(c, excude_cols)]).mean().reset_index()
    ds['dummy'] = 'dummy'

    for label, deconf_method in zip(hyperpar_labels, deconf_methods):

        dx = ds[ds['deconf_method'] == deconf_method]

        f, ax = plt.subplots(1, len(simulation_types), sharey=True, figsize=[7.5,3])
        if len(simulation_types) == 1:
            ax = np.array([ax])
        for i, simulation_type in enumerate(simulation_types):
            arts = sns.lineplot(data=dx[dx['simulation_type'] == simulation_type],
                                x='parameter_value',
                                y='r2_score',
                                hue='method',
                                palette=method_palette,
                                ax=ax[i],
                                style='method',
                                markers=cr_markers, markersize=8)
            ax[i].set_title(highdim_simulation_labels[simulation_type], fontsize=10)
            for coll in arts.collections:
                coll.set_linewidth(0)

            x = np.sort(dx['parameter_value'].unique())

            ax[i].set_xlabel(label)
            ax[i].legend().remove()
            if deconf_method == 'nonlinear_confound_regression_svc':
                ax[i].set_xscale('log')
            ax[i].axhline(0, ls=':', c='grey')
            ax[i].set_xticks(x)

        ax[0].set_ylim(-0.3, 0.7)
        ax[0].set_ylabel('R2 score')
        sns.despine()
        plt.tight_layout()
        plt.show()

        plot_name = 'R2_score_{}.{}'.format(deconf_method, plot_format)
        f.savefig(os.path.join(plots_folder, plot_name), dpi=dpi,
                  bbox_inches="tight")



