import os
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, PoissonRegressor
from constants import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import balanced_accuracy_score, make_scorer, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import knncmi as knn
import sklearn as skl


class NotEnoughTrialsError(ValueError): pass



def load_decoding_data(settings_name, experiment_name,
                       area_spikes='V1', dataset='modid'):

    data_path = os.path.join(DATA_PATH, 'ephys_data', dataset)
    file_name = 'decoding_data_{}_{}_{}.pkl'.format(
        settings_name,
        experiment_name,
        area_spikes)

    data = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
    return data


def log2_likelihood(true, predictions) :
    y_true = true
    y_pred = predictions[:, 1]
    eps = np.finfo(y_pred.dtype).eps
    y_pred = np.clip(y_pred, eps, 1 - eps)
    terms = y_true * np.log2(y_pred) + (1 - y_true) * np.log2(1 - y_pred)
    ll = np.sum(terms) / len(y_true)
    return ll


def poisson_pseudoR2(y_true, y_pred, y_train):
    # This is our scoring function. Implements pseudo-R2
    # y_null is mean of y on training set
    ynull = np.mean(y_train)
    y_pred = y_pred.reshape(y_true.shape)
    eps = np.spacing(1)
    L1 = np.sum(y_true * np.log(eps + y_pred) - y_pred)
    L1_v = y_true * np.log(eps + y_pred) - y_pred
    L0 = np.sum(y_true * np.log(eps + ynull) - ynull)
    LS = np.sum(y_true * np.log(eps + y_true) - y_true)
    R2 = 1 - (LS - L1) / (LS - L0)
    return R2


def decode(X, C, y,
           deconf_method,
           decoder,
           n_kfold_splits=3,
           standardize_features=True,
           standardize_confound=True,
           linear_regression_variation='ridge',
           tune_hyperpars_decoder=False,
           LR_predictability_penalty='l1',
           LR_predictability_solver='saga',
           LR_predictability_C=0.2,
           tune_predictability=False,
           LR_predictability_C_list=None, SVC_predictability_C_list=None,
           LR_decode_C_list=None, SVC_decode_C_list=None,
           LR_predictability_max_iter=7000,
           tune_predictability_nfolds=5,
           LR_decode_C=None, SVC_decode_C=None,
           regularize_naive_model=False,
           SVC_predictability_C=None):

    methods = ['raw',
               'decode_from_confound',
               'predictability_lin',
               'predictability_nonlin',
               'predictability_lin_nodeconf',
               'predictability_nonlin_nodeconf',
               'distribution_match',
               'linear_confound_regression',
               'nonlinear_confound_regression',
               'cmi']

    assert np.isin(deconf_method, methods)

    if not regularize_naive_model:
        raise ValueError('Always regularize both models!')

    decoders = ['logistic_regression', 'random_forest',
                'linear_discriminant_analysis', 'quadratic_discriminant_analysis',
                'ridge_regression', 'SVC', 'none']
    assert np.isin(decoder, decoders)


    total_n_trials = X.shape[0]
    if total_n_trials < 20 :
        raise NotEnoughTrialsError('Not enough trials to decode')


    total_n_trials = X.shape[0]

    kfold = StratifiedKFold(n_splits=n_kfold_splits, shuffle=True,
                            random_state=92)
    iterable = kfold.split(X, y)


    if total_n_trials < 20:
        raise NotEnoughTrialsError('Not enough trials to decode')

    if np.isin(deconf_method, ['raw', 'distribution_match',
                               'linear_confound_regression',
                               'nonlinear_confound_regression',
                               'decode_from_confound']):
        # run the decoding
        kfold_scores = []
        y_test_all, y_pred_all, c_all = [], [], []

        for fold, (training_ind, testing_ind) in enumerate(iterable):

            X_train = X[training_ind, :]
            X_test = X[testing_ind, :]
            y_train = y[training_ind]
            y_test = y[testing_ind]
            C_train = C[training_ind, :]
            C_test = C[testing_ind, :]

            if standardize_features:
                ss = StandardScaler()
                X_train = ss.fit_transform(X_train)
                X_test = ss.transform(X_test)

            if standardize_confound:
                ss = StandardScaler()
                C_train = ss.fit_transform(C_train)
                C_test = ss.transform(C_test)

            # Apply confound regression (linear regression)
            if deconf_method == 'linear_confound_regression':
                if linear_regression_variation == 'ridge':
                    reg = RidgeCV(alphas=[0.001, 0.1, 1, 10], cv=None, scoring=None, store_cv_results=False)
                    reg.fit(C_train, X_train)
                else:
                    reg = LinearRegression().fit(C_train, X_train)
                X_train = X_train  - reg.predict(C_train)
                X_test = X_test - reg.predict(C_test)

            # Apply confound regression (nonlinear variation)
            elif deconf_method == 'nonlinear_confound_regression' :
                reg = MultiOutputRegressor(SVR(C=1, kernel='rbf')).fit(C_train, X_train)
                # reg_nonlin = RandomForestRegressor(n_estimators=50).fit(C_train, X_train)
                if X.shape[1] == 1:
                    X_train = X_train - reg.predict(C_train).reshape(-1, 1)
                    X_test = X_test - reg.predict(C_test).reshape(-1, 1)
                elif X.shape[1] > 1:
                    X_train = X_train - reg.predict(C_train)
                    X_test = X_test - reg.predict(C_test)

            # DECODE
            if decoder == 'random_forest':
                if tune_hyperpars_decoder:
                    raise NotImplementedError
                model = RandomForestClassifier(n_estimators=200, random_state=92)

            elif decoder == 'logistic_regression':
                if tune_hyperpars_decoder:
                    parameters = {'C': LR_decode_C_list}
                    lr = LogisticRegression(penalty='l2')
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
                    model = GridSearchCV(lr, parameters, refit=True, scoring=make_scorer(balanced_accuracy_score),cv=inner_cv)
                else:
                    model = LogisticRegression(penalty='l2', C=LR_decode_C, random_state=1)

            elif decoder == 'linear_discriminant_analysis':
                if tune_hyperpars_decoder:
                    raise NotImplementedError
                model = LinearDiscriminantAnalysis()
            elif decoder == 'quadratic_discriminant_analysis':
                if tune_hyperpars_decoder:
                    raise NotImplementedError
                model = QuadraticDiscriminantAnalysis()
            elif decoder == 'ridge_regression':
                if tune_hyperpars_decoder:
                    raise NotImplementedError
                model = RidgeClassifier()
            elif decoder == 'SVC':
                if tune_hyperpars_decoder:
                    parameters = {'C': SVC_decode_C_list}
                    svc = SVC(probability=True, kernel='rbf', random_state=1)
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
                    model = GridSearchCV(svc, parameters, refit=True,
                                         scoring=make_scorer(balanced_accuracy_score), cv=inner_cv)
                else:
                    model = SVC(C=SVC_decode_C, kernel='rbf', random_state=1)
            else:
                raise ValueError

            if deconf_method == 'decode_from_confound':
                model.fit(C_train, y_train)
                y_pred = model.predict(C_test)

            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            if tune_hyperpars_decoder:
                try:
                    best_C = model.best_params_['C']
                    #print('BEST C: {}'.format(best_C))
                except KeyError:
                    pass
            score = balanced_accuracy_score(y_test, y_pred)
            kfold_scores.append(score)
            y_test_all.append(y_test)
            y_pred_all.append(y_pred)
            c_all.append(C_test[:, 0])

        target = np.hstack(y_test_all)
        predictions = np.hstack(y_pred_all)
        confound = np.hstack(c_all)
        mean_score = np.mean(kfold_scores)

    elif deconf_method == 'cmi':

        cmidata = np.hstack([X, y.reshape(-1, 1), C])
        if X.shape[1] == 1:
            labels = ['X', 'y', 'C']
            xlabel = ['X']
        elif X.shape[1] == 2:
            labels = ['X1', 'X2', 'y', 'C']
            xlabel = ['X1', 'X2']
        else:
            raise ValueError

        if C.shape[1] != 1:
            raise ValueError

        cmidf = pd.DataFrame(cmidata, columns=labels)
        cmi = knn.cmi(xlabel, ['y'], ['C'], k=7, data=cmidf, minzero=0, discrete_dist=1)
        mean_score = cmi
        kfold_scores = []


    elif np.isin(deconf_method, ['predictability_lin',
                                 'predictability_nonlin']):


        kfold_accuracy_full, kfold_accuracy_naive = [], []
        llratios = []

        for fold, (training_ind, testing_ind) in enumerate(iterable):

            X_train = X[training_ind, :]
            X_test = X[testing_ind, :]
            y_train = y[training_ind]
            y_test = y[testing_ind]
            C_train = C[training_ind, :]
            C_test = C[testing_ind, :]

            if standardize_features:
                ss = StandardScaler()
                X_train = ss.fit_transform(X_train)
                X_test = ss.transform(X_test)

            if standardize_confound:
                ss = StandardScaler()
                C_train = ss.fit_transform(C_train)
                C_test = ss.transform(C_test)


            if np.isin(deconf_method, ['predictability_lin',
                                       'predictability_nonlin']):
                features_full_train = np.hstack([X_train, C_train])
                features_full_test = np.hstack([X_test, C_test])
                features_naive_train = C_train
                features_naive_test = C_test

            if np.isin(deconf_method, ['predictability_lin']):
                if tune_predictability:
                    parameters = {'C': LR_predictability_C_list}
                    full_model = LogisticRegression(penalty=LR_predictability_penalty, solver=LR_predictability_solver,
                                                    random_state=1, max_iter=LR_predictability_max_iter)
                    naive_model = LogisticRegression(penalty=LR_predictability_penalty, solver=LR_predictability_solver,
                                                     random_state=1, max_iter=LR_predictability_max_iter)

                else:
                    full_model = LogisticRegression(penalty=LR_predictability_penalty, solver=LR_predictability_solver,
                                                    C=LR_predictability_C, random_state=1,
                                                    max_iter=LR_predictability_max_iter)
                    naive_model = LogisticRegression(penalty=LR_predictability_penalty, solver=LR_predictability_solver,
                                                     C=LR_predictability_C, random_state=1,
                                                     max_iter=LR_predictability_max_iter)

            elif np.isin(deconf_method, ['predictability_nonlin']):
                if tune_predictability:
                    parameters = {'C': SVC_predictability_C_list}
                    full_model = SVC(probability=True, kernel='rbf', gamma='scale', random_state=1)
                    naive_model = SVC(probability=True, kernel='rbf', gamma='scale', random_state=1)
                else:
                    full_model = SVC(probability=True, kernel='rbf', C=SVC_predictability_C, random_state=1)
                    naive_model = SVC(probability=True, kernel='rbf', C=SVC_predictability_C, random_state=1)


            if tune_predictability:
                if skl.__version__ == '1.7.0':
                    log_loss_scorer = make_scorer(log_loss, response_method='predict_proba', greater_is_better=False)
                else:
                    log_loss_scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)

                inner_cv = StratifiedKFold(n_splits=tune_predictability_nfolds, shuffle=True, random_state=1)
                full_model = GridSearchCV(full_model, parameters, refit=True, scoring=log_loss_scorer, cv=inner_cv)
                inner_cv = StratifiedKFold(n_splits=tune_predictability_nfolds, shuffle=True, random_state=1)
                naive_model = GridSearchCV(naive_model, parameters, refit=True, scoring=log_loss_scorer, cv=inner_cv)

            full_model.fit(features_full_train, y_train)
            naive_model.fit(features_naive_train, y_train)

            y_pred_proba_full = full_model.predict_proba(features_full_test)
            y_pred_proba_naive = naive_model.predict_proba(features_naive_test)

            fold_accuracy_full = balanced_accuracy_score(y_test, full_model.predict(features_full_test))
            fold_accuracy_naive = balanced_accuracy_score(y_test, naive_model.predict(features_naive_test))

            kfold_accuracy_full.append(fold_accuracy_full)
            kfold_accuracy_naive.append(fold_accuracy_naive)

            log_likelihood_full = log2_likelihood(y_test, y_pred_proba_full)
            log_likelihood_naive = log2_likelihood(y_test, y_pred_proba_naive)
            # print(fold_accuracy_full)
            # print(log_likelihood_full)
            # print(log_likelihood_naive)

            llratio = log_likelihood_full - log_likelihood_naive
            llratios.append(llratio)

        accuracy_full = np.mean(kfold_accuracy_full)
        accuracy_naive = np.mean(kfold_accuracy_naive)
        predictability = np.mean(llratios)
        # print(accuracy_full)
        # print(accuracy_naive)
        # print(predictability)

        mean_score = predictability
        kfold_scores = llratios

    output = {'mean_score' : mean_score,
              'kfold_scores' : kfold_scores,
              'method' : deconf_method}

    return output



