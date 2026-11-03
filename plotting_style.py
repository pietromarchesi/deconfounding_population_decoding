import seaborn as sns

predictability_methods = ['predictability_lin',
                          'predictability_nonlin',
                          'predictability_nonlin_svc',
                          'predictability_nonlin_rf',
                          'predictability_nonlin_gb',
                          'predictability_nonlin_ensemble',
                          'predictability_lin_nodeconf',
                          'predictability_nonlin_nodeconf',
                          'predictability_nonlin_ensemble_nodeconf',
                          'cmi']

cr_methods =     ['raw',
                  'decode_from_confound',
                  'linear_confound_regression',
                  'linear_confound_regression_ridge',
                  'linear_confound_regression_glm',
                  'linear_confound_regression_glm_tuned',
                  'nonlinear_confound_regression_svc',
                  'nonlinear_confound_regression_svc_tuned',
                  'nonlinear_confound_regression_rf',
                  'nonlinear_confound_regression_rf_tuned']

method_labels = {'raw' : 'Raw',
                 'decode_from_confound' : 'From\nconfound',
                 'distribution_match' : 'Post-hoc\nCB',
                 'confound_isolating_cv' : 'Conf. isolating\nCV',
                 'linear_confound_regression' : 'Linear CR',
                 'linear_confound_regression_reg': 'Linear CR',
                 'nonlinear_confound_regression' : 'Nonlin. CR',
                 'nonlinear_confound_regression_rf': 'Nonlin. CR',
                 'predictability_lin' : 'Linear PR',
                 'predictability_nonlin' : 'Nonlin. PR',
                 'predictability_nonlin_rf': 'Nonlin. PR',
                 'predictability_nonlin_svc' : 'Nonlin. PR',
                 'predictability_nonlin_gb': 'Nonlin. PR',
                 'predictability_nonlin_ensemble' : 'Nonlin. PR',
                 'predictability_lin_nodeconf' : 'Bo',
                 'predictability_nonlin_nodeconf' : 'Bo',
                 'cmi' : 'cMI'}

# --- CR ---

cr_labels_detailed = {'raw' : 'Raw',
             'decode_from_confound' : 'Decode from C',
             'linear_confound_regression' : 'Linear CR',
             'linear_confound_regression_ridge' : 'Linear CR (Ridge)',
             'linear_confound_regression_glm' : 'Linear CR (GLM)',
             'linear_confound_regression_glm_tuned' : 'Linear CR (GLM*)',
             'nonlinear_confound_regression_svc': 'Nonlin. CR (SVM)',
             'nonlinear_confound_regression_svc_tuned': 'Nonlin. CR (SVM*)',
             'nonlinear_confound_regression_rf' : 'Nonlin. CR (RF)',
            'nonlinear_confound_regression_rf_tuned' : 'Nonlin. CR (RF*)'}

cr_labels_detailed_nextline = {'raw' : 'Raw',
             'decode_from_confound' : 'Decode from C',
             'linear_confound_regression' : 'Linear CR',
             'linear_confound_regression_ridge' : 'Linear CR\n(Ridge)',
             'linear_confound_regression_glm' : 'Linear CR\n(GLM)',
             'linear_confound_regression_glm_tuned' : 'Linear CR\n(GLM*)',
             'nonlinear_confound_regression_svc': 'Nonlin. CR\n(SVM)',
             'nonlinear_confound_regression_svc_tuned': 'Nonlin. CR\n(SVM*)',
             'nonlinear_confound_regression_rf' : 'Nonlin. CR\n(RF)',
            'nonlinear_confound_regression_rf_tuned' : 'Nonlin. CR\n(RF*)'}

cr_labels_main = {'raw' : 'Raw',
             'decode_from_confound' : 'From $C$',
             'linear_confound_regression' : 'Linear CR',
             'linear_confound_regression_ridge' : 'Linear CR',
             'linear_confound_regression_glm' : 'Linear CR',
             'linear_confound_regression_glm_tuned' : 'Linear CR',
             'nonlinear_confound_regression_svc': 'Nonlin. CR',
             'nonlinear_confound_regression_svc_tuned': 'Nonlin. CR',
             'nonlinear_confound_regression_rf' : 'Nonlin. CR'}

decoder_labels = {'logistic_regression' : 'LR dec.',
                  'logistic_regression_tuned' : 'LR dec.\n(tuned HP)',
                  'SVC' : 'SVM dec.',
                   'SVC_tuned' : 'SVM dec.\n(tuned HP)',
                  'random_forest' : 'RF dec.',
                  'random_forest_tuned' : 'RF dec.\n(tuned HP)'}

decoder_markers = {'logistic_regression' : '^',
                  'logistic_regression_tuned' : '^',
                   'SVC' : 'd',
                   'SVC_tuned' : 'd',
                  'random_forest' : 'X',
                  'random_forest_tuned' : 'X'}

cr_markers = {'linear_confound_regression' : 'o',
              'nonlinear_confound_regression_rf' : 'D',
              'nonlinear_confound_regression_svc' : 'X'}

# --- PR ---

pr_labels_detailed = {'raw' : 'Raw',
             'predictability_lin' : 'Linear PR',
             'predictability_lin_tuned' : 'Linear PR*',
             'predictability_nonlin_svc': 'Nonlin. PR (SVM)',
            'predictability_nonlin_svc_tuned' : 'Nonlin. PR (SVM*)',
             'predictability_nonlin_rf': 'Nonlin. PR (RF)',
              'predictability_nonlin_rf_tuned': 'Nonlin. PR (RF*)',
              'predictability_nonlin_gb': 'Nonlin. PR (GB)',
              'predictability_nonlin_gb_tuned': 'Nonlin. PR (GB*)'}


pr_markers = {'predictability_lin' : 'o',
             'predictability_lin_tuned' : 'o',
             'predictability_nonlin_svc': 's',
            'predictability_nonlin_svc_tuned' : 's',
             'predictability_nonlin_rf': 'D',
              'predictability_nonlin_rf_tuned': 'D',
              'predictability_nonlin_rf_2': 'D',
              'predictability_nonlin_gb': 'v',
              'predictability_nonlin_gb_tuned': 'v'}

mode_markers = {'raw' : 'Raw',
             'all_sessions_significance_nocalib_notune' : 'o',
             'all_sessions_significance_nocalib_tune' : 'P',
             'all_sessions_significance_calib_sigmoid': 's',
             'all_sessions_significance_calib_isotonic': 'v',
              'all_sessions_significance_calib_vennabers': 'D'}

highdim_simulation_labels = {'linear_highdim' : 'Linear',
                             'nonlinear_highdim' : 'Nonlinear',
                             'nonlinear_highdim_noise' : 'Nonlinear\n+ noise'}

score_labels = {'f1_score': 'F1 score',
                'TPR_TNR_av': 'Balanced accuracy',
                'TPR' : 'Sensitivity (TPR)',
                'TNR' : 'Specificity (TNR)',
                'FPR' : 'False positive rate (FPR)'}

parameter_labels = {'coherence_p' : 'Coherence $p$',
                    'signal_strength' : 'Signal strength $w_{Xy}$',
                    'n_samples' : '# of samples',
                    'n_features' : '# of features'}

calibration_modes_labels = {'nocalib_notune' : 'Uncalib.',
                            'calib_sigmoid' : 'Platt',
                            'calib_isotonic' : 'Isotonic',
                            'calib_vennabers' : 'V.A.',
                            'nocalib_tune' : 'Tuned HP',
                            'all_sessions_significance_nocalib_notune': 'Uncalib.',
                            'all_sessions_significance_calib_sigmoid': 'Platt',
                            'all_sessions_significance_calib_isotonic': 'Isotonic',
                            'all_sessions_significance_calib_vennabers': 'V.A.',
                            'all_sessions_significance_nocalib_tune': 'Tuned HP'
                            }

calibration_modes_labels_detailed = {'all_sessions_significance_nocalib_notune': 'Uncalibrated',
                                    'all_sessions_significance_calib_sigmoid': 'Platt scaling',
                                    'all_sessions_significance_calib_isotonic': 'Isotonic calibration',
                                    'all_sessions_significance_calib_vennabers': 'Venn-Abers calibration',
                                    'all_sessions_significance_nocalib_tune': 'Tuned hyperparameters'
                                    }


# --- PALETTES ---

method_palette = {'raw' : sns.xkcd_rgb['mahogany'],
                  'decode_from_confound' : sns.xkcd_rgb['sun yellow'],
                  'distribution_match' : sns.xkcd_rgb['light red'],

                  'linear_confound_regression' : sns.xkcd_rgb['tangerine'],
                  'linear_confound_regression_ridge': sns.xkcd_rgb['tangerine'],
                  'linear_confound_regression_glm': sns.xkcd_rgb['tangerine'],
                  'linear_confound_regression_glm_tuned': sns.xkcd_rgb['tangerine'],

                  'nonlinear_confound_regression' : sns.xkcd_rgb['azure'],
                  'nonlinear_confound_regression_svc': sns.xkcd_rgb['azure'],
                  'nonlinear_confound_regression_svc_tuned': sns.xkcd_rgb['azure'],
                  'nonlinear_confound_regression_rf': sns.xkcd_rgb['azure'],
                  'nonlinear_confound_regression_rf_tuned': sns.xkcd_rgb['azure'],

                  'predictability_lin' : sns.xkcd_rgb['vibrant green'],
                  'predictability_lin_tuned' : sns.xkcd_rgb['vibrant green'],
                  'predictability_nonlin' : sns.xkcd_rgb['bright purple'],
                  'predictability_nonlin_svc': sns.xkcd_rgb['bright purple'],
                  'predictability_nonlin_svc_tuned': sns.xkcd_rgb['bright purple'],
                  'predictability_nonlin_rf': sns.xkcd_rgb['bright purple'],
                  'predictability_nonlin_rf_tuned': sns.xkcd_rgb['bright purple'],
                  'predictability_nonlin_gb': sns.xkcd_rgb['bright purple'],
                  'predictability_nonlin_gb_tuned': sns.xkcd_rgb['bright purple'],

                  'cmi' : sns.xkcd_rgb['steel grey']}


nonlinear_pr_method_palette = {'predictability_nonlin_rf': sns.xkcd_rgb['bright purple'],
                                'predictability_nonlin_gb': sns.xkcd_rgb['royal purple'],}
