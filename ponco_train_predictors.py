# the script performs training and hyperparmeter optimisation 
# of a set of predictors (Baseline, SVM, DT, XGBoost)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from ponco_report_eval import report_eval

def optimise_predictor(clf_try, X_train, y_train, grid_param, 
                       scoring='roc_auc', cv=5, suppress_output=False):
    grid_clf = GridSearchCV(clf_try, grid_param, scoring = scoring, cv=cv,
                            return_train_score=True)
    grid_clf.fit(X_train, np.ravel(y_train))
    
    if not suppress_output:
        print("Selected settings: ", grid_clf.best_params_)
        if len(grid_param)==1:
            scores_cv = grid_clf.cv_results_['mean_test_score']
            scores_cv_std = grid_clf.cv_results_['std_test_score']
            scores = grid_clf.cv_results_['mean_train_score']
            scores_std = grid_clf.cv_results_['std_train_score']
            plt.figure()
            param = [[k, grid_param[k]] for k in grid_param]
            param_name = param[0][0]
            param_values = param[0][1]
            plt.plot(param_values, scores, 'b')
            plt.plot(param_values, scores + scores_std, 'b--')
            plt.plot(param_values, scores - scores_std, 'b--')
            plt.plot(param_values, scores_cv, 'r')
            plt.plot(param_values, scores_cv + scores_cv_std, 'r--')
            plt.plot(param_values, scores_cv - scores_cv_std, 'r--')
            plt.xlabel(param_name)
            plt.ylabel("CV_" + scoring)
            
    return grid_clf.best_estimator_
    
def train_predictors(scoring_list, scoring_names, baselines,
                     X_train, y_train, X_test, y_test,
                     ax, color, prefix, train_params, def_set,
                     bootstr_test=False, bootstr_n=0,
                     suppress_output=False):
    # returns a scikit-learn classifier-class object
    # CAUTION! when you change the type of predictior, 
    # check which *.predict() method is used in the actual predictor! 
    # different classifiers return different ranges (label, score, probability)
    
    n_folds = train_params[0] # for cross-validation
    class_weight = train_params[1] # 'balanced or 'None' for training'; affects SVM and DT
    random_state = train_params[2]
    flag_optim = train_params[3] # True - will attempt to optimise some hyperparams by CV
    cv_scoring = train_params[4] # which scoring to use in CV
    grid_param = train_params[5]
    
    # defult settings for predictors 
    def_svm = def_set[0]
    def_dt = def_set[1]
    def_xgb = def_set[2]
    
    if class_weight=='balanced':
        def_xgb['scale_pos_weight'] = 1-np.sum(y_train)/y_train.size
    
    
    # gridsearch paramewters used when flag_optim = True, set grid to None to skip
    grid_param_svm = grid_param[0]
    grid_param_dt = grid_param[1]
    grid_param_xgb = grid_param[2]
    
    # main code
    # baseline models
    bsl_stats = []
    bsl_names = []
    for baseline_item in baselines:
        base_col = baseline_item[0]
        base_name = baseline_item[1]
        base_cutoff = baseline_item[2]
        
        base_stats = report_eval(scoring_list, scoring_names, prefix+base_name,
                                 y_train, X_train[:,base_col], 
                                 y_test, X_test[:,base_col],
                                 bootstr_test, bootstr_n,
                                 ax[0], ax[1], color, cutoff=[base_cutoff], 
                                 suppress_output=suppress_output)
        bsl_stats.append(base_stats)
        bsl_names.append(base_name)
    
    # SVM
    clf_try = LinearSVC(**def_svm)
    if flag_optim and grid_param_svm is not None:
        clf_svm = optimise_predictor(clf_try, X_train, y_train, grid_param_svm, 
                                     scoring=cv_scoring, cv=n_folds, 
                                     suppress_output=suppress_output)
    else:
        clf_svm = clf_try.fit(X_train, np.ravel(y_train))
    svm_stats = report_eval(scoring_list, scoring_names, prefix+"svm",
                            y_train, clf_svm.decision_function(X_train),
                            y_test, clf_svm.decision_function(X_test),
                            bootstr_test, bootstr_n,
                            ax[0], ax[1], color, cutoff=[0], 
                            suppress_output=suppress_output)
    
    # decision tree
    clf_try = DecisionTreeClassifier(**def_dt)
    if flag_optim and grid_param_dt is not None:
        clf_dt = optimise_predictor(clf_try, X_train, y_train, grid_param_dt, 
                                    scoring=cv_scoring, cv=n_folds, 
                                    suppress_output=suppress_output)
    else:
        clf_dt = clf_try.fit(X_train, y_train)    
    dt_stats = report_eval(scoring_list, scoring_names, prefix+"dt",
                           y_train, clf_dt.predict_proba(X_train)[:,1],
                           y_test, clf_dt.predict_proba(X_test)[:,1],
                           bootstr_test, bootstr_n,
                           ax[0], ax[1], color, cutoff=[0.5], 
                           suppress_output=suppress_output)

    # XGBoost
    clf_try = xgb.XGBClassifier(**def_xgb)
    if flag_optim and grid_param_xgb is not None:
        clf_xgb = optimise_predictor(clf_try, X_train, y_train, grid_param_xgb, 
                                     scoring=cv_scoring, cv=n_folds, 
                                     suppress_output=suppress_output)
    else:
        clf_xgb = clf_try.fit(X_train, y_train)
    xgb_stats = report_eval(scoring_list, scoring_names, prefix+"xgb",
                            y_train, clf_xgb.predict_proba(X_train)[:,1],
                            y_test, clf_xgb.predict_proba(X_test)[:,1],
                            bootstr_test, bootstr_n,
                            ax[0], ax[1], color, cutoff=[0.5], 
                            suppress_output=suppress_output)

    return [clf_svm, clf_dt, clf_xgb], [bsl_stats, svm_stats, dt_stats, xgb_stats], bsl_names
