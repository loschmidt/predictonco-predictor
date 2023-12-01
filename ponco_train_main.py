# the main script that uses the data uploaded by ponco_data_setup 
# to set all the calculations and output 

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics

from ponco_data_setup import preprocess_data  
from ponco_train_predictors import train_predictors
from ponco_report_eval import fix_axes, report_eval_stats, report_train_results

## SETTINGS
# train/test split settings
same_test_set = False # if True, sequence predictors will be evaluated on the same set as structure predictors
test_frac = 0.2 # in the train/test split
group_method = 'position' # decide on which level test won't overlap: 'protein', 'position', 'mutation' 
bootstr_test = True # set to True to aslo get st.devs for auc and av precision based on boostrap of the test set
bootstr_n = 1000
class_weight = 'balanced' # whether to apply weights for training ('balanced', None)
random_state = 42 # fix for reproducibility; None for randomising

# hyperparameter optimisation
optimize_hyperparams = False # if True, perform hyperparameter search
n_folds = 5 # for cross-validation
cv_scoring = 'roc_auc' # for optimisation based on CV
grid_param_svm = {'penalty': ['l1','l2']} 
grid_param_dt = {'max_depth': np.linspace(1,10,10, dtype='int'),
                 'min_samples_split': np.linspace(2,10,9, dtype='int')}
grid_param_xgb = {'n_estimators': np.linspace(1,40,20, dtype='int'),
                  'max_depth': [1,2,3],
                  'learning_rate': np.logspace(-3,3,20, dtype='int')}

# statistical resampling
evaluate_variance_from_splits = False # if True, retrain the default predictor multiple times to get sterr, no model is saved! No hyperparameters are optimized!
n_rounds = 100 # how many times to simulate when evaluate_variance_from_splits = True; if less than 2, evaluate_variance_from_splits is set to False
quantile = 0.9 # the range for the deviation shown in a graph: [1-quantile quantile]. AUC and av_prec are always reported with st.dev! 
report_cutoffs = np.linspace(0,1,101) # cutoffs to calculate statistics for xgb; set to None if not required

def_svm_str = {'penalty': 'l1', 
               'class_weight': class_weight, 'dual': False, 'max_iter': 10000}
def_dt_str = {'max_depth': 2, 'min_samples_split': 2,
              'class_weight': class_weight, 'random_state': random_state}
def_xgb_str = {'learning_rate': 1, 'max_depth': 1, 'n_estimators': 15,  
               'verbosity': 1, 'eval_metric': 'auc', 'random_state': random_state}

def_svm_seq = {'penalty': 'l1',
               'class_weight': class_weight, 'dual': False,'max_iter': 10000}
def_dt_seq = {'max_depth': 2, 'min_samples_split': 2,
              'class_weight': class_weight, 'random_state': random_state}
def_xgb_seq = {'learning_rate': 1, 'max_depth': 1, 'n_estimators': 9,
               'verbosity': 1, 'eval_metric': 'auc', 'random_state': random_state}

# visualisation and output
skip_paiplots = True # if False, pairplot graphs will be plotted for the input data
print_roc_values = False # prints all the data for roc and pr curves in the following order: [fpt, tpr, threshold] [precision, recall, threshold]
save_model = True # whether to save two models and corresponding feature lists. Always set to False if evaluate_variance_from_splits=True
scoring_list = [metrics.accuracy_score, metrics.f1_score, metrics.balanced_accuracy_score, metrics.matthews_corrcoef] # reports those scores for train/test data !based on the standard cutoff!
scoring_names = ['acc', 'f1','b_acc', 'mcc']  # names for scoring to print
color = iter(plt.cm.Dark2(np.linspace(0, 1, 12))) # colors for graphs
method_name = ["svm", "dt", "xgb"]
graph_fonts = {'size': 16} # settings for the fonts

## MAIN CODE   
# update of fonts
plt.rc('font', **graph_fonts)

# upload and preprocess the data
X_str, str_idx, X_seq, y, groups, \
    f_seq_cols, f_str_cols, f_names_short, \
    baseline_col_name, baseline_cutoffs, ids = \
    preprocess_data(skip_paiplots, group_method)
y_str = y[str_idx]
eval_fig_str, ax = plt.subplots(1,2,figsize=(16, 8)) 

if n_rounds<2:
    evaluate_variance_from_splits = False
    
if not evaluate_variance_from_splits:
    n_rounds = 1
    suppress_output = False 
else:
    random_state = None
    optimize_hyperparams = False
    suppress_output = True
    save_model = False
    bootstr_test = False

def_set_str = [def_svm_str, def_dt_str, def_xgb_str]
def_set_seq = [def_svm_seq, def_dt_seq, def_xgb_seq]
grid_param = [grid_param_svm, grid_param_dt, grid_param_xgb]
train_params = [n_folds, class_weight, random_state, optimize_hyperparams, cv_scoring, grid_param]
stats_str_list = []   
stats_seq_list = []  

for j in range(n_rounds):   
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state) 
    
    # training of the structure-based predictor
    for train_idx, test_idx in gss.split(X_str, y_str, groups=groups[str_idx]):
        baselines = [[f_str_cols.index(item), item, baseline_cutoffs[i]] for (i,item) in enumerate(baseline_col_name) if item in f_str_cols]
        clfs_str, stats_str, base_names_str = \
            train_predictors(scoring_list, scoring_names,
                             baselines,
                             X_str[train_idx], y_str[train_idx],
                             X_str[test_idx], y_str[test_idx],
                             ax, color, "str_", train_params, def_set_str, 
                             bootstr_test, bootstr_n,
                             suppress_output)
    if not suppress_output:
        report_train_results(print_roc_values, stats_str, clfs_str, method_name,
                             [f_names_short.get(s,s) for s in f_str_cols], 
                             prefix="str_")
        fix_axes(ax)
        eval_fig_str.savefig("evaluation_str.svg")
    if save_model:
        clfs_str[2].save_model('xgb_struc.json')
        str_idx_int = np.nonzero(str_idx)[0]
        split_text = np.repeat(['TRAIN'], str_idx_int.size)
        split_text[test_idx] = ['TEST']
        np.savetxt('str_data_split.txt', 
                   np.transpose(
                       np.vstack((str_idx_int.astype(str),
                                  ids[str_idx_int].astype(str)[:,0], 
                                  y_str.astype(str)[:,0],
                                  groups[str_idx_int].astype(str),
                                  split_text))),                          
                   fmt='%s')
     
    # training of the sequence-based predictor
    if not same_test_set:
        if not suppress_output:
            eval_fig_seq, ax = plt.subplots(1,2,figsize=(16, 8))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state) 
        for train_idx, test_idx in gss.split(X_seq, y, groups=groups):
              X_seq_train = X_seq[train_idx]
              y_seq_train = y[train_idx]
              X_seq_test = X_seq[test_idx]
              y_seq_test = y[test_idx]
    else:
        train_idx = np.ones((y.shape[0],), dtype=bool)
        train_idx[np.where(str_idx)[0][test_idx]] = False
        X_seq_train = X_seq[train_idx]
        y_seq_train = y[train_idx]
        X_seq_test = X_seq[~train_idx]
        y_seq_test = y[~train_idx]
    baselines = [[f_seq_cols.index(item), item, baseline_cutoffs[i]] for (i,item) in enumerate(baseline_col_name) if item in f_seq_cols]
    clfs_seq, stats_seq, base_names_seq = \
            train_predictors(scoring_list, scoring_names,
                             baselines,
                             X_seq[train_idx], y[train_idx],
                             X_seq[test_idx], y[test_idx],
                             ax, color, "seq_", train_params, def_set_seq, 
                             bootstr_test, bootstr_n,
                             suppress_output)
    if not suppress_output:
        report_train_results(print_roc_values, stats_seq, clfs_seq, method_name,
                             [f_names_short.get(s,s) for s in f_seq_cols],
                             prefix="seq_" )
        fix_axes(ax)
        eval_fig_seq.savefig("evaluation_seq.svg", format="svg")
        plt.show()
    if save_model:
        clfs_seq[2].save_model('xgb_seq.json')    
        split_text = np.repeat(['TRAIN'], ids.size)
        split_text[test_idx] = ['TEST']
        np.savetxt('seq_data_split.txt', 
                   np.transpose(
                       np.vstack((np.arange(ids.size).astype(str),
                                  ids.astype(str)[:,0], 
                                  y.astype(str)[:,0],
                                  groups.astype(str),
                                  split_text))),                          
                   fmt='%s')
        
    stats_str_list.append(stats_str)
    stats_seq_list.append(stats_seq)

if evaluate_variance_from_splits:
    _, ax_str = plt.subplots(1,2,figsize=(16, 8))
    report_eval_stats(base_names_str, method_name, stats_str_list, ax_str[0], ax_str[1], 
                      color, quantile, prefix='str_', cutoffs=report_cutoffs)
    fix_axes(ax_str)
    _, ax_seq = plt.subplots(1,2,figsize=(16, 8)) 
    report_eval_stats(base_names_seq, method_name, stats_seq_list, ax_seq[0], ax_seq[1], 
                      color, quantile, prefix='seq_', cutoffs=report_cutoffs)
    fix_axes(ax_seq)

# saving the feature names
if save_model:
    with open("cols_struc.txt", "w") as f:
        for s in f_str_cols:
            f.write(str(s) +"\n")
    with open("cols_seq.txt", "w") as f:
        for s in f_seq_cols:
            f.write(str(s) +"\n") 
