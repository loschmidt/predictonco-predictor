from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import xgboost as xgb
import numpy as np


LANGUAGE = {
    "ACTIVE": None,  # change locally to "sk" to activate Slovak language
    "DATABASE": {
        "sk": {
            "ROC curve": "Krivka ROC",
            "True Positive Rate": "Pravdivá pozitivita",
            "False Positive Rate": "Falošná pozitivita",
            "area": "plocha",
            "Precision-Recall curve": "Krivka Precision-Recall",
            "Precision": "Presnosť (Precision)",
            "Recall": "Citlivosť (Recall)",
            "average precision": "priemerná presnosť",
        },
    },
}

def _(phrase):
    return LANGUAGE["DATABASE"].get(LANGUAGE["ACTIVE"], {}).get(phrase, phrase)


def fix_axes(ax):
    # ax[0] is the ROC plot, ax[1] is the precision-recall curve
    
    ax[0].plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel(_("False Positive Rate"))
    ax[0].set_ylabel(_("True Positive Rate"))
    ax[0].set_title(_("ROC curve"))
    ax[0].legend(loc="lower right")
    
    ax[1].set_xlim(0.0, 1.0)
    #ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel(_("Recall"))
    ax[1].set_ylabel(_("Precision"))
    ax[1].set_title(_("Precision-Recall curve"))
    ax[1].legend(loc="lower right")

def report_eval(scoring_list, scoring_names, method_name,
                y_train, y_pred_train, y_test, y_pred_test,
                bootstr_test, bootstr_n,
                roc_ax, prc_ax, color, cutoff, 
                suppress_output=False):     
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test)
    roc_auc = metrics.auc(fpr, tpr)
    roc_output = [fpr, tpr, thresholds, roc_auc]
    
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_test)
    prc_ave = metrics.average_precision_score(y_test, y_pred_test)
    prc_output = [precision, recall, thresholds, prc_ave]
    
    if bootstr_test:
        n_points = y_test.size
        roc_auc_list = np.zeros((bootstr_n,))
        prc_ave_list = np.zeros((bootstr_n,))
        
        for j in range(bootstr_n):
            idx_sample = np.random.randint(0, n_points, size=n_points)  
            fpr_sample, tpr_sample, _ = metrics.roc_curve(y_test[idx_sample], y_pred_test[idx_sample])
            roc_auc_list[j] = metrics.auc(fpr_sample, tpr_sample)
            prc_ave_list[j] = metrics.average_precision_score(y_test[idx_sample], y_pred_test[idx_sample])
        
        label_roc = method_name +\
                " (area = %0.3f +- %0.3f)" % (np.mean(roc_auc_list), np.std(roc_auc_list))
        label_prc = method_name +\
                " (av. precision = %0.3f +- %0.3f)" % (np.mean(prc_ave_list), np.std(prc_ave_list))
        if not suppress_output:
            print("\n" + method_name + ", roc_auc/av_prec: %.3f+-%0.3f/%.3f+-%0.3f"  
                  % (roc_auc, np.std(roc_auc_list), prc_ave, np.std(prc_ave_list)))
    else:
         label_roc = method_name +" (area = %0.3f)" % (roc_auc)
         label_prc = method_name +" (av. precision = %0.3f)" % (prc_ave)
         if not suppress_output:
            print("\n" + method_name + ", roc_auc/av_prec: %.3f/%.3f"  
                  % (roc_auc, prc_ave))
    
    if not suppress_output:
        c = next(color)
        roc_ax.plot(fpr, tpr, c=c, lw=2, label= label_roc)
        prc_ax.plot(recall, precision, c=c, lw=2, label= label_prc)
    
    for c in cutoff:
        if not suppress_output:
            print("\n" + method_name + ", applied cutoff = %.2f:" % c)
        for i, scorer in enumerate(scoring_list):
            score_train = scorer(y_train, y_pred_train>c)
            score_test = scorer(y_test, y_pred_test>c)
            if not suppress_output:
                print("Train/test " + scoring_names[i] + ": %.2f/%.2f"  
                      % (score_train, score_test))
        if not suppress_output:
            print("Train confusion matrix ")
            print(metrics.confusion_matrix(y_train, y_pred_train>c))
            print("Test confusion matrix ")
            print(metrics.confusion_matrix(y_test, y_pred_test>c))
            print("")
    
    return [roc_output, prc_output] 

def report_eval_stats(base_names, method_name, stats_list, roc_ax, prc_ax, color, q, prefix="", n_points=30, cutoffs=None):
    # stats_list[i] = [bsl_stats, svm_stats, dt_stats, xgb_stats], bsl_stats = [bsl_stats1, bsl_stats2,...]
    # *_stats = [roc_output, prc_output]
    # roc_output = [fpr, tpr, thresholds, roc_auc]
    # prc_output = [precision, recall, thresholds, prc_ave]
    
    bsl_stats = [[s[0][i] for s in stats_list] for i in range(len(base_names))]
    svm_stats = [s[1] for s in stats_list]
    dt_stats = [s[2] for s in stats_list]
    xgb_stats = [s[3] for s in stats_list]
    plot_x = np.linspace(0,1,n_points)
    print_names = method_name+base_names
    
    for j,stats in enumerate([svm_stats, dt_stats, xgb_stats]+bsl_stats):
        c = next(color)
        c_fill = c.copy()
        c_fill[3] = 0.2
        
        tpr_vals = np.array([np.interp(plot_x, np.array(s[0][0]), np.array(s[0][1])) for s in stats])      
        auc_vals = np.array([s[0][3] for s in stats])
        tpr_means = np.mean(tpr_vals, axis=0)
        tpr_quan = np.quantile(tpr_vals, [1-q, q], axis=0)
        roc_ax.plot(
            plot_x, tpr_means,
            c=c,
            lw=2,
            label= prefix + print_names[j] +\
                " (area = %0.3f +- %0.3f)" % (np.mean(auc_vals), np.std(auc_vals))
        )
        roc_ax.fill_between(plot_x, tpr_quan[0], tpr_quan[1], color=c_fill)
        
        prc_vals = np.array([np.interp(plot_x, np.flip(np.array(s[1][1])), np.flip(np.array(s[1][0]))) for s in stats])      
        avp_vals = np.array([s[1][3] for s in stats])
        prc_means = np.mean(prc_vals, axis=0)
        prc_quan = np.quantile(prc_vals, [1-q, q], axis=0)
        prc_ax.plot(
            plot_x, prc_means,
            c=c,
            lw=2,
            label= prefix +  print_names[j] +\
                " (av. precision = %0.3f +- %0.3f)" % (np.mean(avp_vals), np.std(avp_vals))
        )
        prc_ax.fill_between(plot_x, prc_quan[0], prc_quan[1], color=c_fill)
        
    if cutoffs is not None:
        # threshold fpr tpr precision recall
        cutoff_stats = np.zeros((len(cutoffs),9))
        cutoff_stats[:,0] = cutoffs
        
        stat_vals = np.array([np.interp(cutoffs, np.flip(np.array(s[0][2])), np.flip(np.array(s[0][0]))) for s in xgb_stats]) 
        cutoff_stats[:,1] = np.mean(stat_vals, axis=0)
        cutoff_stats[:,5] = np.std(stat_vals, axis=0)
        
        stat_vals = np.array([np.interp(cutoffs, np.flip(np.array(s[0][2])), np.flip(np.array(s[0][1]))) for s in xgb_stats]) 
        cutoff_stats[:,2] = np.mean(stat_vals, axis=0)
        cutoff_stats[:,6] = np.std(stat_vals, axis=0)
        
        stat_vals = np.array([np.interp(cutoffs, np.array(s[1][2]), np.array(s[1][0])[:-1]) for s in xgb_stats]) 
        cutoff_stats[:,3] = np.mean(stat_vals, axis=0)
        cutoff_stats[:,7] = np.std(stat_vals, axis=0)
        
        stat_vals = np.array([np.interp(cutoffs, np.array(s[1][2]), np.array(s[1][1])[:-1]) for s in xgb_stats]) 
        cutoff_stats[:,4] = np.mean(stat_vals, axis=0)
        cutoff_stats[:,8] = np.std(stat_vals, axis=0)
        np.savetxt(prefix+'xgb_stats_thre_fpr_tpr_pre_rec_stdvs.txt', cutoff_stats)    
        
def report_train_results(print_roc_values, stats, clfs, method_name, feature_names, prefix=""):
    with np.printoptions(precision=3, suppress=True):
        if print_roc_values:
            np.savetxt(prefix+'xgb_fpr_tpr_thresholds.txt',
                    np.transpose(np.array([stats[3][0][0], stats[3][0][1], stats[3][0][2]])))
            np.savetxt(prefix+'xgb_precision_recall_thresholds.txt',
                   np.transpose(np.array([stats[3][1][0][:-1], stats[3][1][1][:-1], stats[3][1][2]]))) 
            [print(prefix+method_name[i]+"\nfpr, tpr, roc_thresholds",
                   np.transpose(np.array([s[0][0], s[0][1], s[0][2]])), sep='\n') for i,s in enumerate(stats[1:])]
            [print(prefix+method_name[i]+"\nprecision, recall, thresholds",
                   np.transpose(np.array([s[1][0][:-1], s[1][1][:-1], s[1][2]])), sep='\n') for i,s in enumerate(stats[1:])] 
        print(prefix+'svm weights: ', clfs[0].coef_)
    
    plt.figure(figsize=(50,50))
    plot_tree(clfs[1], filled=True, rounded=True) 
    
    plt.figure()
    plt.bar(feature_names, clfs[2].feature_importances_)
    plt.xticks(rotation='vertical')
    plt.figure()
    #xgb.plot_tree(clfs[2], num_trees=0, rankdir='LR')