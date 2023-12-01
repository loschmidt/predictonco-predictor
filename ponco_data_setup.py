# the script uploads the data for training and testing

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize, LabelEncoder

def preprocess_data(skip_plotting=True, group_method='mutation'):
    # Output:
    # X_str - numpy array - a subset with structure features 
    # str_idx - numpy boolean vector - str_idx[i]=True if X_seq[i] belongs to X_str
    # X_seq - numpy array - full set  with sequence features 
    # y - numpy vector - full set of labels 
    # groups - numpy vector - 
    # feature_seq_cols - list - names of sequence features
    # feature_str_cols - list - names of stucture-only features
    # f_names_short - dictionary - shortening of feature names for plotting
    # baseline_col_name - names of the feature that is used as baseline
    
    train_data_file = 'zenodo-features-2023-10-18.tsv' # training data, tab-delimited
    
    feature_seq_cols = ["protein_type", # PROTO_ONCOGENE will be converted to 1
                    "essential", 
                    "predictsnp",
                    "essential_residues_all",
                    "conservation",
                    "msa_data",
                    "domain"# converted to one-hot and moved to the end
                    ]
    feature_str_cols = [ # will be added to feature_seq_cols
                    "pocket",
                    "foldx",
                    "rosetta",
                    "pka_num",
                    "pka_min",
                    "pka_max"]
    label_cols = ["label"] # column for the label
    structure_cols = ["structure"] # column indicating structural data
    baseline_col_name = ["predictsnp", "conservation", "rosetta", "foldx"]  # columns for baselining
    baseline_cutoffs = [50, 3, 0, 0]
    id_cols = ["id"] # column for ids; it is used to group the data by protein + position
    
    f_names_short = { # name shortening for better visualisation
                    "protein_type": "pr_type",
                    "essential": "ess?", 
                    "structure": "struct?",
                    "predictsnp": "predsnp", 
                    "essential_residues_all": "n_ess",
                    "conservation": "cons",
                    "msa_data": "cons_msa",
                    "domain": "dom_type"
                    }

    data = pd.read_table(train_data_file,
                         usecols = feature_seq_cols
                                     + feature_str_cols
                                     + label_cols
                                     + structure_cols
                                     + id_cols)
    print(data)
    
    # replacing "protein_type" with binary 1 = PROTO_ONCOGENE
    data["protein_type"] = label_binarize(
        data["protein_type"], 
        classes=['TUMOR_SUPPRESSOR', 'PROTO_ONCOGENE']) 
    
    # plot data in pairplot
    if not skip_plotting:
        sns.set_theme(style="ticks")    
        seq_data_plot = data[feature_seq_cols+label_cols+structure_cols].copy()
        # turning "dom_type" into numbers
        le = LabelEncoder()
        le.fit(seq_data_plot["domain"])
        seq_data_plot["domain"] = le.transform(seq_data_plot["domain"])
        sns.pairplot(seq_data_plot.rename(columns=f_names_short), 
                     hue = label_cols[0],
                     hue_order = ['Benign', 'Oncogenic'])
        
        str_data_plot = data[feature_seq_cols+feature_str_cols+label_cols].copy()
        str_data_plot = str_data_plot[data["structure"]==1]
        str_data_plot.drop("domain", axis=1)
        sns.pairplot(str_data_plot.rename(columns=f_names_short), 
                     hue = label_cols[0],
                     hue_order = ['Benign', 'Oncogenic'])
    
    # remove "domain" and add its one hot encoding to the end
    domain_1hot = pd.get_dummies(data["domain"])
    data = data.drop("domain", axis=1)
    feature_seq_cols.remove("domain")
    feature_str_cols += feature_seq_cols
    data = data.join(domain_1hot)
    feature_seq_cols += list(domain_1hot.columns)
    
    y = data[label_cols].to_numpy()
    y = label_binarize(y, classes=['Benign', 'Oncogenic']) # 1 = oncogenic
    
    ids = data[id_cols].to_numpy()
    if group_method=='protein':
        groups = np.array([x[0][:x[0].find("_")] for x in ids])
    elif group_method=='position':
        groups = np.array([x[0][:-1] for x in ids])
    else:
        groups = np.array([x[0] for x in ids]) # group by mutations
        if group_method!='mutation':
            print("Error: group_method value is not recognized, applying MUTATION by default")
    
    str_idx = data["structure"]==1
    X_str = data[feature_str_cols][str_idx].to_numpy()
    X_seq = data[feature_seq_cols].to_numpy()
    return  X_str, str_idx.to_numpy(), X_seq,\
            y, groups, feature_seq_cols, feature_str_cols, \
            f_names_short, baseline_col_name, baseline_cutoffs, ids
