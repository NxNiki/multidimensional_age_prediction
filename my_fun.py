
# coding: utf-8

# In[ ]:

def exact_mc_perm_test(xs, ys, nmc):
    import numpy as np
    
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc


def exact_mc_perm_test1(xs, nmc):
    import numpy as np
    
    n = len(xs)
    dbar = np.mean(xs)
    #absx <- np.abs(xs)
    z = []
    for i in range(nmc):
        mn = np.random.choice([-1,1],n)
        xbardash = np.mean(mn*np.abs(xs))
        z.append(xbardash)
    
    pval = (np.sum(z >= np.abs(dbar)) + np.sum(z <= -np.abs(dbar)))/nmc
    return(pval)

def read_brain_feature():
    ## predict age with brain imaging features
    import numpy as np
    import pandas as pd

    # vbm analysis of cat12, 856 subjects. 
    #It contains subjects in the ptsd group. So we need to remove them in training stage.
    cat_vbm = pd.read_csv("data/ROI_catROI_neuromorphometrics_Vgm.csv", delimiter = ",")
    #spm_vbm_aal = pd.read_csv("data/AvgExtract_GMV_AAL.csv", delimiter = "\t")
    print("cat_vbm:")
    print(cat_vbm.shape)
    # print(cat_vbm[:5])

    ## read label FA features:
    label_fa_list = []
    for i in range(6): 
        fa_sub = pd.read_csv("data/WMlabelResults_FA" + str(i+1) + ".csv")
        label_fa_list.append(fa_sub)

    label_fa = pd.concat(label_fa_list, ignore_index = True)
    label_fa.to_csv("data/WMlabelResults_FA_all.csv")
    label_fa = label_fa.drop(['Unnamed: 0'], axis = 1)

    # remove the last column which is empty.
    num_col_fa=len(label_fa.columns) 
    label_fa=label_fa.iloc[:,:num_col_fa-1]

    print("label fa:")
    print(label_fa.shape)
    # print(label_fa[:5])


    ## read tract FA features:
    tract_fa_list = []
    for i in range(6): 
        fa_sub = pd.read_csv("data/WMtractResults_FA" + str(i+1) + ".txt", delimiter = "\t")
        tract_fa_list.append(fa_sub)

    tract_fa = pd.concat(tract_fa_list, ignore_index = True)
    tract_fa.to_csv("data/WMtractResults_FA_all.csv")

    # print("tract fa:")
    # print(tract_fa.shape)
    # print(tract_fa[:5])

    tract_fa = tract_fa.drop(['Unnamed: 0'], axis = 1)
    # remove the last column which is empty.
    num_col_fa=len(tract_fa.columns) 
    tract_fa=tract_fa.iloc[:,:num_col_fa-1]

    print("tract fa:")
    print(tract_fa.shape)
    # print(tract_fa[:5])

    ## read tract MD features:
    tract_md_list = []
    for i in range(6): 
        md_sub = pd.read_csv("data/WMtractResults_MD" + str(i+1) + ".csv")
        tract_md_list.append(md_sub)

    tract_md = pd.concat(tract_md_list, ignore_index = True)
    tract_md.to_csv("data/WMtractResults_MD_all.csv")

    # print("tract md:")
    # print(tract_md.shape)
    # print(tract_md.head(5))

    # print(tract_md.iloc[:,0])
    tract_md = tract_md.drop(['Unnamed: 0'], axis = 1)
    # remove the last column which is empty.
    num_col_md=len(tract_md.columns)
    tract_md=tract_md.iloc[:,:num_col_md-1]

    print("tract md:")
    print(tract_md.shape)
    # print(tract_md.head(5))

    ## read label MD features:
    label_md_list = []
    for i in range(6): 
        md_sub = pd.read_csv("data/WMlabelResults_MD" + str(i+1) + ".txt", delimiter = "\t")
        label_md_list.append(md_sub)

    label_md = pd.concat(label_md_list, ignore_index = True)
    label_md.to_csv("data/WMlabelResults_MD_all.csv")
    label_md = label_md.drop(['Unnamed: 0'], axis = 1)

    # remove the last column which is empty.
    num_col_md=len(label_md.columns) 
    label_md=label_md.iloc[:,:num_col_md-1]

    print("label md:")
    print(label_md.shape)
    # print(label_md[:5])


    # combine vbm and label_fa
    # it's strange that reset_index also removes column names. so we ignore_index in concat label_fa_list.
    # otherwise, we cannot concat cat_vbm and label_fa.
    # cat_vbm.reset_index(drop=True, inplace=True)
    # label_fa_drop.reset_index(drop=True, inplace=True)

    # read resting state features:
    alff = pd.read_csv("data/ALFF_AvgExtract.txt", delimiter = "\t")
    #print(alff.head(5))
    alff = alff.drop(['Unnamed: 0'], axis = 1)
    # remove the last column which is empty.
    num_col_alff=len(alff.columns) 
    alff=alff.iloc[:,:num_col_alff-1]
    print("alff")
    # print(alff.head(5))
    print(alff.shape)
    #print(alff.columns)

    falff = pd.read_csv("data/fALFF_AvgExtract.txt", delimiter = "\t")
    falff = falff.drop(['Unnamed: 0'], axis = 1)
    # remove the last column which is empty.
    num_col_falff=len(falff.columns) 
    falff=falff.iloc[:,:num_col_falff-1] 
    print("falff")
    print(falff.shape)
    #print(falff.columns)
    
    reho = pd.read_csv("data/Reho_AvgExtract.txt", delimiter = "\t")
    reho = reho.drop(['Unnamed: 0'], axis = 1)
    # remove the last column which is empty.
    num_col_reho=len(reho.columns) 
    reho=reho.iloc[:,:num_col_reho-1]
    print("reho")
    print(reho.shape)
    #print(reho.columns)
    subject_id = cat_vbm['names'].to_frame()

    ReHo = pd.concat([cat_vbm.names, reho], axis = 1)
    ALFF = pd.concat([cat_vbm.names, alff], axis = 1)
    fALFF = pd.concat([cat_vbm.names, falff], axis = 1)

    fa = pd.concat([cat_vbm.names, label_fa, tract_fa], axis = 1)
    md = pd.concat([cat_vbm.names, label_md, tract_md], axis = 1)
    
    fa_md_reho_alff = pd.concat([cat_vbm.names, label_fa, tract_fa, label_md, tract_md, reho, alff], axis = 1)

    vbm_fa = pd.concat([cat_vbm, label_fa, tract_fa], axis = 1)
    vbm_md = pd.concat([cat_vbm, label_md, tract_md], axis = 1)
    vbm_dti = pd.concat([cat_vbm, label_fa, tract_fa, label_md, tract_md], axis = 1)

    vbm_reho = pd.concat([cat_vbm, reho], axis = 1)
    vbm_alff = pd.concat([cat_vbm, alff], axis = 1)
    vbm_falff = pd.concat([cat_vbm, falff], axis = 1)

    vbm_alff_reho = pd.concat([cat_vbm, alff, reho], axis = 1)
    vbm_falff_reho = pd.concat([cat_vbm, falff, reho], axis = 1)

    vbm_dti_alff_reho = pd.concat([cat_vbm, label_fa, tract_fa, label_md, tract_md, alff, reho], axis = 1)
    vbm_dti_falff_reho = pd.concat([cat_vbm, label_fa, tract_fa, label_md, tract_md, falff, reho], axis = 1)
    
    brain_feature_list = [[cat_vbm, "GMV"], 
                      [ReHo, "ReHo"],
                      [ALFF, "ALFF"],
                      #[fALFF, "falff"],
                      [fa, "FA"],
                      [md, "MD"],
                      #[vbm_fa, "vbm_fa"],
                      #[vbm_md, "vbm_md"],
                      [vbm_dti, "GMV&DTI"],
                      #[vbm_alff, "vbm_alff"],
                      #[vbm_falff, "vbm_falff"],
                      #[vbm_reho, "vbm_reho"],
                      [vbm_alff_reho, "GMV&rsfMRI"],
                      #[vbm_falff_reho, "vbm_falff_reho"],
                      [fa_md_reho_alff, "DTI&rsfMRI"],
                      [vbm_dti_alff_reho, "Multi-modal"],
                      #[vbm_dti_falff_reho, "vbm_dit_falff_reho"]
                     ]
    
#     # merge dataframes.
#     subject_info_merge = subject_info_all.merge(subject_info, how = "left", on = "SUBJID")
#     # df[["name"]] is still a dataframe so we can merge it with other dataframes.
#     #df["name"] is a series so we cannot merge it.
#     subject_info_merge = subject_info_merge.merge(cat_vbm[["names"]], how = "inner", left_on = "SUBJID", right_on = "names")

#     #subject_info_merge['Sex_x'] = subject_info_merge['Sex_x'].replace({"F", "M"}, {0, 1})
#     subject_info_merge.replace({"Sex_x": {"F": 0, "M": 1}}, inplace = True)
#     subject_info_merge.ptsd = subject_info_merge.ptsd.fillna(999)
    
    #return((brain_feature_list, subject_info_merge))
    return((brain_feature_list, subject_id))

def read_brain_feature_fc():
    ## predict age with brain imaging features
    import numpy as np
    import pandas as pd

    # vbm analysis of cat12, 856 subjects. 
    #It contains subjects in the ptsd group. So we need to remove them in training stage.
    cat_vbm = pd.read_csv("data/ROI_catROI_neuromorphometrics_Vgm.csv", delimiter = ",")
    #spm_vbm_aal = pd.read_csv("data/AvgExtract_GMV_AAL.csv", delimiter = "\t")
    print("cat_vbm:")
    print(cat_vbm.shape)
    # print(cat_vbm[:5])
    subject_id = cat_vbm['names'].to_frame()

    ## read label FA features:
    label_fa_list = []
    for i in range(6): 
        fa_sub = pd.read_csv("data/WMlabelResults_FA" + str(i+1) + ".csv")
        label_fa_list.append(fa_sub)

    label_fa = pd.concat(label_fa_list, ignore_index = True)
    label_fa.to_csv("data/WMlabelResults_FA_all.csv")
    label_fa = label_fa.drop(['Unnamed: 0'], axis = 1)

    # remove the last column which is empty.
    num_col_fa=len(label_fa.columns) 
    label_fa=label_fa.iloc[:,:num_col_fa-1]

    print("label fa:")
    print(label_fa.shape)
    # print(label_fa[:5])


    ## read tract FA features:
    tract_fa_list = []
    for i in range(6): 
        fa_sub = pd.read_csv("data/WMtractResults_FA" + str(i+1) + ".txt", delimiter = "\t")
        tract_fa_list.append(fa_sub)

    tract_fa = pd.concat(tract_fa_list, ignore_index = True)
    tract_fa.to_csv("data/WMtractResults_FA_all.csv")

    # print("tract fa:")
    # print(tract_fa.shape)
    # print(tract_fa[:5])

    tract_fa = tract_fa.drop(['Unnamed: 0'], axis = 1)
    # remove the last column which is empty.
    num_col_fa=len(tract_fa.columns) 
    tract_fa=tract_fa.iloc[:,:num_col_fa-1]

    print("tract fa:")
    print(tract_fa.shape)
    # print(tract_fa[:5])

    ## read tract MD features:
    tract_md_list = []
    for i in range(6): 
        md_sub = pd.read_csv("data/WMtractResults_MD" + str(i+1) + ".csv")
        tract_md_list.append(md_sub)

    tract_md = pd.concat(tract_md_list, ignore_index = True)
    tract_md.to_csv("data/WMtractResults_MD_all.csv")

    # print("tract md:")
    # print(tract_md.shape)
    # print(tract_md.head(5))

    # print(tract_md.iloc[:,0])
    tract_md = tract_md.drop(['Unnamed: 0'], axis = 1)
    # remove the last column which is empty.
    num_col_md=len(tract_md.columns)
    tract_md=tract_md.iloc[:,:num_col_md-1]

    print("tract md:")
    print(tract_md.shape)
    # print(tract_md.head(5))

    ## read label MD features:
    label_md_list = []
    for i in range(6): 
        md_sub = pd.read_csv("data/WMlabelResults_MD" + str(i+1) + ".txt", delimiter = "\t")
        label_md_list.append(md_sub)

    label_md = pd.concat(label_md_list, ignore_index = True)
    label_md.to_csv("data/WMlabelResults_MD_all.csv")
    label_md = label_md.drop(['Unnamed: 0'], axis = 1)

    # remove the last column which is empty.
    num_col_md=len(label_md.columns) 
    label_md=label_md.iloc[:,:num_col_md-1]

    print("label md:")
    print(label_md.shape)
    # print(label_md[:5])


    # combine vbm and label_fa
    # it's strange that reset_index also removes column names. so we ignore_index in concat label_fa_list.
    # otherwise, we cannot concat cat_vbm and label_fa.
    # cat_vbm.reset_index(drop=True, inplace=True)
    # label_fa_drop.reset_index(drop=True, inplace=True)

    # read resting state features:
    fc = pd.read_csv("data/outc01_fc_power264.csv", delimiter = ",")

    fc['0'] = cat_vbm['names']
    fc.rename(columns  = {'0':'names'}, inplace = True)
    
    # fill NaNs with 0:
    fc[np.isnan(fc)] = 0
    fc_drop = fc.drop(['names'], axis = 1)
    
    fa_md_fc = pd.concat([cat_vbm.names, label_fa, tract_fa, label_md, tract_md, fc_drop], axis = 1)

    vbm_fc = pd.concat([cat_vbm, fc_drop], axis = 1)

    vbm_dti_fc = pd.concat([cat_vbm, label_fa, tract_fa, label_md, tract_md, fc_drop], axis = 1)
    
    brain_feature_list = [#[cat_vbm, "GMV"], 
                      [fc, "FC"],
                      #[fa, "FA"],
                      #[md, "MD"],
                      #[vbm_fa, "vbm_fa"],
                      #[vbm_md, "vbm_md"],
                      #[vbm_dti, "GMV&DTI"],
                      #[vbm_alff, "vbm_alff"],
                      #[vbm_falff, "vbm_falff"],
                      #[vbm_reho, "vbm_reho"],
                      [vbm_fc, "GMV&rsfMRIfc"],
                      #[vbm_falff_reho, "vbm_falff_reho"],
                      [fa_md_fc, "DTI&rsfMRIfc"],
                      [vbm_dti_fc, "Multi-modalfc"],
                      #[vbm_dti_falff_reho, "vbm_dit_falff_reho"]
                     ]
    
#     # merge dataframes.
#     subject_info_merge = subject_info_all.merge(subject_info, how = "left", on = "SUBJID")
#     # df[["name"]] is still a dataframe so we can merge it with other dataframes.
#     #df["name"] is a series so we cannot merge it.
#     subject_info_merge = subject_info_merge.merge(cat_vbm[["names"]], how = "inner", left_on = "SUBJID", right_on = "names")

#     #subject_info_merge['Sex_x'] = subject_info_merge['Sex_x'].replace({"F", "M"}, {0, 1})
#     subject_info_merge.replace({"Sex_x": {"F": 0, "M": 1}}, inplace = True)
#     subject_info_merge.ptsd = subject_info_merge.ptsd.fillna(999)
    
    #return((brain_feature_list, subject_info_merge))
    return((brain_feature_list, subject_id))


def get_feature(brain_feature, subject_info, match_label = ["names", "SUBJID"], y_label = ["age_at_cnb"], x_label = ["Sex"]):
    import numpy as np
    
    # merge subject info with brain imaging data:
#     print(brain_feature.head())
#     print(subject_info.head())
    
    subject_info_columns = [match_label[1]] + x_label + y_label
    
    feature_merge = subject_info[subject_info_columns].merge(brain_feature, how = "inner", left_on = match_label[1],right_on = match_label[0])
    y = feature_merge[y_label].values
    y = y.flatten()
    
    subjid = feature_merge[[match_label[0]]].values
    subjid = subjid.flatten()

    # remove duplicated and unneeded features.
    #print(list(feature_merge.columns.values))
    remove_columns = match_label + y_label
    feature_merge = feature_merge.drop(columns = remove_columns)
    # convert string to numeric values.
    #print(subject_info_merge_drop)

    X = feature_merge.values
    #X = preprocessing.scale(feature_merge.values)

    #print(X.mean(axis = 0))
    #print(X.std(axis = 0))
    
#     print("get_feature():")
#     print(X)
#     print(y)

    # remove brain imaging features with zeros more than 80% of all data.
    # we always want the 1 st column which is gender:
    X[:,0] = X[:,0]+1
    nonzero_count = np.count_nonzero(X, axis = 0)
#     print(nonzero_count)
    allowed_zeros = .8*len(y)
    col_idx = np.where(nonzero_count>allowed_zeros)
    X = X[:, col_idx[0]]

    # remove NaNs:
    nan_idx = np.isnan(y)
    y = y[~nan_idx]
    subjid = subjid[~nan_idx]
    X = X[~nan_idx,:]
    
    return((X, y, subjid))


# def qudratic_fun(x, a, b, c):
#     return a + b*x + c**2

def qudratic_r_squared(chro_age, brain_age):
    """
    compute the r squared of curve fit of chro_age vs. brain age.
    1. Use qudratic function to fit brain age with chronological age.
    2. Then, compute r squared of brain age and predicted brain age with qudratic fit model.
    """
    
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    
    def qudratic_fun(x, a, b, c):
        return a + b*x + c*(x**2)
    
    popt, pcov = curve_fit(qudratic_fun, chro_age, brain_age)
    brain_age_pred = qudratic_fun(chro_age, popt[0], popt[1], popt[2])

    r_square = r2_score(brain_age, brain_age_pred)
    return(r_square)

def qudratic_r_squared_gender(chro_age, brain_age, gender):
    """
    compute the r squared of curve fit of chro_age vs. brain age.
    1. Use qudratic function to fit brain age with chronological age and gender.
    2. Then, compute r squared of brain age and predicted brain age with qudratic fit model.
    """
    
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    
    def qudratic_fun(x, a, b, c, d, e, f):
        x1, x2 = x
        #return a + b*x1 + c*(x1**2) + d*x2 + e*x1*x2 f*(x1**2)*x2
        return a + b*x1 + c*(x1**2) + d*x2 + e*x1*x2 + f*(x1**2)*x2
    
    popt, pcov = curve_fit(qudratic_fun, (chro_age, gender), brain_age)
    brain_age_pred = qudratic_fun((chro_age, gender), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

    r_square = r2_score(brain_age, brain_age_pred)
    return(r_square)
    

def run_model(model, subject_info, brain_feature_list, kf, fit_method = 1):
    """
    model: regression model should have model.fit and model.predict methods.
    brain_feature_list: 
    kf: k fold CV index (StratifiedKFold)
    fit_method: 0: for ridge, svr, pgr and keras dnn (model with fit and predict)
                1: for grid_cv (model with fit and best_estimator_.predict)
    """
    
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    #from sklearn.metrics import r2_score
        
    
    # create empty dataframe to save results:
    column_index = pd.MultiIndex.from_product([['Pearson r', 'R square', 'MAE', 'rmse'],
                                               ['boot' + str(i) for i in range(1, kf.n_splits+1, 1)]])
    
    row_index = [el[1] for el in brain_feature_list]
    result_table = pd.DataFrame(index = row_index, columns = column_index)

    plot_data = pd.DataFrame(columns = ['feature', 'SUBJID', 'CV', 'chronological age', 'brain age'])

    for brain_feature, feature_name in brain_feature_list: 
        print('processing on: %s --------------------------', feature_name)
        
        X, y, subjid = get_feature(brain_feature, subject_info)
        
        i = 1
        #for train_index, test_index in kf.split(X, y):
        for train_index, test_index in kf.split(X):
            print('run_model on CV: %d' % i)

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            subjid_test = subjid[test_index]
            
            # normalize X_train and X_test based on mean and sd of X_train.
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)  
            X_test = scaler.transform(X_test)
            
            seed = 7
            np.random.seed(seed)           
            
            if fit_method==0:
                
                fit_result = model.fit(X_train, y_train)
                y_prediction = model.predict(X_test)
                
            elif fit_method ==1:
                
                fit_result = model.fit(X_train, y_train)
                y_prediction = model.best_estimator_.predict(X_test)
                
                print("Best: %f using %s" % (fit_result.best_score_, fit_result.best_params_))
        #         means = grid_gpr_result.cv_results_['mean_test_score']
        #         stds = grid_gpr_result.cv_results_['std_test_score']
        #         params = grid_gpr_result.cv_results_['params']      
        #         for mean, stdev, param in zip(means, stds, params):
        #             print("%f (%f) with: %r" % (mean, stdev, param))

            
            #result_table['R square', 'boot' + str(i)][feature_name] = r2_score(y_test, y_prediction)
            #result_table['R square', 'boot' + str(i)][feature_name] = qudratic_r_squared(y_test, y_prediction)
            result_table['R square', 'boot' + str(i)][feature_name] = qudratic_r_squared_gender(y_test, y_prediction, X_test[:,1])
            result_table['Pearson r', 'boot' + str(i)][feature_name] = np.corrcoef(y_test, y_prediction)[0,1]
            result_table['rmse', 'boot' + str(i)][feature_name] = np.sqrt(np.mean(np.square(y_test - y_prediction)))
            result_table['MAE', 'boot' + str(i)][feature_name] = np.mean(np.abs(y_test - y_prediction))

            plot_data_cv = pd.DataFrame(columns = ['feature', 'SUBJID', 'CV', 'chronological age', 'brain age'])
            plot_data_cv['chronological age'] = y_test
            plot_data_cv['brain age'] = y_prediction
            plot_data_cv.loc[:,'CV'] = i
            plot_data_cv.loc[:,'feature'] = feature_name
            plot_data_cv['SUBJID'] = subjid_test

            plot_data = pd.concat([plot_data,plot_data_cv])
            i = i+1
            
    return((result_table, plot_data))


def summary_result(result_table):
    import pandas as pd
    
    column_index = pd.MultiIndex.from_product([['Pearson r', 'R square', 'MAE', 'rmse'], \
                                               ['mean', 'std']])
    
    result_table_summary = pd.DataFrame(columns = column_index)

    result_table_summary['Pearson r', 'mean'] = result_table['Pearson r'].mean(axis = 1)
    result_table_summary['Pearson r', 'std'] = result_table['Pearson r'].std(axis = 1)

    result_table_summary['R square', 'mean'] = result_table['R square'].mean(axis = 1)
    result_table_summary['R square', 'std'] = result_table['R square'].std(axis = 1)
    
    result_table_summary['MAE', 'mean'] = result_table['MAE'].mean(axis = 1)
    result_table_summary['MAE', 'std'] = result_table['MAE'].std(axis = 1)

    result_table_summary['rmse', 'mean'] = result_table['rmse'].mean(axis = 1)
    result_table_summary['rmse', 'std'] = result_table['rmse'].std(axis = 1)
    
    #print('summary_result:')
    #print(result_table_summary)
    
    result_table2 = result_table[['Pearson r', 'R square']].reset_index(level = 0, inplace = False)
    result_table3 = result_table2.rename(columns ={'index':'feature'}, inplace = False)
    result_accuracy_plot = pd.melt(result_table3, id_vars=['feature'], var_name = "boot")
    
#     result_table2 = result_table['R square'].reset_index(level = 0, inplace = False)
#     result_table3 = result_table2.rename(columns ={'index':'feature'}, inplace = False)
#     result_r2_plot = pd.melt(result_table3, id_vars=['feature'], \
#                                    value_name = "R square", var_name = "boot")
    #print(result_accuracy_plot)
    return((result_table_summary, result_accuracy_plot))
    

    
def plot_result(result_table, plot_data, nfold):
    # plot regression results:
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set(color_codes=True)
    
    plot_data_mi = plot_data.set_index(['feature','CV']).sort_index()
    #plot_data_mi = plot_data.set_index(['feature','CV'])

    #feature_name = [el[1] for el in brain_feature_list]
    feature_name = plot_data_mi.index.unique(level = 0)
    
    i_plt = 1
    sns.set(rc={'figure.figsize':(30,15)}, font_scale = 2)
    
    for i_feature in feature_name:
        #print(i_feature)
        
        plt.figure(i_plt)
        i_plt = i_plt + 1

        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)

        for i_cv in range(nfold):

            label1 = "hc cv%d: r = %.2f, MAE = %.2f" % \
            (i_cv+1, result_table['R square', 'boot' + str(i_cv+1)][i_feature], \
             result_table['MAE', 'boot'+str(i_cv+1)][i_feature])

            chro_age = plot_data_mi.loc[i_feature, i_cv+1]['chronological age']
            brain_age = plot_data_mi.loc[i_feature, i_cv+1]['brain age']
            sns.regplot(x=chro_age, y=brain_age, label = label1, ax = ax1, order = 2, truncate=True)
#             sns.regplot(x=chro_age, y=brain_age, label = label1, ax = ax1)
            ax1.legend(loc="lower right")
            ax1.set(xlabel='chronological age', ylabel='predicted age')

            label2 = "hc cv%d" % (i_cv+1)
            sns.regplot(x=chro_age, y=brain_age - chro_age, label = label2, ax = ax2, order = 2, truncate=True)
#             sns.regplot(x=chro_age, y=brain_age - chro_age, label = label2, ax = ax2)
            ax2.legend(loc="upper left")
            ax2.set(xlabel='chronological age', ylabel='brain age gap')

        plt.suptitle(i_feature)
        plt.show()
        
def plot_result_mergecv(result_table_summary, plot_data):
    # plot regression results:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns; sns.set(color_codes=True)

    plot_data_mi = plot_data.set_index(['feature']).sort_index()
    plot_data_mi['age gap'] = -plot_data_mi['chronological age']+plot_data_mi['brain age']
    
    #feature_name = [el[1] for el in brain_feature_list]
    feature_name = plot_data_mi.index.unique(level = 0)
    
    i_plt = 1
    sns.set(rc={'figure.figsize':(15,15)}, font_scale = 2.5)
    for i_feature in feature_name:
        # chronological age vs brain age:
        plt.figure(i_plt)
        i_plt = i_plt + 1

        label1 = "%s: $R^2$ = %.3f, MAE = %.3f" % \
        (i_feature, result_table_summary.loc[i_feature, 'R square']['mean'], \
         result_table_summary.loc[i_feature, 'MAE']['mean'])

        g = sns.lmplot(x='chronological age', y='brain age', hue = 'CV', \
                       data = plot_data_mi.loc[i_feature], fit_reg=False,\
                       scatter_kws={'alpha':0.5}, x_jitter = .2,\
                       height=10.27, aspect=10.27/10.27, legend = False)
    
        sns.regplot(x='chronological age', y='brain age', \
                    data = plot_data_mi.loc[i_feature], scatter=False, ax=g.axes[0,0], order = 2)

        text_y = plot_data_mi.loc[i_feature, 'brain age'].max()+.2
        g.axes[0,0].text(8, text_y, label1, fontsize = 25)
        
        #g.axes[0].set_ylim(8, 23)
        plt.show()
        
        # chronological age vs age gap:
        plt.figure(i_plt)
        i_plt = i_plt + 1

        corr = np.corrcoef(plot_data_mi.loc[i_feature, 'age gap'].values, 
                           plot_data_mi.loc[i_feature, 'chronological age'].values)
        label1 = "%s: r = %.3f" % (i_feature, corr[0,1])

        g = sns.lmplot(x='chronological age', y='age gap', hue = 'CV', \
                       data = plot_data_mi.loc[i_feature], fit_reg=False,\
                       scatter_kws={'alpha':0.7}, x_jitter = .2,\
                       height=10.27, aspect=10.27/10.27, legend = False)
    
        sns.regplot(x='chronological age', y='age gap', \
                    data = plot_data_mi.loc[i_feature], scatter=False, ax=g.axes[0,0], order = 2)

        text_y = plot_data_mi.loc[i_feature, 'age gap'].max()+.2
        g.axes[0,0].text(8, text_y, label1, fontsize = 25)
        
        plt.show()
        