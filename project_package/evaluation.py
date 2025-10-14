"""
Module to import advanced evaluation test for model.
"""
import warnings
from typing import List,Tuple,Union
import pandas as pd
import numpy as np
import shap
from sklearn.base import clone,is_classifier,is_regressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.metrics import r2_score,make_scorer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

def ablation_test_old(
    model,
    train_data,
    test_data,
    target_col,
    scorer = 'r2',
    remove_features: list = None
):
    scores = []
    feature_cols = train_data.columns.drop(target_col)
    base_score = scorer(
        test_data[target_col],
        model.predict(test_data.drop(columns=[target_col]))
        )
    if remove_features is None:
        for col in feature_cols:
            test_model = clone(model)
            abla_train = train_data.drop(columns=[target_col,col])
            abla_test = test_data.drop(columns=[target_col,col])
            test_model.fit(abla_train,train_data[target_col])

            test_score = scorer(test_data[target_col],test_model.predict(abla_test))
            scores.append(test_score)
    else:
        for features in remove_features:
            test_model = clone(model)
            if type(features) ==list:
                remove_cols = features + [target_col]
            else:
                remove_cols = [target_col,features]
            abla_train = train_data.drop(columns=remove_cols)
            abla_test = test_data.drop(columns=remove_cols)
            test_model.fit(abla_train,train_data[target_col])

            test_score = scorer(test_data[target_col],test_model.predict(abla_test))
            scores.append(test_score)

    scores = np.array(scores)
    result_df = pd.DataFrame({
        'remove_features':feature_cols if remove_features is None else remove_features,
        'score':scores
    })
    
    result_df['delta'] = base_score - result_df['score']

    return result_df, base_score

def ablation_test(
    model,
    train_data,
    test_data,
    target_col,
    scorer = make_scorer(r2_score),
    cv=10,
    stratify = True,
    remove_features: list = None,
    n_jobs = None,
    random_state = None
):
    cross_scores = []
    test_scores = []
    feature_cols = train_data.columns.drop(target_col)
    
    if stratify:
        cv = StratifiedKFold(n_splits=cv, random_state=random_state)

    base_test_score =  scorer(
        model,
        train_data.drop(columns=[target_col]),
        train_data[target_col]
        )

    base_cross_score = cross_val_score(
        model, train_data.drop(columns=[target_col]), train_data[target_col], 
        cv=cv,scoring=scorer,n_jobs=n_jobs
        )
    
    cross_scores.append(base_cross_score)
    test_scores.append(base_test_score)
    remove_list = ['none']

    if remove_features is None:
        for col in feature_cols:
            test_model = clone(model)
            abla_train = train_data.drop(columns=[target_col,col])
            abla_test = test_data.drop(columns=[target_col,col])

            cross_score = cross_val_score(test_model, abla_train, train_data[target_col], cv=cv,scoring=scorer,
                                          n_jobs=n_jobs)

            test_model.fit(abla_train,train_data[target_col])
            test_score = scorer(test_model,abla_test,test_data[target_col])

            cross_scores.append(cross_score)
            test_scores.append(test_score)
            remove_list.append(col)
    else:
        for features in remove_features:
            test_model = clone(model)
            if type(features) ==list:
                remove_cols = features + [target_col]
            else:
                remove_cols = [target_col,features]
            abla_train = train_data.drop(columns=remove_cols)
            abla_test = test_data.drop(columns=remove_cols)

            cross_score = cross_val_score(test_model, abla_train, train_data[target_col], cv=cv,scoring=scorer,
                                          n_jobs=n_jobs)

            test_model.fit(abla_train,train_data[target_col])
            test_score = scorer(test_model,abla_test,test_data[target_col])

            cross_scores.append(cross_score)
            test_scores.append(test_score)
            remove_list.append(features)

    cross_scores = np.array(cross_scores)
    cross_mean = np.mean(cross_scores,axis=1)
    cross_std = np.std(cross_scores,axis=1)

    result_df = pd.DataFrame({
        'remove_features':remove_list,
        'score_mean':cross_mean,
        'score_std':cross_std,
        'score_test':test_scores
    })

    return cross_scores,result_df

def permutation_test(
    model,
    train_data,
    test_data,
    target_col,
    scorer = 'r2',
    n_repeats = 10,
    random_state = None,
    n_jobs = None
):  

    feature_cols = train_data.columns.drop(target_col)

    train_permutation = permutation_importance(
            model, 
            train_data[feature_cols],
            train_data[target_col],
            scoring=scorer,
            n_repeats=n_repeats, 
            random_state=random_state,n_jobs=n_jobs)
    
    test_permutation = permutation_importance(
            model, 
            test_data[feature_cols],
            test_data[target_col],
            scoring=scorer,
            n_repeats=n_repeats, 
            random_state=random_state,n_jobs=n_jobs)
    
    return train_permutation,test_permutation

def supervised_sensivity_test(
    model,
    train_data,
    target_col,
    param_grid:dict,
    scoring: Union[dict,str] = 'r2',
    cv = 10,
    n_jobs = None,
    verbose = 0
):  
    output_df = None
    if type(scoring) == dict:
        for score_name,scorer in scoring.items():
            grid_model = GridSearchCV(model,param_grid,scoring=scorer,cv=cv,verbose=verbose,n_jobs=n_jobs)
            grid_model.fit(train_data.drop(columns=[target_col]),train_data[target_col])
            if output_df is None:
                output_df = pd.DataFrame(grid_model.cv_results_['params'])
                output_df[score_name] = grid_model.cv_results_['mean_test_score']
                output_df[score_name + '_std'] = grid_model.cv_results_['std_test_score']
            else:
                output_df[score_name] = grid_model.cv_results_['mean_test_score']
                output_df[score_name + '_std'] = grid_model.cv_results_['std_test_score']
    else:
        grid_model = GridSearchCV(model,param_grid,scoring=scoring,cv=cv,verbose=verbose,n_jobs=n_jobs)

        grid_model.fit(train_data.drop(columns=[target_col]),train_data[target_col])

        output_df = pd.DataFrame(grid_model.cv_results_['params'])
        output_df[scoring] = grid_model.cv_results_['mean_test_score']
        output_df[scoring + '_std'] = grid_model.cv_results_['std_test_score']

    return output_df

def learning_curve_test(
        model,
        data: pd.DataFrame,
        target_col: str,
        sample_range,
        scorer = make_scorer(r2_score),
        cv=10,
        stratify = True,
        n_jobs = None,
        random_state = None
        ):
    
    if np.max(sample_range) > data.shape[0]:
        raise ValueError("Sampling range exceed the number of data records")
    
    if stratify:
        cv = StratifiedKFold(n_splits=cv, random_state=random_state)
        
    cross_scores = []
    for n in sample_range:
        X = data.drop(columns = [target_col])
        y = data[target_col]
        test_model = clone(model)
        if stratify:
            X_train, _, y_train, _ = train_test_split(
                X, y, train_size=n, random_state=random_state
            )
        else:
            X_train, _, y_train, _ = train_test_split(
                X, y, train_size=n,stratify=target_col, random_state=random_state
            )

        cross_score = cross_val_score(test_model, X_train, y_train, cv=cv,scoring=scorer,n_jobs=n_jobs)
        
        cross_scores.append(cross_score)
    
    cross_scores = np.array(cross_scores)
    cross_mean = np.mean(cross_scores,axis=1)
    cross_std = np.std(cross_scores,axis=1)

    result_df = pd.DataFrame({
        'n_sample':sample_range,
        'score_mean':cross_mean,
        'score_std':cross_std
    })

    return cross_scores,sample_range,result_df

def shap_analysis(
    fitted_model,
    explain_model,
    train_df,
    test_df,
    target_col = 'target_variable',
    on_testdata = True
    ):

    input_train_df = train_df.drop(columns=[target_col]).copy()
    input_test_df = test_df.drop(columns=[target_col]).copy()
    # columns = input_train_df.columns

    input_model = fitted_model
    if type(fitted_model)==Pipeline:
        preprocess = fitted_model[:-1]
        features = preprocess.get_feature_names_out()
        input_model = fitted_model[-1]
        input_train_df = pd.DataFrame(preprocess.transform(input_train_df),columns = features)
        input_test_df = pd.DataFrame(preprocess.transform(input_test_df),columns = features)

    masker = shap.maskers.Independent(input_train_df)
    
    if is_classifier(input_model):
        explainer = explain_model(input_model,masker,model_output="probability")
    else:
        explainer = explain_model(input_model,masker)

    if on_testdata:
        shap_values = explainer(input_test_df)
    else:
        shap_values = explainer(input_train_df)

    return shap_values,explainer

def get_worst_examples(
        model,
        test_data,
        target_col,
        get_n = 3,
        get_random_err = True,
        problem_type = 'classification',
        random_state = None
):
    y_test = test_data[target_col]
    if problem_type == 'classification':  # handle only binary classification for now
        y_pred = model.predict(test_data.drop(columns=[target_col]))
        y_pred_proba = model.predict_proba(test_data.drop(columns=[target_col]))
        if is_classifier(model) and hasattr(model,'decision_function'):
            y_decision = model.decision_function(test_data.drop(columns=[target_col]))
            df = pd.DataFrame({
                'true_label': y_test,
                'predicted_label': y_pred,
                'probability_0': y_pred_proba[:,0],
                'probability_1':y_pred_proba[:,1],
                'decision_function':y_decision
            })
        else:
            df = pd.DataFrame({
                'true_label': y_test,
                'predicted_label': y_pred,
                'probability_0': y_pred_proba[:,0],
                'probability_1':y_pred_proba[:,1],
            })
        
        if get_random_err:
            pre_fail_df = df[((df['true_label'] == 0) & (df['predicted_label'] == 1))|
                         (df['true_label'] == 1) & (df['predicted_label'] == 0)]
            pred_fail_idexes = pre_fail_df.sample(get_n,random_state=random_state).index
            return test_data.loc[pred_fail_idexes],df.loc[pred_fail_idexes]
        else:
            # predict 1, but true label 0
            false_positives = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)]
            FP_top_idx = false_positives.sort_values(by='probability_1', ascending=False).head(get_n).index.tolist()
            FP_bot_idx = false_positives.sort_values(by='probability_1', ascending=False).tail(get_n).index.tolist()
            FP_indexes = FP_top_idx + FP_bot_idx
            # predicted 0, but true label 1
            false_negatives = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)]
            FN_top_idx = false_negatives.sort_values(by='probability_0', ascending=True).head(get_n).index.tolist()
            FN_bot_idx = false_negatives.sort_values(by='probability_0', ascending=True).tail(get_n).index.tolist()
            FN_indexes = FN_top_idx + FN_bot_idx

            data_df = pd.concat((test_data.loc[FP_indexes],test_data.loc[FN_indexes]))
            info_df = pd.concat((df.loc[FP_indexes],df.loc[FN_indexes]))
            return data_df,info_df
            
    else: # problem type is regression
        y_pred = model.predict(test_data.drop(columns=[target_col]))
        diff = y_test - y_pred

        df = pd.DataFrame({
            'true_value': y_test,
            'predicted_value': y_pred,
            'residual': diff,
            'absolute_residual': np.abs(diff)
        })

        df = df.sort_values('absolute_residual',ascending=False)
        if get_random_err:
            pred_fail_idexes = df.head(get_n*10).sample(get_n,random_state=random_state).index
        else:
            pred_fail_idexes = df.head(get_n).index

        return test_data.loc[pred_fail_idexes],df.loc[pred_fail_idexes]