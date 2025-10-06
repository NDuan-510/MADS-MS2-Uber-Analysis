"""
Module to import advanced evaluation test for model.
"""
import warnings
from typing import List,Tuple
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.metrics import r2_score,make_scorer

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
        cv=cv,scoring=scorer
        )
    
    cross_scores.append(base_cross_score)
    test_scores.append(base_test_score)
    remove_list = ['none']

    if remove_features is None:
        for col in feature_cols:
            test_model = clone(model)
            abla_train = train_data.drop(columns=[target_col,col])
            abla_test = test_data.drop(columns=[target_col,col])

            cross_score = cross_val_score(test_model, abla_train, train_data[target_col], cv=cv,scoring=scorer)

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

            cross_score = cross_val_score(test_model, abla_train, train_data[target_col], cv=cv,scoring=scorer)

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
    random_state = None
    
):  

    feature_cols = train_data.columns.drop(target_col)

    train_permutation = permutation_importance(
            model, 
            train_data[feature_cols],
            train_data[target_col],
            scoring=scorer,
            n_repeats=n_repeats, 
            random_state=random_state)
    
    test_permutation = permutation_importance(
            model, 
            test_data[feature_cols],
            test_data[target_col],
            scoring=scorer,
            n_repeats=n_repeats, 
            random_state=random_state)
    
    return train_permutation,test_permutation

def sensivity_test(
    model,
    train_data,
    target_col,
    param_grid:dict,
    scoring = 'r2',
    cv = 10,
    verbose = 0
):  
    output_df = None
    if type(scoring) == dict:
        for score_name,scorer in scoring.items():
            grid_model = GridSearchCV(model,param_grid,scoring=scorer,cv=cv,verbose=verbose)
            grid_model.fit(train_data.drop(columns=[target_col]),train_data[target_col])
            if output_df is None:
                output_df = pd.DataFrame(grid_model.cv_results_['params'])
                output_df[score_name] = grid_model.cv_results_['mean_test_score']
                output_df[score_name + '_std'] = grid_model.cv_results_['std_test_score']
            else:
                output_df[score_name] = grid_model.cv_results_['mean_test_score']
                output_df[score_name + '_std'] = grid_model.cv_results_['std_test_score']
    else:
        grid_model = GridSearchCV(model,param_grid,scoring=scoring,cv=cv,verbose=verbose)

        grid_model.fit(train_data.drop(columns=[target_col]),train_data[target_col])

        output_df = pd.DataFrame(grid_model.cv_results_['params'])
        output_df['score'] = grid_model.cv_results_['mean_test_score']
        output_df['score_std'] = grid_model.cv_results_['std_test_score']

    return output_df
