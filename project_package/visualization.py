"""
Visualization module to show insight from the result of modeling step.
"""

from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils.fixes import parse_version
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve,roc_curve,RocCurveDisplay
from sklearn.metrics import confusion_matrix
class Model_Visualizer():
    def __init__(self,X_train=None, X_test=None, y_train=None, y_test=None,feature_names=None,target_name = None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.target_name = target_name

    def reconstruct_dataframe(self):
        if self.feature_names is None:
            raise ValueError("Require 'feature_names' to reconstruct dataframe.")
        if self.target_name is None:
            target_name = 'target_variable'
        else:
            target_name = self.target_name

        train_df = pd.DataFrame(self.X_train,columns = self.feature_names)
        train_df[target_name] = self.y_train

        test_df = pd.DataFrame(self.X_test,columns = self.feature_names)
        test_df[target_name] = self.y_test

        return train_df,test_df

    def ROC_curve(self,model=None,binary_result = True,y_test=None,y_score=None,ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if binary_result:
            if model is not None:
                disp = RocCurveDisplay.from_estimator(
                    model, self.X_test, self.y_test, ax=ax, 
                    plot_chance_level=True
                    )
            else:
                disp = RocCurveDisplay.from_predictions(
                    y_test, y_score,ax=ax,
                    plot_chance_level=True
                    )
        else:  # TODO: handle multiple label feature, if required
            pass
        return disp
    
    def precision_recall_curve(self,model=None,binary_result = True,y_test=None,y_score=None,ax = None):
        if binary_result:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if model is not None:
                disp = PrecisionRecallDisplay.from_estimator(
                    model, self.X_test, self.y_test, ax=ax, 
                    plot_chance_level=True
                    )
            else:
                disp = PrecisionRecallDisplay.from_predictions(
                    y_test, y_score,ax=ax,
                    plot_chance_level=True
                    )
        else:  # TODO: handle multiple label feature, if required
            pass
        return disp
    
    def confusion_matrix(self,model,label_names,normalize=True):
        if (len(label_names) != np.unique(self.y_train).size) or (len(label_names) != np.unique(self.y_test).size):
            raise ValueError("The number of input label names doesn't match the Y output.")
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        if normalize:
            normalize = 'all'
        else:
            normalize = None

        train_matrix = confusion_matrix(self.y_train,y_pred_train,normalize=normalize)

        test_matrix = confusion_matrix(self.y_test,y_pred_test,normalize=normalize)
        
        fig, axes = plt.subplots(1,2,figsize=(14,6))

        sns.heatmap(train_matrix, annot=True,ax=axes[0],xticklabels=label_names,yticklabels=label_names)
        axes[0].set_title("Train dataset", fontsize=18, pad=20)
        axes[0].set_xlabel("Predicted label", fontsize=14, labelpad=20)
        axes[0].set_ylabel("True label", fontsize=14, labelpad=20)
        sns.heatmap(test_matrix, annot=True,ax=axes[1],xticklabels=label_names,yticklabels=label_names)
        axes[1].set_title("Test dataset", fontsize=18, pad=20)
        axes[1].set_xlabel("Predicted label", fontsize=14, labelpad=20)
        axes[1].set_ylabel("True label", fontsize=14, labelpad=20)
        plt.show()

    def residual_plot(self,model,bins=30,alpha=0.6):

        res_train = self.y_train - model.predict(self.X_train)
        res_test = self.y_test - model.predict(self.X_test)

        fig, axes = plt.subplots(1,2, width_ratios=[3, 1.5])
        fig.subplots_adjust(wspace = 0.5)

        axes[0].scatter(model.predict(self.X_train), res_train, label='train', color='blue',alpha=alpha)
        axes[0].scatter(model.predict(self.X_test), res_test, label='test', color='green',alpha=alpha)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)

        # Add labels, title, and legend
        axes[0].set_xlabel('Predicted value')
        axes[0].set_ylabel('Residual error')
        axes[0].set_title('Residual error plot')
        axes[0].legend()

        axes[1].hist(res_train, bins=bins, alpha=alpha, 
                     label='train', color='blue', orientation='horizontal',
                     edgecolor='black'
                     )
        axes[1].hist(res_test, bins=bins, alpha=alpha, 
                     label='test', color='green', orientation='horizontal',
                     edgecolor='black'
                     )
        # Add labels, title, and legend
        axes[1].set_xlabel('Residual error')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Residual error histogram')
        axes[1].legend()

        plt.show()
    
    def prediction_validate_plot(self,model,alpha=0.6):

        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        y_train_min,y_train_max = np.min(self.y_train),np.max(self.y_train)
        y_test_min,y_test_max = np.min(self.y_test),np.max(self.y_test)

        fig, axes = plt.subplots(1,2,figsize=(14,6))
        axes[0].scatter(self.y_train, y_pred_train, color='blue',alpha=alpha)
        axes[0].plot([y_train_min, y_train_max], [y_train_min, y_train_max],
                      color='black', linestyle='--', linewidth=1.5
                      )
        # Add labels, title
        axes[0].set_xlabel('True value')
        axes[0].set_ylabel('Predicted value')
        axes[0].set_title('Train data')

        axes[1].scatter(self.y_test, y_pred_test, color='green',alpha=alpha)
        axes[1].plot([y_test_min, y_test_max], [y_test_min, y_test_max],
                      color='black', linestyle='--', linewidth=1.5
                      )
        # Add labels, title
        axes[1].set_xlabel('True value')
        axes[1].set_ylabel('Predicted value')
        axes[1].set_title('Test data')

        plt.show()
    
    def feature_importance_plot(
        self,
        score_array: Union[np.ndarray,list],
        height: float = 0.7,
        figsize = (8, 4),
        color: str ='skyblue',
        normalize: bool =False,
        top_n: int = None,
        ax = None
        ):

        if normalize:
            score_array = np.array(score_array)
            score_array = score_array/np.max(np.abs(score_array))
        
        feature_names = np.array(self.feature_names)

        if top_n is not None:
            indexes = np.argsort(-np.abs(score_array))[:top_n][::-1]
            feature_names = feature_names[indexes]
            score_array = score_array[indexes]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.barh(feature_names, score_array, color=color,height=height)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        ax.set_title('Feature Importance plot')

        return ax
    
    def horizon_bar_plot(
        self,
        y: Union[np.ndarray,list],
        width: Union[np.ndarray,list],
        height: float = 0.7,
        figsize = (8, 4),
        color: str ='blue',
        normalize: bool =False,
        sort = True,
        xlabel: str = '',
        ylabel: str = '',
        title: str = '',
        ax = None
        ):

        if normalize:
            width = np.array(width)
            width = width/np.max(np.abs(width))
        feature_names = []
        for item in y:
            if type(y)!=str:
                feature_names.append(str(item))
            else:
                feature_names.append(item)
        feature_names = np.array(feature_names)

        if sort:
            indexes = np.argsort(-np.abs(width))[::-1]
            feature_names = feature_names[indexes]
            width = width[indexes]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.barh(feature_names, width, color=color,height=height)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        return ax
    
    def permutation_importance_plot(
        self,
        permutation_result,
        figsize = (8, 4),
        xlabel = "Decrease in metric score",
        title = '',
        ax=None
        ):
        perm_sorted_idx = permutation_result.importances_mean.argsort()

        tick_labels_parameter_name = (
            "tick_labels"
            if parse_version(matplotlib.__version__) >= parse_version("3.9")
            else "labels"
        )

        features = np.array(self.feature_names)

        tick_labels_dict = {tick_labels_parameter_name: features[perm_sorted_idx]}

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(permutation_result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel(xlabel)
        return ax

    def box_plot(
        self,
        array,
        labels,
        figsize = (8, 4),
        xlabel = '',
        title = '',
        ax=None,
        vertical_val:float =None
        ):
        if type(labels)==list:
            labels = [str(item) for item in labels]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(array, vert=False, labels = labels)
        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel(xlabel)
        if vertical_val is not None:
            ax.axvline(x=vertical_val, color="k", linestyle="--")

        return ax