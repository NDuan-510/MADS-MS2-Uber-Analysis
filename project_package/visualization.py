"""
Visualization module to show insight from the result of modeling step.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve,roc_curve,RocCurveDisplay
from sklearn.metrics import confusion_matrix
class Model_Visualizer():
    def __init__(self,X_train=None, X_test=None, y_train=None, y_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def ROC_curve(self,model=None,binary_result = True,y_test=None,y_score=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if binary_result:
            if model is not None:
                disp = RocCurveDisplay.from_estimator(
                    model, self.X_test, self.y_test, ax=ax, 
                    plot_chance_level=True, despine=True
                    )
            else:
                disp = RocCurveDisplay.from_predictions(
                    y_test, y_score,ax=ax,
                    plot_chance_level=True, despine=True
                    )
        else:  # TODO: handle multiple label feature, if required
            pass
        return disp
    
    def precision_recall_curve(self,model=None,binary_result = True,y_test=None,y_score=None):
        if binary_result:
            fig, ax = plt.subplots(figsize=(8, 6))
            if model is not None:
                disp = PrecisionRecallDisplay.from_estimator(
                    model, self.X_test, self.y_test, ax=ax, 
                    plot_chance_level=True, despine=True
                    )
            else:
                disp = PrecisionRecallDisplay.from_predictions(
                    y_test, y_score,ax=ax,
                    plot_chance_level=True, despine=True
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
    