from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, RandomUnderSampler, TomekLinks
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import lightgbm as lgb

from src.fraud.nodes.metrics import *
import pandas as pd


def sampling(X_train, y_train, X_test, y_test, sampling_instances, model_instances, func):
    """
    Function to test different sampling methods on different models.
    For each sampling methods in list(sampling_instance) and each model in list(models) compute the set of metrics of metrics function

    :input X_train, y_train, X_test, y_test: np.array or pd.DataFrame of data
    :input sampling_instance: list of instances of sampling methods (tested for all methods of Imblearn)
    :input model_instances: list of instances of models (from SKLearn or with SKLearn API)
    :input func: function to compute metrics - either compute_metrics or compute_main_metrics

    :output metrics: pd.Dataframe with metrics as column and model_x_sampling
    """

    metrics = []
    # go through all sampling methods
    for sampling_instance in sampling_instances:
        if sampling_instance is not None:
            print('fitting sampling '+ str(sampling_instances.index(sampling_instance) + 1) + ' on ' +  
                str(len(sampling_instances)), " : ", type(sampling_instance).__name__)
            X_train1, y_train1 = sampling_instance.fit_resample(X=X_train, y=y_train)
        else:
            print('fitting sampling '+ str(sampling_instances.index(sampling_instance) + 1) + ' on ' +  
                str(len(sampling_instances)), " : ", type(sampling_instance).__name__)
            X_train1, y_train1 = X_train, y_train

        # Go through all models
        for model_instance in model_instances:
            print('fitting model ' + str(model_instances.index(model_instance) + 1) + ' on ' +  
                str(len(model_instances)), " : ", type(model_instance).__name__)
            model_instance.fit(X_train1, y_train1)
            metrics.append(func(y_test, model_instance.predict(X_test)))

    models = [type(model).__name__ for model in model_instances]
    methods = [type(sampling).__name__ for sampling in sampling_instances]
    index = [model + '_' + method for model in models for method in methods]

    #Dry run of compute metrics with return_index=True to get indexes
    columns = func(y_test, y_test, average='weighted', return_index=True)
    metrics = pd.DataFrame(metrics, columns=columns, index=index)

    return metrics


#e.g. :
"""
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
SVM = svm.LinearSVC()
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

NCL = NeighbourhoodCleaningRule()
NM = NearMiss()
ENN = EditedNearestNeighbours()

models_instance = [LR, RF]
undersamplings = [NCL, NM, ENN]

output = sampling(X_train, y_train, X_test, y_test undersamplings, models_instance, compute_metrics)


"""


def double_sampling(X_train, y_train, X_test, y_test, sampling_instances1, sampling_instances2, model_instances, func):
    """
    Function to test different combination of sampling methods (over and under) on different models.
    For each combination of sampling methods and each model, it computes the set of metrics of metrics function

    :input X_train, y_train, X_test, y_test: np.array or pd.DataFrame of data
    :input sampling_instance1: list of instances of sampling methods (tested for all methods of Imblearn) (1st round)
    :input sampling_instance2: list of instances of sampling methods (tested for all methods of Imblearn) (2nd round)
    :input model_instances: list of instances of models (from SKLearn or with SKLearn API)
    :input func: function to compute metrics - either compute_metrics or compute_main_metrics

    :output metrics: pd.Dataframe with metrics as column and model_x_sampling
    """

    list_of_df_metrics = []

    for sampling_instance1 in  sampling_instances1:
        if sampling_instance1 is not None:
            print('fitting sampling1 '+ str(sampling_instances1.index(sampling_instance) + 1) + ' over ' +  str(len(sampling_instances1)))

            X_train_1st_round, y_train1st_round = sampling_instance.fit_resample(X=X_train, y=y_train)
        else:
            X_train1st_round, y_train1st_round = X_train, y_train

        df_metrics = sampling(X_train1st_round, y_train1st_round, X_test, y_test, sampling_instances2, model_instances, func)

        list_of_df_metrics.append(df_metrics)

    df_metrics_all = pd.concat(list_of_df_metrics, keys=[type(x).__name__ for x in sampling_instances1], names=['First Sampling Round'])

    return df_metrics_all


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t)

    :input y_scores: pd.DataFrame or np.array of predicted probability
    :input t: probability thresold

    :output: np.array of predicted label
    """
    return [1 if y >= t else 0 for y in y_scores]


############### TO TEST ###############  

def tuning_sample_grid(X_train, y_train, X_test, y_test, model, nb_model, grid):
    
    best_score = 0
    
    #select a subset of models
    Param_grid = random.choices(grid, k = nb_model)
    
    #Loop to compute accuracy of each model
    for i in range(nb_model):
        grid_tested = Param_grid[i]
        print(grid_tested)
        model.set_params(**grid_tested)
        model.fit(X=X_train, y=y_train)
        y_preds = model.predict(X_test)
        score = average_precision_score(y_test, y_preds)
        if score > best_score:
            best_score = score
            best_grid = grid_tested
            
    print('Best average_precision_score is: ' + str(best_score))
    return best_grid


