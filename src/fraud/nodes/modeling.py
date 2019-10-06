from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
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
        print('First Round:')
        if sampling_instance1 is not None:
            print('fitting sampling of 1st round '+ str(sampling_instances1.index(sampling_instance1) + 1) + ' over ' +  str(len(sampling_instances1)) + ': ' + type(sampling_instance1).__name__)
            X_train_1st_round, y_train1st_round = sampling_instance1.fit_resample(X=X_train, y=y_train)
            print('Second Round:')
        else:
            print('No 1st round Sampling methods applied')
            X_train_1st_round, y_train1st_round = X_train, y_train

        df_metrics = sampling(X_train_1st_round, y_train1st_round, X_test, y_test, sampling_instances2, model_instances, func)

        list_of_df_metrics.append(df_metrics)

    df_metrics_all = pd.concat(list_of_df_metrics, keys=[type(x).__name__ for x in sampling_instances1], names=['First Sampling Round'])

    return df_metrics_all



############### TO TEST ###############

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t)

    :input y_scores: pd.DataFrame or np.array of predicted probability
    :input t: probability threshold

    :output: np.array of predicted label
    """
    return [1 if y >= t else 0 for y in y_scores]

def model_thresold(X_train,y_train,X_test,t,model,**param):
    """
    This function predict class  based on  prediction threshold (t)

    :input X_train, y_train, X_test: np.array or pd.DataFrame of data
    :input t: probability threshold
    :model: string 'lightgbm' or 'randomforest'

    :output: np.array of predicted label
    """
    if model == 'randomforest':
        rf = RandomForestClassifier(param)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)
        y_pred = adjusted_classes(y_pred,t)
    else:
        lgb_class = lgb.LGBMClassifier(param)
        lgb_class.fit(X_train,y_train)
        y_pred = lgb_class.predict_proba(X_test)
        y_pred = adjusted_classes(y_pred,t)

    return y_pred


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


############### TO TEST ###############

# assymetric MSE. Performing well.
def custom_asymmetric_train(y_test,y_pred):
    residual = (y_test - y_pred).astype("float")
    grad = np.where(residual<0, -2*residual, -2*10.0*residual)
    hess = np.where(residual<0, 2, 2.0*10.0)
    return grad, hess

def custom_asymmetric_valid(y_test, y_pred):
    residual = (y_test - y_pred).astype("float")
    loss = np.where(residual < 0, (residual**2), (residual**2)*10.0) 
    return "custom_asymmetric_eval", np.mean(loss), False

# assymetric cross-entropy. Not performing well though
C_FN = 200
C_TP = 10
C_FP = 20
C_TN = 0

def cross_entropy_cost_sensitive(y_test,y_pred):
    if y_test == 1 :
        result = C_FN*np.log(y_pred)+C_TP*np.log(1-y_pred)
    else :
        result = C_FP*np.log(1-y_pred)+C_TN*np.log(y_pred)
    return result

def custom_cross_ent_cs_train(y_test,y_pred):
    grad = np.where(y_test == 1, C_FN/y_pred + C_TP/(1-y_pred), C_FP/(1-y_pred) + C_TN/y_pred)
    hess = np.where(y_test == 1, -C_FN/(y_pred**2) - C_TP/((1-y_pred)**2), - C_FP/((1-y_pred)**2) - C_TN/(y_pred**2))
    return grad,hess

def custom_cross_ent_cs_valid(y_test,y_pred):
    loss = cross_entropy_cost_sensitive(y_test,y_pred)
    return "custom_cross_ent_cs_valid", np.mean(loss), False


