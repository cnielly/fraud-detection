import os

#os.chdir('../')

import pandas as pd
import numpy as np

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score 
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *

from src.fraud.nodes.metrics import *
from src.fraud.nodes.preprocessing import *
from src.fraud.nodes.modeling import *


df = pd.read_csv('data/01_raw/creditcard.csv')
X, y = get_xy(df, 'Class')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
#LR.fit(X_train, y_train)

SVM = svm.LinearSVC()
#SVM.fit(X, y)

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#RF.fit(X, y)

LGBM = lgb.LGBMClassifier(
    boosting_type = 'goss',
    objective =  'cross_entropy',
    metric = 'cross_entropy',
    num_leaves = 20,
    learning_rate = 0.05,
    feature_fraction = 0.8,
    n_estimators = 10000,
    verbose = 0)

NCL = NeighbourhoodCleaningRule(n_jobs=-1)
NM = NearMiss(n_jobs=-1)
ENN = EditedNearestNeighbours(n_jobs=-1)
CC = ClusterCentroids(n_jobs=-1)
CNN = CondensedNearestNeighbour(n_jobs=-1)
AKNN = AllKNN(n_jobs=-1)
IHT = InstanceHardnessThreshold(n_jobs=-1)
OSS = OneSidedSelection(n_jobs=-1)
Random = RandomUnderSampler()
TL = TomekLinks(n_jobs=-1)

SM = SMOTE(n_jobs=-1)
BDSMOTE = BorderlineSMOTE(n_jobs=-1)
ADA = ADASYN(n_jobs=-1)
KMSM = KMeansSMOTE(n_jobs=-1)
RO = RandomOverSampler()
SVMSM = SVMSMOTE(n_jobs=-1)


SMOTE_ENN = SMOTEENN()
SMOTET = SMOTETomek()



models = [LGBM, RF]
undersamplings = [None, NCL, NM, OSS, IHT]
oversamplings = [None, SM, BDSMOTE, RO, SVMSM]


comparaison = double_sampling(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                sampling_instances1=oversamplings, 
                sampling_instances2=undersamplings, 
                model_instances=models,
                func=compute_metrics)

comparaison.to_csv('metrics_double_overfirst.csv')

comparaison = double_sampling(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                sampling_instances1=undersamplings, 
                sampling_instances2=oversamplings, 
                model_instances=models,
                func=compute_metrics)

comparaison.to_csv('metrics_double_underfirst.csv')