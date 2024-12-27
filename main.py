import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from Decision_tress import DesicionTrees
from Regression import *
from pandas_tasks import Preprosessing
from clustering import Clustering

# from preprocessing import df_copy


df = pd.read_csv('Final_test/Preprocessed_Accident_Information.csv')

"""
Decision-Trees
and
Logistic Regression
"""
target_decision_tree = "Accident_Severity"
features_decision_tree = [
    '1st_Road_Class', '2nd_Road_Class', 'Carriageway_Hazards', 'Day_of_Week',
    'Junction_Control', 'Light_Conditions', 'Road_Surface_Conditions',
    'Road_Type', 'Speed_limit', 'Weather_Conditions'
]

df_decision_tree = df.copy()
df_logistic = df.copy()

# DesicionTrees.decision_trees(df_decision_tree, target_decision_tree, features_decision_tree)
# Regression.logisticRegression(df_logistic, 'InScotland')


"""
K-means clustering
"""
df_clustering = df.copy()
features_clustering = ['Longitude', 'Latitude']
# scaler, X_scaled = Clustering.elbow_method(df_clustering, features_clustering)
# Clustering.k_means_clustering(df_clustering, features_clustering, 3, scaler, X_scaled)


"""
Linear and Non-linear (randomForest) Regression
"""
features_regression = ['Speed_limit',
             'Weather_Conditions_Fine no high winds', 'Weather_Conditions_Fine + high winds', 'Weather_Conditions_Fog or mist', 'Weather_Conditions_Other',
               'Weather_Conditions_Raining + high winds', 'Weather_Conditions_Raining no high winds', 'Weather_Conditions_Snowing + high winds', 
               'Weather_Conditions_Snowing no high winds']
target_regression = 'Accident_Severity'

df_regression = df.copy()

df_regression[target_regression] = df_regression[target_regression].map({'Slight': 0, 'Serious':1, 'Fatal': 2})

Regression.linearRegression(df_regression, features_regression, target_regression)

Regression.randomForestRegressor(df_regression, features_regression, target_regression)