
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

class KickstarterModel:

    def __init__(self):

        self.model = None



    def preprocess_training_data(self, df):
        '''
        Implement the method preprocess_training_data that takes:
        * 'df' - a csv file

        and returns:
        - X and y object

        This method is essential cleaning and preparing X and y data frames for the model.
        '''
        df =  df[df.isnull() == False]
        df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: 1 if x == 'live' else 0)
        X = df[df.columns[4:5]]
        y = df[df.columns[-1]]
        return X, y

    def fit(self, X, y):
        '''
        Implement the method fit that takes:
        * 'X' - X training data set (= independent variable(s))
        * 'y' - y_training data set (= dependent variable)

        This method is just fitting the training data in the model and doesn't have any output.

        '''
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)

        self.model = knn

    def preprocess_unseen_data(self, df):

        df =  df[new_df.isnull() == False]
        X_test = new_df['goal']
        return X_test

    def predict(self, X_test):

        y_pred = self.model.predict(X)
        return y_pred
