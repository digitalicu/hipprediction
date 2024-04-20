import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy

class DataManager(object):
    test_size = 0.25
    random_state = None
    use_procedure = False

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.tested_features = None

    def get_random_state(self):
        return self.random_state

    def switch_procedure(self):
        self.use_procedure = not self.use_procedure

    def set_tested_features(self, features):
        self.tested_features = features

    def set_feature(self, feature='m30'):
        self.selected_feature = feature

    def set_random_state(self, random_state):
        self.random_state = random_state

    def set_test_size(self, test_size):
        self.test_size = test_size

    def get_tested_features(self):
        if self.tested_features == None:
            exclude_column = ['m30', 'm90', 'm180', 'm365']
            if not self.use_procedure:
                exclude_column.append('procedure')
            self.tested_features = [i for i in list(self.data.columns) if i not in exclude_column]
        return self.tested_features
    
    def train_split(self):
        features = pd.get_dummies(self.get_data())
        labels = np.array(features[self.selected_feature])
        features = features.drop(self.selected_feature, axis = 1)
        return train_test_split(features, labels, test_size=self.test_size, random_state=self.random_state)
        
    def get_data(self, featured=True):
        full_data_columns = copy.deepcopy(self.get_tested_features())
        if featured:
            full_data_columns.append(self.selected_feature)
        return self.data[full_data_columns]
