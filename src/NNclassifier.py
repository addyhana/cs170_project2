import numpy as np
import pandas as pd
import src.util

class NNclassifier:
    def __init__(self, features):
        self.train_x = None
        self.train_y = None
        self.features = features

    def euclidean_distance(self, test_point, train_points):
        distances = []
        for training_value in train_points:
            distances.append(np.sqrt(np.sum((test_point - training_value) ** 2)))
        return distances

    def train(self, training_data):
        self.train_x = training_data.iloc[:, self.features].values
        self.train_y = training_data.iloc[:, 0].values
    
    def test(self, test_point):
        distances = self.euclidean_distance(test_point, self.train_x)
        closest_index = np.argmin(distances)
        return self.train_y[closest_index]

    def validate(self, df):
        correct = 0
        
        for i in range(len(df)):
            train_df = df[df.index!= i]
            y_hat = df.iloc[i, :]

            self.train(train_df)
            y_pred = self.test(y_hat[self.features].values)

            if y_pred == y_hat.iloc[0]:
                correct += 1
        
        accuracy = correct / len(df)
        return accuracy