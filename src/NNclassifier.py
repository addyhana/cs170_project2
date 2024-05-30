import numpy as np
import pandas as pd

class NNclassifier:
    def __init__(self):
        self.df = None
        self.data = None

    #def set_train(self, training_data):
        #setting the training data to our input 
        #self.training_data = training_data 

    #def train(self, training_data):
        #self.set_train(training_data)

        
   #def test(self, test_instance):
        #print("output predicted class label")

    def euclideanDistance(self, test_set, train_set):
        return np.sqrt(np.sum((test_set - train_set) ** 2))

    

    #for test find data point closest to it using euclidean dist and use its label as the output 


    