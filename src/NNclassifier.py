import numpy as np

class NNclassifier:
    def __init__(self, df):
        self.df = df 
        self.training_data = None
        self.training_labels = None 

    def set_train(self, training_data):
        #setting the training data to our input 
        self.training_data = training_data 

    #def set_training_labels()

    def train(self, training_data):
        self.set_train(training_data)

        
    def test(self, test_instance):
        print("output predicted class label")

    #nearest neighbor - put all data points in a dict (class: label(this might have to be a tuple/list))
    #for test find data point closest to it using euclidean dist and use its label as the output 


    