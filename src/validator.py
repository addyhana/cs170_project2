import numpy as np

class Validator:
    def __init__(self, feature_subset, classifier, df):
        self.feature_subset = None
        self.classifier = classifier 
        self.df = df

    def leave_one_out(self):
        score = 0
        for i in range(len(self.df)):
            training = self.df[self.df.index != i]
            training_instance = self.df.iloc[i]
            self.classifier.train(training)
            predict = self.classifier.test(test_instance[self.feature_subset].values)
            if predict == training_instance.iloc[0]:
                score = score + 1
        accuracy = score / len(self.df)
        return accuracy
        
