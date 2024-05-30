import numpy as np

class Validator:
    def __init__(self, feature_subset, classifier, df):
        self.feature_subset = None
        self.classifier = classifier 
        self.df = df