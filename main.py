#import src.test
import src.util
from src.NNclassifier import NNclassifier
from src.search import greedyforward, greedybackward
from sklearn.datasets import make_classification
from src.util import normalize_vals
import pandas as pd


src.util.menu()

filename = "small-test-dataset.txt"
df = src.util.smallDfLoader()
normalized_df = normalize_vals(df)
#print("with greedy forward:")
#selected_features, max_accuracies = greedyforward(df)
#testingWSearch = NNclassifier(features=selected_features)
#accuracy = testingWSearch.validate(df=df)
#print(accuracy)

#print("vs. with greedy backward:")
#selected_features, max_accuracies, removed_features = greedybackward(normalized_df)
#testingWSearch = NNclassifier(features=selected_features)
#print(f"{selected_features}")
#accuracy = testingWSearch.validate(df=normalized_df)
#print(accuracy)

#print("vs given without search:")
#testingNoSearch = NNclassifier(features=[3,5,7])
#accuracy = testingNoSearch.validate(df=df)
#print(accuracy)



