#import src.test
import src.util
from src.NNclassifier import NNclassifier
from src.search import greedyforward, greedybackward

src.util.menu()

#filename = "small-test-dataset.txt"
#df = src.util.smallDfLoader(filename)
#print("with greedy forward:")
#selected_features, max_accuracies = greedyforward(df)
#testingWSearch = NNclassifier(features=selected_features)
#accuracy = testingWSearch.validate(df=df)
#print(accuracy)

#print("vs. with greedy backward:")
#selected_features, max_accuracies, removed_features = greedybackward(df)
#testingWSearch = NNclassifier(features=selected_features)
#print(f"{selected_features}")
#accuracy = testingWSearch.validate(df=df)
#print(accuracy)

#print("vs given without search:")
#testingNoSearch = NNclassifier(features=[3,5,7])
#accuracy = testingNoSearch.validate(df=df)
#print(accuracy)

