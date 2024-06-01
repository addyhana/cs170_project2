import src.test
import src.util
from src.NNclassifier import NNclassifier


# src.util.menu()

filename = "small-test-dataset.txt"
df = src.util.smallDfLoader(filename)
#testing.loadData(filename)

testing = NNclassifier(features=[3,5,7])
accuracy = testing.validate(df=df)
print(accuracy)