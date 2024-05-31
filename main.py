import src.test
import src.util
from src.NNclassifier import NNclassifier


src.util.menu()

filename = "small-test-dataset.txt"
df = src.util.smallDfLoader(filename)
#testing.loadData(filename)

testing = NNclassifier()

#include 0 as that is feature column
feat_list = [0, 3]


#new_df = src.util.normalize_vals(df)
testing.test(["Feature",3,5,7], df)


