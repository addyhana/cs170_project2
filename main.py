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


features_df = df.iloc[:,1]
new_df = src.util.normalize_vals(features_df)
testing.test(new_df)

