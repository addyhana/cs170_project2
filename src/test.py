
import pandas as pd
from src.NNclassifier import NNclassifier

#data is a dict
data = {
    "Instance ID": [0, 1, 2, 3, 4],
    "Class Label": [1, 2, 1, 1, 2],
    "Feature1": [0.01, 0.01, 0.02, 0.03, 0.05],
    "Feature3": [0.02, 0.01, 0.03, 0.02, 0.01],
    "Feature7": [0.02, 0.03, 0.02, 0.02, 0.05]
}

df = pd.DataFrame(data)




