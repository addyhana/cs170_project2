
import pandas as pd
from sklearn.datasets import make_classification
from search import greedyforward, greedybackward

def create_sample_data():
    X, y = make_classification(n_samples=100, n_features=13, n_informative=5, n_redundant=2, random_state=50)
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(13)])
    df['Revenue'] = y
    return df

# backward
df = create_sample_data()
selected_features, maxAccuracies, removed_features = greedybackward(df)

# Print the results
print("Final selected features and their accuracies:")
for i, (feature, accuracy) in enumerate(zip(removed_features, maxAccuracies)):
    print(f"Step {i+1}: Feature {feature} removed to produce accuracy {accuracy * 100:.2f}%")
print("\n")

# forward
df = create_sample_data()
selected_features, maxAccuracies = greedyforward(df)

# Print the results
print("Final selected features and their accuracies:")
for i, (feature, accuracy) in enumerate(zip(selected_features, maxAccuracies)):
    print(f"Step {i+1}: Feature {feature} added to produce accuracy {accuracy * 100:.2f}%")
print("\n")



