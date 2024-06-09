from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.NNclassifier import NNclassifier

def greedyforward(df, maxSelectedFeatures=100, k=7): 

    # Initialize storage arrays
    selected_features = []
    maxAccuracies = []

    i = 0
    while i < (df.shape[1] - 1):
        maxAccuracy = 0
        maxIndex = -1
        
        for j in range(1, df.shape[1]): # assuming FIRST FIRST column is class
            # Skipping features already selected
            if j in selected_features:
                continue
            # add the j-th column to the current tuple 
            current_features = selected_features + [j]

            # Create NNclassifier instance with column names
            testingNoSearch = NNclassifier(features=current_features)
            accuracy = testingNoSearch.validate(df=df)

            print(f"Using feature(s) {current_features} with accuracy {accuracy * 100:.2f}")
            
            # If curr feature is the best, prepare to add to feature-tuple
            if accuracy > maxAccuracy:
                maxAccuracy = accuracy
                maxIndex = j

        # Check if accuracy decreased
        if (maxAccuracies and maxAccuracy < maxAccuracies[-1]):
            print("\nWarning, Accuracy has decreased!\n")
            break
        
        # adding to feature-tuple 
        selected_features.append(maxIndex)
        maxAccuracies.append(maxAccuracy)
        print(f"\nFeature set {selected_features} was best, accuracy is {maxAccuracy * 100:.2f}%.\n")
        if maxAccuracy == 1:
            break
        i += 1

    return selected_features, maxAccuracies

def greedybackward(df, k=7):

    # Initialize storage arrays
    selected_features = list(range(1, df.shape[1] - 1))  # Start with all features selected
    maxAccuracies = []
    removed_features = []

    i = 0
    while len(selected_features) > 0:
        maxAccuracy = 0
        worstIndex = -1
        
        for j in selected_features:  
            # Create a list excluding the j-th feature
            remaining_features = [feat for feat in selected_features if feat != j]
            
            # Create NNclassifier instance with column names
            testingNoSearch = NNclassifier(features=remaining_features)
            accuracy = testingNoSearch.validate(df=df)

            print(f"Using feature(s) {remaining_features} with accuracy {accuracy * 100:.2f}")
            
            # If the current feature set has better accuracy, update maxAccuracy and worstIndex
            if accuracy > maxAccuracy:
                maxAccuracy = accuracy
                worstIndex = j

        # If removing the worst feature decreases accuracy, terminate the loop
        if maxAccuracies and maxAccuracy < maxAccuracies[-1]:
            print("\nWarning, Accuracy has decreased!\n")
            break
        
        # Remove the worst feature from the selected features list
        selected_features.remove(worstIndex)
        removed_features.append(worstIndex)
        maxAccuracies.append(maxAccuracy)
        print(f"\nFeature set {selected_features} was best, accuracy is {maxAccuracy * 100:.2f}%.\n")
        if maxAccuracy == 1:
            break
        i += 1

    return selected_features, maxAccuracies, removed_features   


def find_correlated_features(df, target_variable, num_features, threshold_value):
    #make the correlation matrix
    corr_matrix = df.corr()

    #compute the absolute correlation values with the target variable
    abs_corr_with_target = corr_matrix[target_variable].apply(lambda x: abs(x))
    abs_corr_with_target = abs_corr_with_target.drop(target_variable)

    #find features with correlation above the specified threshold
    selected_features = abs_corr_with_target[abs_corr_with_target > threshold_value].nlargest(num_features).index.tolist()

    trace = [f"Selected Feature {index + 1}: {feature}" for index, feature in enumerate(selected_features)]

    for index, feature in enumerate(selected_features):
        print(f"Selected Feature {index + 1}: {feature} with correlation {abs_corr_with_target[feature]:.2f}")

    return selected_features, trace






