from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def greedyforward(df, maxSelectedFeatures=100, k=7): 

    # Initialize storage arrays
    selected_features = []
    maxAccuracies = []

    i = 0
    while i < (df.shape[1] - 1):
        maxAccuracy = 0
        maxIndex = -1
        
        for j in range(df.shape[1] - 1):  # assuming last column is class
            # Skipping features already selected
            if j in selected_features:
                continue

            # add the j-th column to the current tuple 
            current_features = selected_features + [j]
            X = df.iloc[:, current_features]
            y = df.iloc[:, -1]
            
            # 20% of dataset is in test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create and train KNN classifier
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred = knn_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
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
    selected_features = list(range(df.shape[1] - 1))  # Start with all features selected
    maxAccuracies = []
    removed_features = []

    i = 0
    while len(selected_features) > 0:
        maxAccuracy = 0
        worstIndex = -1
        
        for j in selected_features:  
            # Create a list excluding the j-th feature
            remaining_features = [feat for feat in selected_features if feat != j]
            
            X = df.iloc[:, remaining_features]
            y = df.iloc[:, -1]
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create and train KNN classifier
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred = knn_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Using feature(s) {remaining_features} with accuracy {accuracy * 100:.2f}")
            
            # If the current feature set has better accuracy, update maxAccuracy and worstIndex
            if accuracy > maxAccuracy:
                maxAccuracy = accuracy
            else:
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



