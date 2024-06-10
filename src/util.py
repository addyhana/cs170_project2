
import pandas as pd
from src.NNclassifier import NNclassifier
import src.search
from src.search import greedyforward, greedybackward, find_correlated_features

def menu():
    print("Welcome to Group 62's Feature Selection Algorithm")
    file_to_test = input("Type in the name of the file to test: ")
    print("Type the number of the algorithm you want to run")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Group 62's special algorithm")
    selected_algo = input("Enter number: ")

    if file_to_test == 'small-test-dataset.txt':
        df = smallDfLoader()

    elif file_to_test == 'CS170_Spring_2024_Small_data__62.txt':
        df = smallDfCustomLoader()

    elif file_to_test == 'large-test-dataset.txt':
        df = largeDfLoader()

    elif file_to_test == 'CS170_Spring_2024_Large_data__62.txt':
        df = largeDfCustomLoader()

    print()

    print(f"This dataset has {len(df.columns) - 1} features (not including the class attribute), with {len(df)} instances.\n")
    print("Please wait while I normalize the data...")

    normalized_df = normalize_vals(df)

   

    if selected_algo == '1':
        selected_features, max_accuracies = greedyforward(normalized_df)
        testingWSearch = NNclassifier(features=selected_features)
        
        

    elif selected_algo == '2':

        selected_features, max_accuracies, removed_features = greedybackward(normalized_df)
        testingWSearch = NNclassifier(features=selected_features)


    else:
        print("Running the special algorithm...")
        selected_features, trace = find_correlated_features(df, 'Feature', len(normalized_df.columns) - 1, threshold_value=.05)
        #testingWSearch = NNclassifier(features=selected_features)

        
        

def smallDfLoader():
    labels = []
    labels.insert(0, "Feature")
    for i in range(1, 11):
        labels.append(i)
    df = pd.read_csv("small-test-dataset.txt", sep='\s+', names=labels)
    #print(df.iloc[:,[3,5,7]])
    return df


def largeDfLoader():
    col_names = []
    col_names.insert(0, "Feature")
    for i in range(1, 41):
        col_names.append(i)
    df = pd.read_csv("large-test-dataset.txt", sep='\s+', names=col_names)
    return df

def smallDfCustomLoader():
    col_names = []
    col_names.insert(0, "Feature")
    for i in range(1, 11):
        col_names.append(i)
    df = pd.read_csv("CS170_Spring_2024_Small_data__62.txt", sep='\s+', names=col_names)
    return df

def largeDfCustomLoader():
    col_names = []
    col_names.insert(0, "Feature")
    for i in range(1, 41):
        col_names.append(i)
    df = pd.read_csv("CS170_Spring_2024_Large_data__62.txt", sep='\s+', names=col_names)
    return df

def normalize_vals(df):
    normalized_df = df.copy()
    normalized_df[df.columns[1:]] = normalized_df[df.columns[1:]].apply(lambda x: (x - x.mean()) / x.std())
    return normalized_df


