
import pandas as pd

def menu():
    print("Welcome to Group 63's Feature Selection Project")
    featureNum = input("Please enter total number of features: ")
    print("Type the number of the algorithm you want to run")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    print("3. Group 63's Special Algorithm ")
    featureNum = input("Enter number: ")

def smallDfLoader(filename):
    labels = []
    labels.insert(0, "Feature")
    for i in range(1, 11):
        labels.append(i)
    df = pd.read_csv("small-test-dataset.txt", sep='\s+', names=labels)
    #print(df.iloc[:,[3,5,7]])
    return df


def largeDfLoader():
    col_names = []
    col_names.insert(0, "Label")
    for i in range(1, 41):
        col_names.append(i)
    df = pd.read_csv("large-test-dataset.txt", sep='\s+', names=col_names)
    return df

def normalize_vals(df):
    normalized_df = df.copy()
    normalized_df[df.columns[1:]] = normalized_df[df.columns[1:]].apply(lambda x: (x - x.mean()) / x.std())
    return normalized_df


