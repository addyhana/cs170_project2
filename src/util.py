
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
    df = pd.read_csv("small-test-dataset.txt", delim_whitespace=True, names=labels)
    #print(df)
    return df





