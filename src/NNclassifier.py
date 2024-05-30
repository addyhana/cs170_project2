import numpy as np
import pandas as pd
import src.util

class NNclassifier:
    def __init__(self):
        self.df = None
        self.data = None

    #def set_train(self, training_data):
        #setting the training data to our input 
        #self.training_data = training_data 

    #def train(self, training_data):
        #self.set_train(training_data)

    def test(self, feature_column, test_instance):
        #test_instance is a dict --> {class:[feature, value]}
        #compare test instance with every value in the column
        print('hi')
       


        
   #def test(self, test_instance):
        #print("output predicted class label")

    #def euclideanDistance(self, test_set, train_set):
        #return np.sqrt(np.sum((test_set - train_set) ** 2))
    
    #only use features 3, 5, 7


    #def test(self, df, feature):
        #row_to_drop = -1
        #norm_df = src.util.normalize_vals(df)
        #print("entered test")
        #features_df = norm_df.iloc[:,feature]
        #leave out 1st row..test, 2nd row..test, 

        #features_df_copy = features_df.copy()
        #print(features_df_copy)
        #print(features_df_copy)
        
        #dict --> {1: eudist1, eudist2, 2: eucdist 1, eucdist2}
            #create a dict for each feature you have like that 
            #find smallest value in each dict and then compare those values and find the smallest, map it to its parent and thats the class 

        #print(len(feature_list[1:]))

        #for row in features_df_copy.iterrows():
            #row_to_drop = row_to_drop + 1
            #temp_df = features_df_copy.drop(row_to_drop).loc[0:]
            #in temp_df, we have our data set minus the row we wanted to leave out


            #print(temp_df)
            #break

    






    









    



