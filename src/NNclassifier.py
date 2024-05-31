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

    #def test(self, feature_column, test_instance):
        #test_instance is a dict --> {class:[feature, value]}
        #compare test instance with every value in the column

    def euclideanDistance(self, test_set, train_set):
        test_array = np.array(test_set)
        train_array = np.array(train_set)

        return np.sqrt(np.sum((test_array - train_array) ** 2))
    
    def train(self, df, feature_set, leave_out_index):
        #euclidean_distances = {}
        class1 = []
        class2 = []
        
        #print("training")
        #normalized_df = src.util.normalize_vals(df)
        #print(normalized_df)

        #test_columns = normalized_df.loc[:,feature_set]
        #print(test_columns)

        test_list = []
        for values in df.iterrows():
            curr_index = values[0]
            if curr_index == leave_out_index:
                #create your test set 
                for i in range(1, len(feature_set)):
                    test_list.append(values[1][i])

        
        #print(test_list)

        for values in df.iterrows():
            curr_index = values[0]
            
            if curr_index != leave_out_index:
                #create your train set
                train_set = [] 
                for i in range(1, len(feature_set)):
                    #print(values[1][i])
                    train_set.append(values[1][i])
                #print(train_set)
                feature_val = df.at[curr_index, 'Feature']
                #euclidean_distances[feature_val] = self.euclideanDistance(test_list, train_set)
                #euc_dist[class] = euc dist val
                if feature_val == 1.0:
                    class1.append(self.euclideanDistance(test_list, train_set))
                elif feature_val == 2.0:
                    class2.append(self.euclideanDistance(test_list, train_set))

        
        min_dist_class1 = min(class1)
        min_dist_class2 = min(class2)

        if min_dist_class1 < min_dist_class2:
            return 1.0
        elif min_dist_class2 < min_dist_class1:
            return 2.0


    def test(self, feature_set, df):
        print("we do be testin tho")
        correct = 0
        data_count = 0

        #function train- takes in feature set, index of leave one out, returns class 1 or class 2

        normalized_df = src.util.normalize_vals(df)
        #print(normalized_df)
        test_columns = normalized_df.loc[:,feature_set]

        #result = self.train(df, ["Feature",1,2], 0)
        #print(result)

        for values in test_columns.iterrows():
            data_count += 1
            curr_index = values[0]
            #print(curr_index)
            predicted_label = self.train(df, feature_set, curr_index)
            true_label = test_columns.at[curr_index, 'Feature']

            if predicted_label == true_label:
                correct+= 1
            
        accuracy = correct / data_count
        print(accuracy)

        return accuracy



            





        



                



            
            

            
            
                
                
            


        

    



   






    









    



