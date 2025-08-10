import pandas as pd
import numpy as np
import os,sys


from project_pipeline.custom_exception import custom_exception


class user_data:
    def __init__(self):
        try :
            input_file_name="seattle-weather.csv"
            input_file_path=os.path.dirname(os.getcwd())
            # print(input_file_path)
            input_file_path=os.path.join(input_file_path,"LSTM","data")
            # print(input_file_path)
            input_file_path=os.path.join(input_file_path,input_file_name)
            self.input_file_path=input_file_path

            print(f"input file path : {input_file_path}")
            # print("E:/GitHubFIles/py_play_ground/dl/simple_rnn/LSTM/data/seattle-weather.csv")

        except Exception as e:
            raise custom_exception(e,sys)



    def input_data(self):
        try:
            df=pd.read_csv(self.input_file_path)
            training_set=df.iloc[:,2:3].values
            return training_set
        
        except Exception as e:
            raise custom_exception(e,sys)


    def df_to_XY(self,training_set,window_size=10):  
        try:
            X_train=[]
            y_train=[]

            for i in range(window_size,len(training_set)):
                X_train.append(training_set[i-window_size:i,0])
                y_train.append(training_set[i,0])
                
            X_train, y_train = np.array(X_train), np.array(y_train)
            return X_train, y_train
        
        except Exception as e:
            raise custom_exception(e,sys)





    def transform_data(self):
        try:    
            window_size = 10
            training_set=self.input_data()
            x,y = self.df_to_XY(training_set,window_size)
            # print(len(x),len(y))
            x_train = x[:800]
            y_train = y[:800]
            x_val = x[800:1000]
            y_val = y[800:1000]
            x_test = x[1000:]
            y_test = y[1000:]


            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)


            x_train=np.reshape(
                x_train,
                (x_train.shape[0],x_train.shape[1],1)
            )
            x_val=np.reshape(
                x_val,
                (x_val.shape[0],x_val.shape[1],1)
            )
            x_test=np.reshape(
                x_test,
                (x_test.shape[0],x_test.shape[1],1)
            )

            return (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test
            )
        
        except Exception as e:
            raise custom_exception(e,sys)





if __name__ == "__main__":
    try:
        data = user_data()
        x_train, y_train, x_val, y_val, x_test, y_test = data.transform_data()

        print("Training data shape:", x_train.shape)
        print("Validation data shape:", x_val.shape)
        print("Test data shape:", x_test.shape)

    except Exception as e:
        raise custom_exception(e,sys)




'''
try:
    pass
except Exception as e:
    raise custom_exception(e,sys)
'''