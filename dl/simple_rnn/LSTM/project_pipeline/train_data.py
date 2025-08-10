import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import os,sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import keras
from keras.models import Sequential
from keras.layers import (
    LSTM, 
    Dropout, 
    Dense, 
    Activation
)
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


from project_pipeline.handle_data import user_data
from project_pipeline.custom_exception import custom_exception


class user_model:
    def __init__(self):
        try:
            self.class_start_time=datetime.now().strftime("%d_%m_%y__%Hh_%Mm_%Ss")
        except Exception as e:
            raise custom_exception(e,sys)



    def construct_model(
            self,
            n, 
            input_shape
    ):
        try:
            model = Sequential()
            
            model.add(
                LSTM(
                    units=n,
                    return_sequences=True,
                    input_shape=input_shape
                )
            )
            model.add(Dropout(0.2))
            model.add(Activation("relu"))
            
            model.add(
                LSTM(
                    units=n,
                    return_sequences=True
                )
            )
            model.add(Dropout(0.2))
            model.add(Activation("relu"))
            
            model.add(
                LSTM(
                    units=n,
                    return_sequences=True
                )
            )
            model.add(Dropout(0.2))
            model.add(Activation("relu"))
            
            model.add(
                LSTM(
                    units=n,
                    return_sequences=False
                )
            )
            model.add(Dropout(0.2))
            model.add(Activation("relu"))
            
            model.add(
                Dense(
                    units=1
                )
            )
            model.add(Activation("linear"))
            
            return model

        except Exception as e:
            raise custom_exception(e,sys)



    def compile_model(
            self,
            model,
            loss,
            optimizer,
            metrics
    ):
        try:
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=metrics
            )

            return model

        except Exception as e:
            raise custom_exception(e,sys)



    def get_callbacks(
            self
    ):
        try:
            # file_name=datetime.now().strftime("%d_%m_%y__%Hh_%Mm_%Ss")
            file_name=self.class_start_time
            main_dir=os.path.dirname(os.getcwd())
            file_path=os.path.join(main_dir,"data")
            os.makedirs(file_path,exist_ok=True)
            log_dir=os.path.join(file_path,file_name)

            tensorboard=TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )

            '''
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=0,
                verbose=0,
                mode='auto',
                baseline=None,
                restore_best_weights=False,
                start_from_epoch=0
            )
            '''

            early_stopping=EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="auto",
                restore_best_weights=True,
                start_from_epoch=20
            )

            return [tensorboard,early_stopping]

        except Exception as e:
            raise custom_exception(e,sys)




    def model_data(
            self,
            model
    ):
        try:
            print(f"model summary : \n {model.summary()}")
            keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

        except Exception as e:
            raise custom_exception(e,sys)



    def train_model(
            self,
            model,
            x_train,
            y_train,
            batch_size,
            epochs,
            x_val,
            y_val,
            callbacks
    ):
        try:
            history=model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(
                    x_val,
                    y_val
                ),
                callbacks=callbacks
            )

            return model,history

        except Exception as e:
            raise custom_exception(e,sys)




    def training_history(
            self,
            history
    ):
        try:
            print(f"Training history keys:\n {history.history.keys()}\n")
            for key, values in history.history.items():
                print(f"{key}: {values[-5:]}\n")  # Print last 5 values of each metric
            # print(history.head())

        except Exception as e:
            raise custom_exception(e,sys)




    def evaluate_model(
            self,
            model,
            x_test,
            y_test
    ):
        try:
            results = model.evaluate(x_test, y_test)
            # print(f"Model evaluation results:\n{results}\n")

            # Get metric names to interpret the result properly
            metric_names = model.metrics_names
            # print(f"Metric names:\n{metric_names}\n")

            # Nicely pair them up
            for name, value in zip(metric_names, results):
                print(f"{name}: {value}")

        except Exception as e:
            raise custom_exception(e,sys)





    def run_model(
            self
    ):
        try:
            user_data_object=user_data()
            x_train,y_train,x_val,y_val,x_test,y_test=user_data_object.transform_data()
            
            n_units = 50
            input_shape = (x_train.shape[1], 1)
            model = self.construct_model(n_units, input_shape)
            # optimizer='adam', loss='mse', metrics=['mae']
            model = self.compile_model(model, loss="mse", optimizer="adam", metrics=["mae","mse"])
            self.model_data(model)

            # Assume x_train, y_train, x_val, y_val are already defined and preprocessed
            callbacks = self.get_callbacks()
            model, history = self.train_model(model, x_train, y_train, batch_size=32, epochs=300, x_val=x_val, y_val=y_val, callbacks=callbacks)

            self.training_history(history)

            # Evaluate on test set
            self.evaluate_model(model, x_test, y_test)

        except Exception as e:
            raise custom_exception(e,sys)




if __name__=="__main__":
    user_model_object=user_model()
    user_model_object.run_model()