import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.util import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }

            params = {
                'DecisionTree':{
                    'criterion':['squared_error', 'absolute_error', 'friedman_mse','poisson'],
                    'splitter':['best', 'random'],
                    'max_features': ['sqrt','log2']
                },
                'RandomForest':{
                    'criterion':['squared_error', 'absolute_error', 'friedman_mse','poisson'],
                    'max_features': ['sqrt','log2', None],
                    'n_estimators':[8,16,32,64,100,128,256]
                },
                'GradientBoosting':{
                    'loss':['squared_error','absolute_error','huber','quantile'],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt', 'log2'],
                    'n_estimators':[8,16,32,64,100,128,256]
                },
                'LinearRegression':{},
                'KNeighbors':{
                    'n_neighbors':[5,7,9,11],
                    'weights':['uniform','distance'],
                    'algorithm':['ball_tree','kd_tree','brute']
                },
                'XGBoost':{
                    'n_estimators':[8,16,32,64,100,128,256],
                    'learning_rate':[0.1,0.01,0.05,0.001]
                },
                'AdaBoost':{
                    'n_estimators':[8,16,32,64,100,128,256],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'loss':['linear','square','exponential']
                }
            }

            model_report,_ = evaluate_models(x_train,y_train,x_test,y_test,models,params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info('Found best model for both train and test data')

            save_object(self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted = best_model.predict(x_test)
            r2score = r2_score(y_test,predicted)

            return r2score


        except Exception as e:
            raise CustomException(e,sys)
