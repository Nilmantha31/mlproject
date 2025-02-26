import os
import sys
import dill
from src.exception import CustomException

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        test_report = {}
        train_report = {}

        for name, model in models.items():
            param_grid = params.get(name, {}) 
            if param_grid:
                gs = GridSearchCV(model,param_grid, cv=5)
                gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_true=y_train,y_pred=y_train_pred)
            test_model_score = r2_score(y_true=y_test, y_pred=y_test_pred)

            test_report[name]=test_model_score
            train_report[name]=train_model_score

        return test_report,train_report
    except Exception as e:
        raise CustomException(e,sys)