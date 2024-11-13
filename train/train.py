import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import warnings
# import the_module_that_warns

warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# First let us load the datasets into different Dataframes
def load_data(datapath):
    data = pd.read_csv(datapath)
   # Dimensions
    print('Shape:', data.shape)
    # Set of features we have are: date, store, and item
    return data
    
    
train_df = load_data('../data/train.csv')
test_df = load_data('../data/test.csv')


def split_data(train_data,test_data):
    train_data['date'] = pd.to_datetime(train_data['date'])
    test_data['date'] = pd.to_datetime(test_data['date'])

    train_data['month'] = train_data['date'].dt.month
    train_data['day'] = train_data['date'].dt.dayofweek
    train_data['year'] = train_data['date'].dt.year

    test_data['month'] = test_data['date'].dt.month
    test_data['day'] = test_data['date'].dt.dayofweek
    test_data['year'] = test_data['date'].dt.year

    col = [i for i in test_data.columns if i not in ['date','id']]
    y = 'sales'
    train_x, test_x, train_y, test_y = train_test_split(train_data[col],train_data[y], test_size=0.2, random_state=2018)
    return (train_x, test_x, train_y, test_y,col)

train_x, test_x, train_y, test_y,col = split_data(train_df,test_df)

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

def model(train_x, train_y, test_x, test_y, col):
    params = {
        'nthread': 10,
        'max_depth': 5,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mape',
        'num_leaves': 64,
        'learning_rate': 0.2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 3.097758978478437,
        'lambda_l2': 2.9482537987198496,
        'verbose': 1,
        'min_child_weight': 6.996211413900573,
        'min_split_gain': 0.037310344962162616,
	'verbose': -1  # Suppress warnings
    }

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(test_x, test_y)

    callbacks = [
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=callbacks
    )

    y_test = model.predict(test_x)
    return y_test, model


y_test, model = model(train_x,train_y,test_x,test_y,col)


model.save_model('../model/lightgbm_model.txt')
