import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class SalesForecastPipeline:
    def __init__(self, train_data_path, test_data_path, model_save_path="./model/lightgbm_model.txt"):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_save_path = model_save_path
        self.model = None

    def load_data(self, path):
        data = pd.read_csv(path)
        print(f'Loaded data from {path}. Shape: {data.shape}')
        return data

    def preprocess_data(self, train_data, test_data):
        train_data['date'] = pd.to_datetime(train_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])

        for df in [train_data, test_data]:
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.dayofweek
            df['year'] = df['date'].dt.year

        features = [col for col in test_data.columns if col not in ['date', 'id']]
        target = 'sales'
        train_x, test_x, train_y, test_y = train_test_split(train_data[features], train_data[target], test_size=0.2, random_state=2018)
        return train_x, test_x, train_y, test_y, features

    def train_model(self, train_x, train_y, test_x, test_y, features):
        params = {
            'nthread': 10,
            'max_depth': 5,
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
            'verbose': -1  # Suppress warnings
        }

        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(test_x, test_y)

        callbacks = [
            early_stopping(stopping_rounds=50),
            log_evaluation(period=50)
        ]

        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=callbacks
        )
        print("Model training complete.")

    def save_model(self):
        if self.model is None:
            print("No model found to save.")
            return
        self.model.save_model(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def run(self):
        train_data = self.load_data(self.train_data_path)
        test_data = self.load_data(self.test_data_path)
        train_x, test_x, train_y, test_y, features = self.preprocess_data(train_data, test_data)
        self.train_model(train_x, train_y, test_x, test_y, features)
        self.save_model()


# Usage example:
#pipeline = SalesForecastPipeline(train_data_path='../data/train.csv', test_data_path='../data/test.csv')
#pipeline.run()

