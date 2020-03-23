import luigi
import os
import pandas as pd
import feature_creation as fc
import numpy as np
import common as cm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

class modeling(luigi.Task):
    sample_size = luigi.FloatParameter()
    version = luigi.IntParameter()

    def evaluate_model(self, model, val, val_label, train, train_label):
        y_pred = model.predict(val, num_iteration=model.best_iteration)
        print('RMSE = ' + round(np.sqrt(mean_squared_error(val_label, y_pred)), 3))
        cm.evaluate_model(y_pred, val, val_label, train, train_label)

    def train_model(self, df, model = 'lgb'):
        train, train_label, val, val_label = self.train_val_split(df)
        if model == 'lgb':
            lgb_train = lgb.Dataset(train, train_label)
            lgb_eval = lgb.Dataset(val, val_label, reference = lgb_train)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'poisson',
                'metric': 'rmse',
                'num_leaves': 63,
                'alpha': 0.1,
                'lambda': 0.1,
                'learning_rate': 0.1,
                'feature_fraction': 0.77,
                'bagging_fraction': 0.66,
                'bagging_freq': 2,
                'verbose': 0
            }

            gbm = lgb.train(params,
                            lgb_train,
                            categorical_feature = cm.categorical_features,
                            early_stopping_rounds = 200,
                            num_boost_round = 2000,
                            valid_sets = lgb_eval)

            self.evaluate_model(gbm, val, val_label, train, train_label)

        return gbm

    def model_data(self, df):
        df = cm.encode_categorical(df, cm.categorical_features)
        self.train_model(df)


    def train_val_split(self, df):
        max_day = df['d'].max()
        min_day = df['d'].min()

        delta = max_day - min_day
        test_split_threshold = max_day - (delta * cm.test_percentage)
        train_set = df[df['d'] <= test_split_threshold]
        test_set = df[df['d'] > test_split_threshold]

        return train_set[cm.training_mask], train_set[cm.label], test_set[cm.training_mask], test_set[cm.label],

    def run(self):
        cm.create_all_directories([os.path.join(cm.checkpoint_path, 'modeling')])

        df = cm.read_data(os.path.join(cm.cleaned_data_path, 'regression', 'features_extractor_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))
        self.model_data(df)

        pd.DataFrame().to_csv(os.path.join(cm.checkpoint_path, 'modeling', 'success_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))

    def requires(self):
        requirements_list = [fc.feature_creation(sample_size = self.sample_size, version = self.version)]
        return requirements_list

    def output(self):
        return luigi.LocalTarget(os.path.join(cm.checkpoint_path, 'modeling', 'success_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))

if __name__ == '__main__':
    luigi.build([modeling(sample_size = 0.01, version = 1)], workers = 4, local_scheduler = True, log_level = 'CRITICAL')