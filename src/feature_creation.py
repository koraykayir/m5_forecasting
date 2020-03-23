import luigi
import os
import pandas as pd
import data_munger as dm
import numpy as np
import math

checkpoint_path = os.path.join('..', 'data', 'checkpoints')
raw_data_path = os.path.join('..', 'data', 'raw')
cleaned_data_path = os.path.join('..', 'data', 'cleaned_data')
output_files_path = os.path.join('..', 'outputs', 'csv')
output_images_path = os.path.join('..', 'outputs', 'img')

class feature_creation(luigi.Task):
    sample_size = luigi.FloatParameter()
    version = luigi.IntParameter()
    def create_directories(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_all_directories(self, directory_list):
        for directory in directory_list:
            self.create_directories(directory)

    def read_data(self):
        if self.sample_size == 1:
            data_path = os.path.join(cleaned_data_path, 'regression', 'regression.csv')
        else:
            data_path = os.path.join(cleaned_data_path, 'regression', 'regression_sample_' + str(self.sample_size).replace('.', '') + '.csv')
        return pd.read_csv(data_path)

    def create_features(self, df):
        df['wday_sin'] = np.sin(df['wday'] * 2 * math.pi / 7)
        df['wday_cos'] = np.cos(df['wday'] * 2 * math.pi / 7)
        df['month_sin'] = np.sin(df['month'] * 2 * math.pi / 12)
        df['month_cos'] = np.cos(df['month'] * 2 * math.pi / 12)
        return df

    def save_data(self, df):
        df.to_csv(os.path.join(cleaned_data_path, 'regression', 'features_extractor_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'), index = False)

    def run(self):
        self.create_all_directories([os.path.join(checkpoint_path, 'feature_creation')])
        df = self.read_data()
        df = self.create_features(df)
        self.save_data(df)
        pd.DataFrame().to_csv(os.path.join(checkpoint_path, 'feature_creation', 'success_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))

    def requires(self):
        requirements_list = [dm.data_munger()]
        return requirements_list

    def output(self):
        return luigi.LocalTarget(os.path.join(checkpoint_path, 'feature_creation', 'success_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))

if __name__ == '__main__':
    luigi.build([feature_creation(sample_size = 0.01, version = 1)], workers = 4, local_scheduler = True, log_level = 'CRITICAL')