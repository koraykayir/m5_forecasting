import luigi
import os
import pandas as pd
import data_munger as dm
import numpy as np
import math
import common as cm

class feature_creation(luigi.Task):
    sample_size = luigi.FloatParameter()
    version = luigi.IntParameter()

    def read_data(self):
        if self.sample_size == 1:
            data_path = os.path.join(cm.cleaned_data_path, 'regression', 'regression.csv')
        else:
            data_path = os.path.join(cm.cleaned_data_path, 'regression', 'regression_sample_' + str(self.sample_size).replace('.', '') + '.csv')
        return cm.read_data(data_path)

    def create_features(self, df):
        df['wday_sin'] = np.sin(df['wday'] * 2 * math.pi / 7)
        df['wday_cos'] = np.cos(df['wday'] * 2 * math.pi / 7)
        df['month_sin'] = np.sin(df['month'] * 2 * math.pi / 12)
        df['month_cos'] = np.cos(df['month'] * 2 * math.pi / 12)
        return df

    def run(self):
        cm.create_all_directories([os.path.join(cm.checkpoint_path, 'feature_creation')])
        df = self.read_data()
        df = self.create_features(df)
        cm.save_data(df, os.path.join(cm.cleaned_data_path, 'regression', 'features_extractor_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))
        pd.DataFrame().to_csv(os.path.join(cm.checkpoint_path, 'feature_creation', 'success_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))

    def requires(self):
        requirements_list = [dm.data_munger()]
        return requirements_list

    def output(self):
        return luigi.LocalTarget(os.path.join(cm.checkpoint_path, 'feature_creation', 'success_sample_rate_' + str(self.sample_size).replace('.', '') + '_version_' + str(self.version) + '.csv'))

if __name__ == '__main__':
    luigi.build([feature_creation(sample_size = 0.01, version = 1)], workers = 4, local_scheduler = True, log_level = 'CRITICAL')