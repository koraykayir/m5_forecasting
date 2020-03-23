import luigi
import os
import pandas as pd
import data_munger as dm

checkpoint_path = os.path.join('..', 'data', 'checkpoints')
raw_data_path = os.path.join('..', 'data', 'raw')
cleaned_data_path = os.path.join('..', 'data', 'cleaned_data')
output_files_path = os.path.join('..', 'outputs', 'csv')
output_images_path = os.path.join('..', 'outputs', 'img')

class feature_creation(luigi.Task):
    sample_size = luigi.FloatParameter()
    def create_directories(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_all_directories(self, directory_list):
        for directory in directory_list:
            self.create_directories(directory)

    def read_data(self):
        if self.sample_size == 0:
            data_path = os.path.join(cleaned_data_path, 'regression', 'regression.csv')
        else:
            data_path = os.path.join(cleaned_data_path, 'regression', 'regression_sample_' + str(self.sample_size).replace('.', '') + '.csv')
        return pd.read_csv(data_path)

    def create_features(self, df):
        return df

    def run(self):
        self.create_all_directories([os.path.join(checkpoint_path, 'eda')])
        df = self.read_data()
        df = self.create_features(df)
        pd.DataFrame().to_csv(os.path.join(checkpoint_path, 'eda', 'success.csv'))

    def requires(self):
        requirements_list = [dm.data_munger()]
        return requirements_list

    def output(self):
        return luigi.LocalTarget(os.path.join(checkpoint_path, 'eda', 'success.csv'))

if __name__ == '__main__':
    luigi.build([feature_creation(sample_size = 0.01)], workers = 4, local_scheduler = True, log_level = 'CRITICAL')