import luigi
import os
import pandas as pd
import common as cm
import feature_creation as fc
import matplotlib.pyplot as plt

class exploratory_data_analysis(luigi.Task):
    sample_size = luigi.FloatParameter()
    version = luigi.IntParameter()
    def create_directories(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_all_directories(self, directory_list):
        for directory in directory_list:
            self.create_directories(directory)

    def run(self):
        self.create_all_directories([os.path.join(cm.checkpoint_path, 'eda')])

        pd.DataFrame().to_csv(os.path.join(cm.checkpoint_path, 'eda', 'success.csv'))

    def requires(self):
        requirements_list = [fc.feature_creation(sample_size = self.sample_size, version = self.version)]
        return requirements_list

    def output(self):
        return luigi.LocalTarget(os.path.join(cm.checkpoint_path, 'eda', 'success.csv'))


if __name__ == '__main__':
    luigi.build([exploratory_data_analysis()], workers = 4, local_scheduler = True, log_level = 'CRITICAL')