import luigi
import os
import pandas as pd
import feature_creation as fc
import common as cm
import matplotlib.pyplot as plt
import seaborn as sns

class exploratory_data_analysis(luigi.Task):
    sample_size = luigi.FloatParameter()
    version = luigi.IntParameter()
    df = None

    def target_analysis(self, df):
        plt.figure(figsize=(24, 10))
        sns.distplot(df['demand'], bins=100)
        plt.savefig(os.path.join(cm.output_images_path, 'target_analysis'))

    def calculate_weights(self, df):
        days_to_consider = 28
        last_day = max(df['d'])

        demand_recent = df[df['d'] > last_day - days_to_consider].groupby('id')['demand'].agg('sum').reset_index()
        sell_price_recent = df[df['d'] > last_day - days_to_consider].groupby('id')['sell_price'].mean().reset_index()
        demand_recent['weight'] = demand_recent['demand'] * sell_price_recent['sell_price']

        plt.figure(figsize=(24, 10))
        sns.distplot(demand_recent['weight'], bins=100)
        plt.savefig(os.path.join(cm.output_images_path, 'weights'))

        plt.figure(figsize=(24, 10))
        kwargs = {'cumulative': True}
        sns.distplot(demand_recent['weight'], bins=100, hist_kws=kwargs)
        plt.savefig(os.path.join(cm.output_images_path, 'weights_cumulative'))

        demand_recent.to_csv(os.path.join(cm.output_files_path, 'weights.csv'))

    def run(self):
        cm.create_all_directories([os.path.join(cm.checkpoint_path, 'eda')])

        self.df = cm.read_data(os.path.join(cm.cleaned_data_path, 'regression', 'features_extractor'
                                    +'_sample_rate_' + str(self.sample_size).replace('.', '')
                                    + '_version_' + str(self.version)
                                    +'.csv'))

        self.target_analysis(self.df)
        self.calculate_weights(self.df)

        # pd.DataFrame().to_csv(os.path.join(cm.checkpoint_path, 'eda',
        #                                               'success_sample_rate_' + str(self.sample_size).replace('.','') +
        #                                               '_version_' + str(self.version) + '.csv'))

    def requires(self):
        requirements_list = [fc.feature_creation(sample_size = self.sample_size, version = self.version)]
        return requirements_list

    def output(self):
        return luigi.LocalTarget(os.path.join(cm.checkpoint_path, 'eda',
                                              'success_sample_rate_' + str(self.sample_size).replace('.','') +
                                              '_version_' + str(self.version) + '.csv'))


if __name__ == '__main__':
    luigi.build([exploratory_data_analysis(sample_size=0.01, version=1)], workers = 4, local_scheduler = True, log_level = 'INFO')