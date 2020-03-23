import luigi
import os
import pandas as pd

checkpoint_path = os.path.join('..', 'data', 'checkpoints')
raw_data_path = os.path.join('..', 'data', 'raw')
cleaned_data_path = os.path.join('..', 'data', 'cleaned_data')
output_files_path = os.path.join('..', 'outputs', 'csv')
output_images_path = os.path.join('..', 'outputs', 'img')

class data_munger(luigi.Task):
    def create_directories(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_all_directories(self, directory_list):
        for directory in directory_list:
            self.create_directories(directory)

    def read_raw_data(self):
        calendar = pd.read_csv(os.path.join(raw_data_path, 'calendar.csv'))
        sales_train = pd.read_csv(os.path.join(raw_data_path, 'sales_train_validation.csv'))
        sell_prices = pd.read_csv(os.path.join(raw_data_path, 'sell_prices.csv'))
        sample_submission = pd.read_csv(os.path.join(raw_data_path, 'sample_submission.csv'))
        return sample_submission, calendar, sales_train, sell_prices

    def generate_regression_data(self, data, exogenous_data, sell_prices):
        id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        regression_data = data.melt(id_vars = id_columns, var_name = "d", value_name = "demand")
        regression_data = pd.merge(regression_data, exogenous_data, how = 'left', on = 'd')
        regression_data = pd.merge(regression_data, sell_prices, how = 'left', on = ['store_id', 'item_id', 'wm_yr_wk'])
        regression_data['d'] = regression_data['d'].str.replace('d_', '').astype(int)
        return regression_data

    def save_preprocessed_data(self, regression_data):
        regression_data.to_csv(os.path.join(cleaned_data_path, 'regression', 'regression.csv'), index = False)
        max_day = regression_data['d'].max()
        min_day = regression_data['d'].min()
        delta = max_day - min_day
        regression_data[regression_data['d'] >= ((delta * 0.9) + min_day)].to_csv(os.path.join(cleaned_data_path, 'regression', 'regression_sample_01.csv'), index=False)
        regression_data[regression_data['d'] >= ((delta * 0.99) + min_day)].to_csv(os.path.join(cleaned_data_path, 'regression', 'regression_sample_001.csv'), index=False)

    def preprocess_data(self, data, exogenous_data, sell_prices):
        regression_data = self.generate_regression_data(data, exogenous_data, sell_prices)
        self.save_preprocessed_data(regression_data)
        print('Pre-processing has been done.')

    def run(self):
        self.create_all_directories([checkpoint_path,
                                     raw_data_path,
                                     output_files_path,
                                     output_images_path,
                                     os.path.join(cleaned_data_path, 'time_series'),
                                     os.path.join(cleaned_data_path, 'regression'),
                                     os.path.join(checkpoint_path, 'data_munger')])

        sample_submission, calendar, sales_train, sell_prices = self.read_raw_data()

        self.preprocess_data(sales_train, calendar, sell_prices)

        pd.DataFrame().to_csv(os.path.join(checkpoint_path, 'data_munger', 'success.csv'))

    def requires(self):
        requirements_list = []
        return requirements_list

    def output(self):
        return luigi.LocalTarget(os.path.join(checkpoint_path, 'data_munger', 'success.csv'))


if __name__ == '__main__':
    luigi.build([data_munger()], workers = 4, local_scheduler = True, log_level = 'CRITICAL')