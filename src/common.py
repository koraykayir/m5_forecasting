import pandas as pd
import numpy as np
import os

checkpoint_path = os.path.join('..', 'data', 'checkpoints')
raw_data_path = os.path.join('..', 'data', 'raw')
cleaned_data_path = os.path.join('..', 'data', 'cleaned_data')
output_files_path = os.path.join('..', 'outputs', 'csv')
output_images_path = os.path.join('..', 'outputs', 'img')

training_mask = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd', 'date', 'wm_yr_wk', 'weekday',
                 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA',
                 'snap_TX', 'snap_WI', 'sell_price', 'wday_sin', 'wday_cos', 'month_sin', 'month_cos']
label = 'demand'

def reduce_mem_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                          100 * (start_mem - end_mem) / start_mem))
    return df

def create_directories(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_all_directories(directory_list):
    for directory in directory_list:
        create_directories(directory)

def read_data(path):
    return reduce_mem_usage(pd.read_csv(path))

def save_data(df, path):
    df.to_csv(path, index = False)


