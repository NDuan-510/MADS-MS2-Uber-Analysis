"""
Utility module that can be used to support other modules.
"""
import re
import glob

import pandas as pd
import dask.dataframe as dd

def require_batching(
    data_frame,
    max_mb = 100
    ):
    memory_size = data_frame.memory_usage().sum()/1024**2
    if memory_size >= max_mb:
        return True
    return False

def save_dataframe(
    data_frame,
    file_name,
    directory='datasets/raw',
    max_mb = 100,
    partition_size = '50MB'
    ):
    if require_batching(data_frame,max_mb=max_mb):
        dask_df = dd.from_pandas(data_frame,npartitions=1)
        dask_df = dask_df.repartition(partition_size=partition_size)
        dask_df.to_csv(directory + '/' + file_name + '*.csv',index=False)
    else:
        data_frame.to_csv(directory + '/' + file_name + '.csv',index=False)
    print('Save data completed.')
    
def read_data(
    file_name: str,
    directory: str ='datasets/raw',
    convert_pandas: bool = True
):
    path = directory + '/' + file_name + '*.csv'
    files = glob.glob(path)

    n = len(files)
    if n > 1:  # a dask directory with multiple csv
        dask_df = dd.read_csv(path)
        # convert to pandas df
        if convert_pandas:
            pd_df = dask_df.compute().reset_index(drop=True)
            return pd_df
        else:
            return dask_df
    elif n == 1:  # a csv file
        pd_df = pd.read_csv(files[0])
        return pd_df
    else:
        raise Exception('No file found.')