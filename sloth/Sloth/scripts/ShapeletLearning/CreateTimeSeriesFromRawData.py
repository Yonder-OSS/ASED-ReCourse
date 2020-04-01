import pandas as pd
import numpy as np
import time

from datetime import datetime

datapath = 'data/output_tweets_small_garrett_09172018.csv'
series = pd.read_csv(datapath,dtype=str,header=0)
headers = list(series)
user_names = list(set(series['user_name'].values))

print("The headers are:")
print(headers)
# print("The unique user names are:")
# print(user_names)

series['2_hr_bin_start_time'] = series['created_at'].apply(lambda x: str(datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date())+' '+"{0:02d}".format(datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour))

binned_series = series.groupby(['user_name','2_hr_bin_start_time']).size()

# compare to be sure parsing works as expected
# print(series['2_hr_bin_start_time'])
# print(series['created_at'])
# print("DEBUG::binned series:")
# print(binned_series)
# print(binned_series['Amazing day'].to_frame())
# print(binned_series['Trouble'].to_frame())

# Method 1 - requires more knowledge of data (didn't bother generalizing, just use to spot check some columns)
# df = pd.DataFrame(data={'Amazing day':binned_series['Amazing day'].sort_index().to_frame().values.flatten(), 'Trouble':binned_series['Trouble'].sort_index().to_frame().values.flatten()[1:]}, \
# index = binned_series['Amazing day'].sort_index().to_frame().index)
# print(df.sort_values(by=['2_hr_bin_start_time']))

# Method 2 - inner join, i.e., select only data points in all series
user_name = user_names[0]
df_merged = binned_series[user_name].sort_index().to_frame()
for user_name in user_names[1:]:
    df_merged = pd.merge(df_merged,binned_series[user_name].to_frame(),left_index=True,right_index=True)
df_merged.columns = user_names

print("The processed binned posting frequency dataframe is:")
print(df_merged.sort_values(by=['2_hr_bin_start_time']))

df_merged.to_csv('ProcessedFrame.csv')