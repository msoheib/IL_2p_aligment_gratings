#%%
import numpy as np
import pandas as pd

import sys
sys.path.append('G:\\My Drive\\0-Main\\tuning_curve_pipeline\\grating_task')

import data_import as di

# definition of processing function with data_import.py from pycontrol

#sdf = di.session_dataframe('998-2023-06-14-151011.txt')

# %%
def df_to_column_df(session_df):
    # Drop rows 2-8
    session_df = session_df.drop(index=range(0, 8))

    # Convert float columns to int and NaN values handling
    for col in session_df.columns:
        if session_df[col].dtype == np.float64:
            # Fill NaN values with 0
            session_df[col] = session_df[col].fillna(0).astype(int)

    # the maximum time value to set the total number of rows in the new DataFrame
    max_time = session_df['time'].max()

    # new DataFrame with unique events as columns and rows being 0 to the last time in column 'time'
    unique_events = session_df['name'].unique()
    epoch_columns = pd.DataFrame(index=range(max_time + 1), columns=unique_events, data=0)

    # go thru each row in the original DataFrame
    for index, row in session_df.iterrows():
        event_name = row['name']
        start_time = row['time']
        duration = row['duration']

        # if the type is "state" then set the duration to the value of the corresponding duration cell
        if row['type'] == 'state':
            duration = session_df.loc[index, 'duration']
        #fill the row with 1s for the duration of the event
        epoch_columns.loc[start_time:start_time+duration-1, event_name] = 1
        

        # is the cell is of type "event" then set duration to 1 for a single time pulse
        if row['type'] == 'event':
            duration = 1

        # Mark the time points based on duration for each unique event
        epoch_columns.loc[start_time:start_time+duration-1, event_name] = 1

    return epoch_columns

    # The resulting DataFrame 'epoch_columns' has columns representing unique events
    # and rows representing timepoints with values filled based on the duration.
