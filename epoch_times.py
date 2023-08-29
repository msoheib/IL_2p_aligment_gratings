#%%
# the function of this code is to provide function to process the events and states from the raw data from pycontrol to the df format that can be used in the analysi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.express as px

import data_import as di
import data_processing as dp
#import session_plot as sp

import xml.etree.ElementTree as ET

import plotly.express as px


#%%
#file_path = 'pycontrol_data\\998-2023-06-21-081841.txt'
# Instantiate session object from data file.
#session = di.Session(file_path)

#sp.play_session('pycontrol_data\\998-2023-06-21-081841.txt')
#sp.session_plot('pycontrol_data\\998-2023-06-21-081841.txt')

###############################################################

#%%
#fetches the frame period from xml file
#returns a dict of [frame_period, session_duration, frames] given the xml file name

def get_frame_details(xml_filename):

    #load xml file
    xml = ET.parse(xml_filename)
    
    #code below to be removed becasue it gives the wrong value for frame_period
    #just keeping so I remember why I did not use this method
    #frame_periodElement = xml.find(".//*[@key=\"frame_period\"]")
    #frame_period = float(frame_periodElement.get('value'))

    root = xml.getroot()

    sequenceElement = root.find('Sequence')
    frameElements = []

    for frameElement in sequenceElement.iter('Frame'):
        frameElements.append(frameElement)

    frame_period = float(frameElements[1].get('relativeTime')) - float(frameElements[0].get('relativeTime'))

    duration = float(frameElements[-1].get('relativeTime'))+ frame_period - float(frameElements[0].get('relativeTime'))
    frames = len(frameElements)

    #the frame relative time is the time of each frame relative to the start of the session
    frame_relative_time = np.arange(0, duration, frame_period)

    keys = ["frame_period", "session_duration", "total_frames", "frame_relative_time"]

    values =  [frame_period, duration, frames, frame_relative_time]

    return dict(zip(keys, values))



# convert the pycontrol txt file to a dataframe with the events as columns and the time as rows

def py_control_df_conversion(file_path):

    df = di.session_dataframe(file_path)

    # Drop rows 2-8
    df = df.drop(index=range(0, 8))

    # Convert float columns to int (and handle NaN values)
    for col in df.columns:
        if df[col].dtype == np.float64:
            # Fill NaN values with 0 (or any other default integer value)
            df[col] = df[col].fillna(0).astype(int)

    # Find the maximum time value to set the number of rows in the new DataFrame
    max_time = df['time'].max()

    # Create a new DataFrame with unique events as columns and rows being 0 to the last time in column 'time'
    unique_events = df['name'].unique()
    events_as_columns = pd.DataFrame(index=range(max_time + 1), columns=unique_events, data=0)

    # Process each row in the original DataFrame
    for index, row in df.iterrows():
        event_name = row['name']
        start_time = row['time']
        duration = row['duration']

        # If the "type" column value is "event", set duration to 1
        if row['type'] == 'event':
            duration = 1

        # Mark the time points based on duration for each unique event
        events_as_columns.loc[start_time:start_time+duration-1, event_name] = 1

    # The resulting DataFrame 'events_as_columns' has columns representing unique events
    # and rows representing timepoints with values filled based on the duration.
    return events_as_columns


# fillss gaps between the risign and falling edges
def correct_gaps_photodetector(dataframe):

    #copy of df
    df = dataframe.copy()
    # find contiguous blocks of 0s
    blocks = []
    for k, g in df.groupby(df['photodetector'].ne(df['photodetector'].shift()).cumsum()):
        if g.iloc[0]['photodetector'] == 0:
            blocks.append(g)

    # fill in 0s with 1s if the block size is less than 100
    for block in blocks:
        if len(block) < 100:
            df.loc[block.index, 'photodetector'] = 1

    return df


# remove any stray firings of the photodetector
def remove_stray_firings(df, threshold=10):

    data = df
    
    # Identify the rising and falling edges of the photodetector signal
    rising_edges = (data['photodetector'].diff() == 1)
    falling_edges = (data['photodetector'].diff() == -1)
    
    # Get the indices of the rising and falling edges
    rising_indices = data.index[rising_edges].tolist()
    falling_indices = data.index[falling_edges].tolist()
    
    # Check if the first detected edge is a falling edge, which would indicate the start of the data is in a 'firing' state
    if falling_indices[0] < rising_indices[0]:
        rising_indices.insert(0, 0)
    
    # Check if the last detected edge is a rising edge, which would indicate the end of the data is in a 'firing' state
    if rising_indices[-1] > falling_indices[-1]:
        falling_indices.append(len(data))
    
    # Iterate over the pairs of rising and falling edges and identify short firings
    for start, end in zip(rising_indices, falling_indices):
        if (end - start) < threshold:
            data['photodetector'][start:end] = 0

    # Save the corrected data back to the file
    return data



# fill in the over or under heads in the states from the phtodetector column

def expand_and_count(df):
    data = df
    filled_count = 0

    for col in data.columns:
        if col.startswith("degrees_"):
            rising = data.index[data[col].diff() == 1].tolist()
            falling = data.index[data[col].diff() == -1].tolist()
            
            for r in rising:
                w_start = max(0, r - 1500)
                if (data['photodetector'][w_start:r].diff() == -1).any():
                    idx = data['photodetector'][w_start:r].idxmax()
                    filled_count += r - idx
                    data[col][idx:r] = 1
                    
            for f in falling:
                w_end = min(len(data), f + 1500)
                if (data['photodetector'][f:w_end].diff() == 1).any():
                    idx = data['photodetector'][f:w_end].idxmin()
                    filled_count += idx - f
                    data[col][f:idx] = 1

    return filled_count



# get the start and end time of each block of 1s in the photodetector column
def get_orientation_blocks(dataframe):
    #MUST HAVE CONTIGIOUS COLUMN WITH NAME  "photodetector" AND "degrees_{ANGLE}" IN THE COLUMN NAME
    #1S AND 0S
    #get columns starting the with the name degree
    #because events were based on which occured first the column df is also arranged so
    degrees = dataframe.filter(regex='degrees').columns
    #find the contigious blocks of 1s in the photodetector column and get their start and end row index
    blocks = []
    #using cumulative sum to find the contigious blocks of 1s here
    for k, g in dataframe.groupby(dataframe['photodetector'].ne(dataframe['photodetector'].shift()).cumsum()):
        if g.iloc[0]['photodetector'] == 1:
            blocks.append([g.index[0], g.index[-1]])

    #start and end row index (1 row == 1 ms) of each block
    timepoints_in_ms = dict(zip(degrees, blocks))

    #getting the duration of each block then adfing it to the dictionary as the third element
    for _, j in timepoints_in_ms.items():
        j.append(j[1]-j[0])

    return timepoints_in_ms


# get timepoints fix
def get_orientation_indices_v2(df):
    data = df
    orientation_dict = {}
    for col in data.columns:
        if col.startswith("degrees_"):
            rising = data.index[data[col].diff() == 1].tolist()
            falling = data.index[data[col].diff() == -1].tolist()
            orientation_dict[col] = list(zip(rising, falling))
    return orientation_dict



# #%%
# def get_closest_frame_s2p(timepoints_in_ms, frame_details):
    
#     key_frames = {}

#     start_frames = []
#     end_frames = []
    
#     frame_relative_time = frame_details['frame_relative_time']
#     frame_period = frame_details['frame_period']


#     for angle, times in timepoints_in_ms.items():
#         start =  times[0][0]/1000
#         #end =  times[0]/1000
#         #end =  (times[0]+times[2])/1000
#         end =  (times[1]+times[2])/1000

#         print(f"{start}, {end}")
#         closest_start = min(frame_relative_time, angle=lambda x: abs(x - start))
#         closest_end = min(frame_relative_time, key=lambda x: abs(x - end))
#         start_frame = np.where(frame_relative_time == closest_start)[0][0]
#         start_frames.append(np.where(frame_relative_time == closest_start)[0][0])
#         end_frame = np.where(frame_relative_time == closest_end)[0][0]
#         end_frames.append(np.where(frame_relative_time == closest_end)[0][0])
        
#         key_frames[angle].append((start_frame, end_frame))

#     return key_frames




#%%
def get_closest_frame_s2p(timepoints_in_ms, frame_details):
    
    key_frames = {}

    start_frames = []
    end_frames = []
    
    frame_relative_time = frame_details['frame_relative_time']
    frame_period = frame_details['frame_period']

    for angle, _ in timepoints_in_ms.items():
        key_frames[angle] = []

    for angle, times in timepoints_in_ms.items():
        for time in times:
            print(time)
            start_index = time[0]
            start =  start_index/1000
            #end =  times[0]/1000
            #end =  (times[0]+times[2])/1000
            end_index = time[1]
            end =  end_index/1000

            print(f"{start}, {end}")
            closest_start = min(frame_relative_time, key=lambda x: abs(x - start))
            closest_end = min(frame_relative_time, key=lambda x: abs(x - end))
            
            start_frame = np.where(frame_relative_time == closest_start)[0][0]
            start_frames.append(np.where(frame_relative_time == closest_start)[0][0])
            
            end_frame = np.where(frame_relative_time == closest_end)[0][0]
            end_frames.append(np.where(frame_relative_time == closest_end)[0][0])
            
            key_frames[angle].append((start_frame, end_frame))

    return key_frames

#%%

#function that adds a a column for each orientation and fills in '1' for the duration of the orientation
#and zeros for the rest of the time
def add_orientation_columns(df, key_frames):
    deltaF_F = df.copy()
    for angle, frame_start_stops in key_frames.items():
        deltaF_F[f'{angle}'] = 0
        for frame_start_stop in frame_start_stops:
            start = frame_start_stop[0]
            end = frame_start_stop[1]+1
            deltaF_F.loc[start:end, f'{angle}'] = 1
            test = deltaF_F.loc[start:end, f'{angle}']
            print(test)
        
        #dropping the degree from the column name
        #deltaF_F.columns = deltaF_F.columns.str.replace('degrees_', '')
        # # adding the degree symbol to the column name
        # of.columns = [f'{i}°' for i in of.columns]
    return deltaF_F



#%%

# returns a dataframe with the mean response of each neuron in each block of 1s
#key_frames = get_closest_frame_s2p(timepoints_in_ms, frame_details)

def get_mean_in_block(deltaF_F, key_frames):
    neurons_means = {}
    for neuron in deltaF_F.columns:
        mean_all_angles = {}
        for angle, frame_start_stop in key_frames.items():
            start = frame_start_stop[0]
            end = frame_start_stop[1]+1
            sum_in_block = deltaF_F[f'{neuron}'][start:end].sum()
            mean_neuron_ori_response = deltaF_F[f'{neuron}'][start:end].mean()

            mean_all_angles[angle] = mean_neuron_ori_response
        neurons_means[neuron] = mean_all_angles

    df = pd.DataFrame(neurons_means).transpose()
    return df




#%%

# returns a heatmap of the mean response of each neuron to each orientation
def plot_heatmap_mean(df, title='Heatmap of Mean Responses', xlabel="Stimulus Orientation (degrees)", ylabel="Neuron", color="Mean (deltaF/F)"):


    fig = px.imshow(df, color_continuous_scale='viridis', 
                    title=title,
                    labels = dict(x=xlabel, y=ylabel, color=color),
                    aspect='auto')
    fig.update_layout(width=700, title_x=0.11,  title_font_size=20, title_font_family='Arial')
    fig.show()
    return fig

# %%
#return a polar plot of the mean response of ALL neurons to each orientation
def plot_polar_mean(df, title='Mean Response of All  Neurons to Each Orientation', 
                    color_discrete_sequence=px.colors.sequential.Agsunset):
    fig = px.line_polar(df, r=df.mean(axis=0), theta=df.columns, line_close=True, color_discrete_sequence=color_discrete_sequence)
    fig.update_layout(title=title, width=700, title_x=0.5, title_font_size=20, title_font_family='Arial')
    fig.update_traces(fill='toself')
    fig.show()
    return fig


#%%
#plot a scatter plot with the mean response of each neuron in the y axis and the orientation in the x axis
def plot_mean_response(dataframe, title='Mean Response of Each Neuron to Each Orientation',
                        color_discrete_sequence=px.colors.sequential.Agsunset):
    df = dataframe.copy()

    fig = px.line(df.iloc[1:], color_discrete_sequence=color_discrete_sequence, markers=True)
    fig.update_layout(title=title, width=300, title_x=0.5, title_font_size=20, title_font_family='Arial') 
    fig.show()
    return fig, df

# #%%

# #fig.write_html("heatmap.html")

# #%%
# DF_F = pd.read_csv('deltaF.csv')
# DF_F = DF_F.drop(columns=['Unnamed: 0'])
# df = py_control_df_conversion(file_path)

# df = correct_gaps_photodetector(df)

# timepoints_in_ms = get_orientation_blocks(df)

# #%%
# frame_details = get_frame_details('TSeries-06142023-2113-184.xml')
# key_frames = get_closest_frame_s2p(timepoints_in_ms, frame_details)

# mf = get_mean_in_block(deltaF_F, key_frames)

# # %%
# heatmap = plot_heatmap_mean(mf)

# plot_polar_mean = plot_polar_mean(mf)

# # %%
# df = add_orientation_columns(deltaF_F, key_frames)

# df.to_csv('example.csv')


# # %%

# degrees = ['degrees_0', 'degrees_45', 'degrees_90', 'degrees_135', 'degrees_180', 'degrees_225', 'degrees_270', 'degrees_315']

# tig, df = plot_mean_response(mf)
# tig.show()

# # %%
