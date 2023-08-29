#%%
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

#%%
#load numpy file from file
data = np.load('F.npy')

#%%
df =  pd.DataFrame(data).transpose()

#%%
df.to_csv('F.csv')

#%%
#read xml file from file

#xml_filename =  'TSeries-06142023-2113-184.xml'

#%%
#fetching the frame period from xml file
#returns a dict of [framePeriod, session_duration, frames] given the xml file name
def get_frame_details(xml_filename):

    #load xml file
    xml = ET.parse(xml_filename)
    
    #code below to be removed becasue it is giving the wrong value for framePeriod
    #just keeping so I remember why I did not use this method
    #framePeriodElement = xml.find(".//*[@key=\"framePeriod\"]")
    #framePeriod = float(framePeriodElement.get('value'))

    root = xml.getroot()

    sequenceElement = root.find('Sequence')
    frameElements = []

    for frameElement in sequenceElement.iter('Frame'):
        frameElements.append(frameElement)

    framePeriod = float(frameElements[1].get('relativeTime')) - float(frameElements[0].get('relativeTime'))

    duration = float(frameElements[-1].get('relativeTime'))+ framePeriod - float(frameElements[0].get('relativeTime'))
    frames = len(frameElements)

    keys = ["framePeriod", "session_duration", "total_frames"]

    values =  [framePeriod, duration, frames]

    return dict(zip(keys, values))

#get_frame_details(xml_filename)

#%%

#function that calculates the deltaF/F for each row by column given the suite2p F.npy file
# and returns a df of deltaF/F values and save the df as a csv file and npy file in the same directory
# the F0 is the mean of the all rows for a particular column
def delta_fify(npy):
    df = pd.DataFrame(npy).transpose()
    for column in df.columns:
        #F0 = df[column].mean() #F0 is the average Ft value measured from the trace over the whole recording (for eah columsn separately)
        F0 = df[column].quantile(.1) #F0 as 10th percentile of the whole trace
        df[column] = df[column] - F0
        df[column] = df[column]/F0
    np.save('deltaF_over_F.npy', df)
    df.to_csv('deltaF_over_F.csv')
    return df


#delta F calculation from Leena using 10% baseline

# import numpy as np

# from scipy.signal import convolve
# Fofiscell = F[iscell[:, 0] == 1, :]

# Fneuofiscell = Fneu[iscell[:, 0] == 1, :]

# Spksofiscell = spks[iscell[:, 0] == 1, :]

# correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell

# correctedFofiscell = np.transpose(correctedFofiscell)

# Spksofiscell = np.transpose(Spksofiscell)

# Fofiscell = np.transpose(Fofiscell)

# for i in range(Fofiscell.shape[1]):
#     filteredtrace = np.convolve(correctedFofiscell[:, i], np.ones(30) / 30, mode='same')  # running average of 30 points to filter trace
#     bl = np.percentile(filteredtrace, 10)  # calculating baseline as the 10th percentile
#     dFF[:, i] = (filteredtrace - bl) / bl:+1::skin-tone-3:1






