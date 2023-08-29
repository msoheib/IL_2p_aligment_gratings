# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_import as di 
import epoch_times as et
import deltaF_cal as dfc

import session_plot as sp # Import the session plot module.

from scipy.interpolate import interp1d



# %%
#specify numpy file F.npy
F_file_path = r''
F_file_path = r'F.npy'

xml_filepath = r'TSeries-08212023-1221-989-1-003.xml'

#%%

path_pycontrol_txt = r'989-2023-08-21-132131.txt'

pyc_session = di.session_dataframe(path_pycontrol_txt)

F = np.load(F_file_path)

#import voltage data as npy array
# V_file_path = r'E:\June\June\998-4-1TSeries-06142023-2113-187\TSeries-06142023-2113-187_Cycle00001_VoltageRecording_001.csv'
# V = np.genfromtxt(V_file_path, delimiter=',')
# V = V[1:,1:]

# x_old = np.arange(V.shape[0]) 
# x_new = np.linspace(0, V.shape[0]-1, 2750)
# f_new = interp1d(x_old, V, axis=0)

# interpolated_array = f_new(x_new)


# #convert to dataframe
# interpolated_pyc_session = pd.DataFrame(interpolated_array)

# interpolated_pyc_session.plot()


# %%

f_df = pd.DataFrame(F)
#convert the rows to columns
f_df = f_df.transpose()
F = f_df.to_numpy()
F.shape

# %%

frame_details = et.get_frame_details(xml_filepath)
events_as_columns = et.py_control_df_conversion(path_pycontrol_txt)


# %%
print(frame_details)

# %%
sp.session_plot('pycontrol/989-2023-08-21-132015.txt')



# %%
events_as_columns['photodetector'].plot()





# %%
events_as_columns.to_csv('events_as_columns.csv')


# %%
#correct jitter of the photodetector
events_as_columns_corrected = et.correct_gaps_photodetector(events_as_columns)

# %%
timepoints = et.get_orientation_indices_v2(events_as_columns_corrected)

# %%
#get the keyframes
keyframes = et.get_closest_frame_s2p(timepoints, frame_details)


events_as_columns_corrected = et.add_orientation_columns(events_as_columns_corrected, keyframes)

events_as_columns_corrected['photodetector'].plot()



# %%
deltaF = dfc.delta_fify(F).transpose()




events_as_columns_corrected.to_csv('test.csv')






# %%
deltaF






# %%
deltaF[20].plot()










# %%
delta_F = et.add_orientation_columns(deltaF, keyframes)







# %%
keyframes









# %%
delta_F.to_csv('test.csv')