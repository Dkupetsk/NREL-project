#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('omnidirectional_water.out', sep='	', header=None)

wavel = data.iloc[7:,73].astype(float).to_numpy()
time = data.iloc[7:,0].astype(float).to_numpy()

plt.plot(time,wavel,'.')
plt.ylabel('η(t)')
plt.xlabel('t')
plt.show()
surge = data.iloc[7:,11].astype(float).to_numpy()
sway = data.iloc[7:,12].astype(float).to_numpy()
heave = data.iloc[7:,13].astype(float).to_numpy()
roll = data.iloc[7:,14].astype(float).to_numpy()
pitch = data.iloc[7:,15].astype(float).to_numpy()
yaw = data.iloc[7:,16].astype(float).to_numpy()

plt.plot(time,surge,'.')
plt.plot(time,sway,'.')
plt.plot(time,heave,'.')

plt.legend(['surge','sway','heave'])
plt.xlabel('t')
plt.show()

plt.plot(time,roll,'.')
plt.plot(time,pitch,'.')
plt.plot(time,yaw,'.')
plt.legend(['roll','pitch','yaw'])
plt.xlabel('t')
plt.show()
#%%
data=pd.read_csv('3D_water.out', sep='	', header=None)

wavel = data.iloc[7:,72].astype(float).to_numpy()
time = data.iloc[7:,0].astype(float).to_numpy()
#%%
plt.plot(time,wavel,'.')
plt.ylabel('η(t)')
plt.xlabel('t')
plt.show()
surge = data.iloc[7:,10].astype(float).to_numpy()
sway = data.iloc[7:,11].astype(float).to_numpy()
heave = data.iloc[7:,12].astype(float).to_numpy()
roll = data.iloc[7:,13].astype(float).to_numpy()
pitch = data.iloc[7:,14].astype(float).to_numpy()
yaw = data.iloc[7:,15].astype(float).to_numpy()

plt.plot(time,surge,'.')
plt.plot(time,sway,'.')
plt.plot(time,heave,'.')

plt.legend(['surge','sway','heave'])
plt.xlabel('t')
plt.ylabel('displacement in m')
plt.show()

plt.plot(time,roll,'.')
plt.plot(time,pitch,'.')
plt.plot(time,yaw,'.')
plt.legend(['roll','pitch','yaw'])
plt.ylabel('displacement in degrees')
plt.xlabel('t')
plt.show()
# %%
data=pd.read_csv('3D_water_100mdepth.out', sep='	', header=None)

wavel = data.iloc[7:,72].astype(float).to_numpy()
time = data.iloc[7:,0].astype(float).to_numpy()
#%%
plt.plot(time,wavel,'.')
plt.ylabel('η(t)')
plt.xlabel('t')
plt.show()
surge = data.iloc[7:,10].astype(float).to_numpy()
sway = data.iloc[7:,11].astype(float).to_numpy()
heave = data.iloc[7:,12].astype(float).to_numpy()
roll = data.iloc[7:,13].astype(float).to_numpy()
pitch = data.iloc[7:,14].astype(float).to_numpy()
yaw = data.iloc[7:,15].astype(float).to_numpy()

plt.plot(time,surge,'.')
plt.plot(time,sway,'.')
plt.plot(time,heave,'.')

plt.legend(['surge','sway','heave'])
plt.xlabel('t')
plt.ylabel('displacement in m')
plt.show()

plt.plot(time,roll,'.')
plt.plot(time,pitch,'.')
plt.plot(time,yaw,'.')
plt.legend(['roll','pitch','yaw'])
plt.ylabel('displacement in degrees')
plt.xlabel('t')
plt.show()
# %%
