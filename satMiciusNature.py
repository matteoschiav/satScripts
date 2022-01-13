#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:07:34 2021

@author: schiavon

This examples calculates the orbital parameters for the passage of a real satellite
(in this case QSS-Micius), using the two-line elements. It then calculates the
elevation, the channel length and the atmospheric losses with respect to two ground
stations, placed in Paris and Delft. These parameters can be input to the
FixedSatelliteLossModel to implement the corresponding channel on netsquid.

"""

from netsquid_freespace import channel

from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt

#%% Initialize channel paramters
wavelength = 850e-9

#%% Initialize the satellite

# Micius (QSS)
tleMicius = ['1 41731U 16051A   16353.90152003  .00000364  00000-0  17957-4 0  9993',
             '2 41731  97.3699 267.4500 0013200 178.5836 246.0824 15.23915303 19063']

satMicius = channel.Satellite(tleMicius)

#%% Initialize the ground stations

# Xinglong
latXinglong = 40.39587
longXinglong = 117.57746
altXinglong = 890.

staXinglong = channel.GroundStation(latXinglong, longXinglong, altXinglong, 'Xinglong')

#%% Initialize the downlink channels

downSatXinglong = channel.SimpleDownlinkChannel(satMicius, staXinglong, wavelength)

#%% Initialize the time array

# passage of Micius of 19/12/2016

dt = startTime = datetime(2016, 12, 19, 16, 45, 0)
endTime = datetime(2016, 12, 19, 17, 0, 0)
timeStep = timedelta(seconds = 10.)

timeList = []
while dt < endTime:
    timeList.append(dt)
    dt += timeStep
    
#%% Calculate the orbital parameters for the two channels

lenSatXinglong, tSatXinglong, elSatXinglong = downSatXinglong.calculateChannelParameters(timeList)

#%% Plot data

times = np.array([ (timeList[i]-timeList[0]).seconds  for i in range(len(timeList)) ])

plt.figure(figsize=(18,6))

plt.subplot(131)
plt.plot(times/60/60,elSatXinglong,'b')
plt.ylim([0,90])
plt.ylabel('Elevation [degrees]')
plt.xlabel('Passage time [minutes]')
plt.legend(['Paris','Delft'])

plt.subplot(132)
plt.plot(times/60/60,lenSatXinglong/1000,'b')
plt.ylabel('Channel length [km]')
plt.xlabel('Passage time [minutes]')
plt.legend(['Paris','Delft'])

plt.subplot(133)
plt.plot(times/60,tSatXinglong,'b')
plt.ylabel('Tatm')
plt.xlabel('Passage time [minutes]')
plt.legend(['Paris','Delft'])

plt.suptitle('Micius satellite - startDate 15/05/2021 00:00')