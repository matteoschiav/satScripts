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

from netsquid_freespace import channel, lossmodel

from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt

#%% Initialize channel paramters
wavelength = 810e-9

#%% Initialize the satellite

# TLE data
tleMicius = ['1 41731U 16051A   18258.22623758  .00000492  00000-0  22724-4 0  9994',
             '2 41731  97.3609 172.0392 0014964 104.4528  64.8735 15.24673830115845']


satMicius = channel.Satellite(tleMicius)

#%% Initialize the ground stations

#Delingha
latDelingha =  37+22/60+44.43/3600
longDelingha = 97+43/60+37.01/3600
altDelingha = 3153

staDelingha = channel.GroundStation(latDelingha,longDelingha, altDelingha, 'Delingha')

#Nanshan 
latNanshan = 43+28/60+31.66/3600
longNanshan = 87+10/60+36.07/3600
altNanshan = 2028

staNanshan = channel.GroundStation(latNanshan,longNanshan, altNanshan, 'Nanshan')

#%% Initialize the downlink channels

atmModelDelingha = channel.AtmosphereTransmittanceModel(wavelength, altDelingha, aerosolModel='RURAL_23KM')
atmModelNanshan = channel.AtmosphereTransmittanceModel(wavelength, altNanshan, aerosolModel='RURAL_23KM')

TzenDelingha = atmModelDelingha.calculateTransmittance(90)
TzenNanshan = atmModelNanshan.calculateTransmittance(90)

downSatDelingha = channel.SimpleDownlinkChannel(satMicius, staDelingha, wavelength, atmModelDelingha)

downSatNanshan = channel.SimpleDownlinkChannel(satMicius, staNanshan, wavelength, atmModelNanshan)

#%% Initialize the time array

dt = startTime = datetime(2018, 9, 14, 18, 12, 0)
endTime = datetime(2018, 9, 14, 18, 20, 0)
timeStep = timedelta(seconds = 10.)

timeList = []
while dt < endTime:
    timeList.append(dt)
    dt += timeStep
    
#%% Calculate the orbital parameters for the two channels

lenSatDelingha, tSatDelingha, elSatDelingha = downSatDelingha.calculateChannelParameters(timeList)

lenSatNanshan, tSatNanshan, elSatNanshan = downSatNanshan.calculateChannelParameters(timeList)

#%% Plot data

times = np.array([ (timeList[i]-timeList[0]).seconds  for i in range(len(timeList)) ])

plt.figure(figsize=(18,6))

plt.subplot(131)
plt.plot(times/60,elSatDelingha,'b')
plt.plot(times/60,elSatNanshan,'r')
plt.ylim([0,90])
plt.ylabel('Elevation [degrees]')
plt.xlabel('Passage time [minutes]')
plt.legend(['Delingha','Nanshan'])
plt.axhline(13)

plt.subplot(132)
plt.plot(times/60,lenSatDelingha/1000,'b')
plt.plot(times/60,lenSatNanshan/1000,'r')
plt.ylabel('Channel length [km]')
plt.xlabel('Passage time [minutes]')
plt.legend(['Delingha','Nanshan'])
plt.plot(times/60, (lenSatDelingha+lenSatNanshan)/1000,'k--')

plt.subplot(133)
plt.plot(times/60,tSatDelingha,'b')
plt.plot(times/60,tSatNanshan,'r')
plt.ylabel('Tatm')
plt.xlabel('Passage time [minutes]')
plt.legend(['Delingha','Nanshan'])

plt.suptitle('Micius satellite - passage of 14/09/2018 (18.00 UTC)')

# np.savez('MiciusDelinghaNanshan.npz',times=times,lenSatParis=lenSatParis, tSatParis=tSatParis, elSatParis=elSatParis,lenSatDelft=lenSatDelft, tSatDelft=tSatDelft, elSatDelft=elSatDelft )

#%% calculate the average transmittance

# loss model parameters
txDiv = 10e-6
sigmaPoint = 0.5e-6
rx_aperture = 0.6
Cn2 = 0

Nmeas = 100

# construct the satellite lossmodel
lmN = lossmodel.FixedSatelliteLossModel(txDiv, sigmaPoint, rx_aperture, Cn2, wavelength)
lmNnp = lossmodel.FixedSatelliteLossModel(txDiv, 0, rx_aperture, Cn2, wavelength)
lmD = lossmodel.FixedSatelliteLossModel(txDiv, sigmaPoint, rx_aperture, Cn2, wavelength)
lmDnp = lossmodel.FixedSatelliteLossModel(txDiv, 0, rx_aperture, Cn2, wavelength)

TN = np.zeros((len(times),))
TNnp = np.zeros((len(times),))
TD = np.zeros((len(times),))
TDnp = np.zeros((len(times),))

for i in range(len(times)):
    locTN = np.zeros((Nmeas,))
    locTNnp = np.zeros((Nmeas,))
    locTD = np.zeros((Nmeas,))
    locTDnp = np.zeros((Nmeas,))
    
    lmN.Tatm = lmNnp.Tatm = tSatNanshan[i]
    lmD.Tatm = lmDnp.Tatm = tSatDelingha[i]
    
    for j in range(Nmeas):
        locTN[j] = 1 - lmN._sample_loss_probability(lenSatNanshan[i]/1000)
        locTNnp[j] = 1 - lmNnp._sample_loss_probability(lenSatNanshan[i]/1000)
        locTD[j] = 1 - lmD._sample_loss_probability(lenSatDelingha[i]/1000)
        locTDnp[j] = 1 - lmDnp._sample_loss_probability(lenSatDelingha[i]/1000)
        
    TN[i] = np.mean(locTN)
    TNnp[i] = np.mean(locTNnp)
    TD[i] = np.mean(locTD)
    TDnp[i] = np.mean(locTDnp)
    
logTN = -10*np.log10(TN)
logTNnp = -10*np.log10(TNnp)
logTD = -10*np.log10(TD)
logTDnp = -10*np.log10(TDnp)

#%% plot the results

plt.figure()
plt.plot(times, logTN+logTD, 'b')
plt.plot(times, logTNnp+logTDnp, 'r')
plt.xlabel('Passage times [sec]')
plt.ylabel('Losses [dB]')
plt.title('Optical losses with pointing error')

minPos = np.argmin(logTN+logTD)
tmin = times[minPos]
plt.xlim([tmin-100,tmin+100])
plt.ylim([np.min(logTN+logTD)-5,np.min(logTN+logTD)+20])

plt.legend(['With pointing', 'Without pointing'])
plt.grid()


# plot atmospheric transmittance
plt.figure()
plt.plot(times,-10*np.log10(tSatDelingha*tSatNanshan))
plt.xlabel('Passage times [sec]')
plt.ylabel('Atmospheric Losses [dB]')

minPos = np.argmin(logTN+logTD)
tmin = times[minPos]
plt.xlim([tmin-100,tmin+100])
plt.ylim([np.min(-10*np.log10(tSatDelingha*tSatNanshan))-5,np.min(-10*np.log10(tSatDelingha*tSatNanshan))+10])
plt.grid()

#%% Janice comparison

LatmJanice = -10*np.log10(0.49**(1/np.cos(np.pi/2-np.deg2rad(elSatNanshan))+1/np.cos(np.pi/2-np.deg2rad(elSatDelingha))))

plt.figure()
plt.plot(times,LatmJanice)
plt.xlim([tmin-100,tmin+100])
plt.ylim([8,20])
plt.grid()

#%% beam wandering pointing loss

print('POINTING ERROR')
print('The model of Edoardo would give:',-10*np.log10(np.exp(-2*(sigmaPoint/txDiv)**2)),'dB')
print('The log-negative Weibull model gives:',np.mean(logTN+logTD-logTNnp-logTDnp),'dB')