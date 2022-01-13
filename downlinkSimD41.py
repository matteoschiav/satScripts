#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:58:40 2021

@author: schiavon
"""

from netsquid_freespace import lossmodel as ls
from netsquid_freespace import channel

import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler
import matplotlib as mpl

import lowtran

from timeit import default_timer as timer
from datetime import datetime, timedelta

#%% sim-1 diffraction losses - size of the beam on the ground

wl_1 = 795e-9
wl_2 = 1550e-9

txApRadius = 0.04
txDiv_1 = wl_1/(np.pi*txApRadius)
txDiv_2 = wl_2/(np.pi*txApRadius)


lgth = np.linspace(300,2500,100)

groundSpot_1 = lgth*1e3*txDiv_1
groundSpot_2 = lgth*1e3*txDiv_2

plt.figure()
plt.plot(lgth,groundSpot_1,label=r'$\lambda = 795 \, nm$')
plt.plot(lgth,groundSpot_2,label=r'$\lambda = 1550 \, nm$')
plt.xlabel('Satellite distance [km]')
plt.ylabel('Beam radius on the ground [m]')
plt.grid()

plt.legend()

#%% sim-2 atmospheric transmittance 

elevations = np.arange(20,90,1)

# urban - 5 km
atmMod1urban = channel.AtmosphereTransmittanceModel(wl_1, 0, aerosolModel='URBAN_5KM')
atmMod2urban = channel.AtmosphereTransmittanceModel(wl_2, 0, aerosolModel='URBAN_5KM')

Tatm_1_urban = np.array([atmMod1urban.calculateTransmittance(i) for i in elevations])
Tatm_2_urban = np.array([atmMod2urban.calculateTransmittance(i) for i in elevations])

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(elevations,-10*np.log10(Tatm_1_urban),label=r'$\lambda = 795 \, nm$')
plt.plot(elevations,-10*np.log10(Tatm_2_urban),label=r'$\lambda = 1550 \, nm$')
plt.xlabel('Satellite elevation [deg]')
plt.ylabel('Atmospheric absorption losses [dB]')
plt.title('Urban aerosol model - vis = 5 km')
plt.grid()

plt.legend()

# rural - 23 km
atmMod1rural = channel.AtmosphereTransmittanceModel(wl_1, 0, aerosolModel='RURAL_23KM')
atmMod2rural = channel.AtmosphereTransmittanceModel(wl_2, 0, aerosolModel='RURAL_23KM')

Tatm_1_rural = np.array([atmMod1rural.calculateTransmittance(i) for i in elevations])
Tatm_2_rural = np.array([atmMod2rural.calculateTransmittance(i) for i in elevations])

plt.subplot(122)
plt.plot(elevations,-10*np.log10(Tatm_1_rural),label=r'$\lambda = 795 \, nm$')
plt.plot(elevations,-10*np.log10(Tatm_2_rural),label=r'$\lambda = 1550 \, nm$')
plt.xlabel('Satellite elevation [deg]')
# plt.ylabel('Atmospheric absorption losses [dB]')
plt.title('Rural aerosol model - vis = 23 km')
plt.grid()

plt.suptitle('Midlatitude summer')

plt.legend()

#%% sim-3 atmopsheric background

c1 = {'model': channel.atmModel['MIDLAT_SUMMER'],
      'h1': 0.,
      'angle': 45,
      'wlshort': 400.0,
      'wllong': 2000.0,
      'wlstep': 10,
      'itype': 3,
      'iemsct': 2,
      'ihaze': channel.aeroModel['RURAL_23KM'],
      'gndAlt': 0.0,
      'vis': 0.0,
      'iseasn': 0,
      'imult': 1}

radResArray = lowtran.scatter(c1)
wls = radResArray.wavelength_nm
radVals = radResArray.pathscatter[0,:,0]

plt.figure()
plt.semilogy(wls,radVals,label='Rural - 23 km')
plt.xlim([400,2000])
# plt.ylim([1e-9,1.5e-2])
plt.xlabel('Wavelength [nm]')
plt.ylabel(r'Radiance [W $ster^{-1}$ $cm^{-2}$ $\mu m^{-1}$]')


c1['ihaze'] = channel.aeroModel['URBAN_5KM']
radResArray_1 = lowtran.scatter(c1)
radVals_1 = radResArray_1.pathscatter[0,:,0]
plt.semilogy(wls,radVals_1,label='Urban - 5 km')

# plt.axvline(850)
# plt.axvline(1550)

plt.title('Diffuse atmospheric radiance spectrum')
plt.legend(loc='lower left')
plt.tight_layout()

plt.grid('on')

#%% sim-4 telescope dependance - diffraction losses

RxRadius = np.array([0.1, 0.2, 0.4, 0.75])

diffLoss_1 = np.zeros((len(lgth),len(RxRadius)))
diffLoss_2 = np.zeros((len(lgth),len(RxRadius)))

for i in range(len(RxRadius)):
    diffLoss_1[:,i] = -10*np.log10(1-np.exp(-2*(RxRadius[i]/groundSpot_1)**2))
    diffLoss_2[:,i] = -10*np.log10(1-np.exp(-2*(RxRadius[i]/groundSpot_2)**2))


plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(lgth,diffLoss_1)
plt.xlabel('Satellite distance [km]')
plt.ylabel('Diffraction losses [dB]')
plt.title('Signal at 795 nm')
plt.legend(['200 mm','400 mm', '800 mm', '1500 mm'])
plt.grid()


plt.subplot(122)
plt.plot(lgth,diffLoss_2)
plt.xlabel('Satellite distance [km]')
plt.title('Signal at 1550 nm')
plt.grid()

plt.suptitle('Diffraction losses for different receiver diameters')

#%% sim-5 overall link budget at zenith

# wl_1 = 795e-9
# wl_2 = 1550e-9

txApRadius = 0.04
txDiv_1 = wl_1/(np.pi*txApRadius)
txDiv_2 = wl_2/(np.pi*txApRadius)

lgth = np.linspace(300,2500,100)

sigmaPoint = [1e-6, 5e-6, 10e-6]
Cn2 = 0

NavgPoints = 1000

# atmosphere models
# urban - 5 km
atmMod1urban = channel.AtmosphereTransmittanceModel(wl_1, 0, aerosolModel='URBAN_5KM')
atmMod2urban = channel.AtmosphereTransmittanceModel(wl_2, 0, aerosolModel='URBAN_5KM')
# rural - 23 km
atmMod1rural = channel.AtmosphereTransmittanceModel(wl_1, 0, aerosolModel='RURAL_23KM')
atmMod2rural = channel.AtmosphereTransmittanceModel(wl_2, 0, aerosolModel='RURAL_23KM')


TatmUrban_1 = atmMod1urban.calculateTransmittance(90)
TatmUrban_2 = atmMod2urban.calculateTransmittance(90)
TatmRural_1 = atmMod1rural.calculateTransmittance(90)
TatmRural_2 = atmMod2rural.calculateTransmittance(90)

TsimZenUrban_1 = np.zeros((len(lgth),len(RxRadius),len(sigmaPoint)))
TsimZenUrban_2 = np.zeros((len(lgth),len(RxRadius),len(sigmaPoint)))
TsimZenRural_1 = np.zeros((len(lgth),len(RxRadius),len(sigmaPoint)))
TsimZenRural_2 = np.zeros((len(lgth),len(RxRadius),len(sigmaPoint)))

calculate = True
if calculate:
    tStart = timer()
    for k in range(len(sigmaPoint)):
        for i in range(len(RxRadius)):
            channelUrban_1 = ls.FixedSatelliteLossModel(txDiv_1, sigmaPoint[k], RxRadius[i], Cn2, wl_1, Tatm=TatmUrban_1)
            channelUrban_2 = ls.FixedSatelliteLossModel(txDiv_2, sigmaPoint[k], RxRadius[i], Cn2, wl_2, Tatm=TatmUrban_2)
            channelRural_1 = ls.FixedSatelliteLossModel(txDiv_1, sigmaPoint[k], RxRadius[i], Cn2, wl_1, Tatm=TatmRural_1)
            channelRural_2 = ls.FixedSatelliteLossModel(txDiv_2, sigmaPoint[k], RxRadius[i], Cn2, wl_2, Tatm=TatmRural_2)
        
            for j in range(len(lgth)):
                TsimZenUrban_1[j,i,k] = np.mean(np.array([(1-channelUrban_1._sample_loss_probability(lgth[j])) for k in range(NavgPoints)]))
                TsimZenUrban_2[j,i,k] = np.mean(np.array([(1-channelUrban_2._sample_loss_probability(lgth[j])) for k in range(NavgPoints)]))
                TsimZenRural_1[j,i,k] = np.mean(np.array([(1-channelRural_1._sample_loss_probability(lgth[j])) for k in range(NavgPoints)]))
                TsimZenRural_2[j,i,k] = np.mean(np.array([(1-channelRural_2._sample_loss_probability(lgth[j])) for k in range(NavgPoints)]))
    tEnd = timer()
    print('Simulation time =',tEnd-tStart,'sec.')
    
    np.savez('zenLossSim.npz',lgth=lgth,TsimZenUrban_1=TsimZenUrban_1,TsimZenRural_1=TsimZenRural_1,TsimZenUrban_2=TsimZenUrban_2,TsimZenRural_2=TsimZenRural_2)

#%% plot the results of the simulation
ymin = 5
ymax = 58

load = False
if load:
    ll = np.load('zenLossSim.npz')
    lgth=ll['lgth']
    TsimZenUrban_1=ll['TsimZenUrban_1']
    TsimZenRural_1=ll['TsimZenRural_1']
    TsimZenUrban_2=ll['TsimZenUrban_2']
    TsimZenRural_2=ll['TsimZenRural_2']

style = ('-',':','--')
lineLegStyle = []

mpl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4','#ff7f0e','#2ca02c','#d62728']) 

plt.figure(figsize=(10,10))

for k in range(len(sigmaPoint)):
    # plt.gca().set_prop_cycle(None)
        
    plt.subplot(221)
    plt.plot(lgth, -10*np.log10(TsimZenUrban_1[:,:,k]), style[k])
    plt.ylim([ymin,ymax])
    plt.ylabel('Average losses [dB]')
    plt.grid('on')
    plt.title('795 nm\nUrban - visibility 5 km')
    plt.legend(['200 mm','400 mm', '800 mm', '1500 mm'],title='Rx diameter',loc = 'lower right')
    
    plt.subplot(222)
    line = plt.plot(lgth, -10*np.log10(TsimZenUrban_2[:,:,k]), style[k])
    lineLegStyle.append(line[0])
    plt.ylim([ymin,ymax])
    plt.grid('on')
    plt.title('1550 nm\nUrban - visibility 5 km')
    plt.legend(lineLegStyle,[r'$1 \, \mu rad$',r'$5 \, \mu rad$',r'$10 \, \mu rad$'], title='Pointing error', loc='lower right')
    
    plt.subplot(223)
    plt.plot(lgth, -10*np.log10(TsimZenRural_1[:,:,k]), style[k])
    plt.ylim([ymin,ymax])
    plt.ylabel('Average losses [dB]')
    plt.xlabel('Satellite altitude [km]')
    plt.grid('on')
    plt.title('Rural - visibility 23 km')
    
    plt.subplot(224)
    plt.plot(lgth, -10*np.log10(TsimZenRural_2[:,:,k]), style[k])
    plt.ylim([ymin,ymax])
    plt.xlabel('Satellite altitude [km]')
    plt.grid('on')
    plt.title('Rural - visibility 23 km')

plt.suptitle(r'Losses at zenith')
plt.tight_layout()

#%% sim-6 total passage simulation

# model parameters
# wl_1 = 795e-9
# wl_2 = 1550e-9

txApRadius = 0.04
txDiv_1 = wl_1/(np.pi*txApRadius)
txDiv_2 = wl_2/(np.pi*txApRadius)

# sigmaPoint = [1e-6, 5e-6, 10e-6]
sigmaPoint = 5e-6
Cn2 = 0

# atmosphere models
# urban - 5 km
atmMod1urban = channel.AtmosphereTransmittanceModel(wl_1, 0, aerosolModel='URBAN_5KM')
atmMod2urban = channel.AtmosphereTransmittanceModel(wl_2, 0, aerosolModel='URBAN_5KM')
# rural - 23 km
atmMod1rural = channel.AtmosphereTransmittanceModel(wl_1, 0, aerosolModel='RURAL_23KM')
atmMod2rural = channel.AtmosphereTransmittanceModel(wl_2, 0, aerosolModel='RURAL_23KM')

# satellite object
satAlt = 500
incAngle = 0
sat = channel.Satellite(simType='polOrbPass',incAngle=incAngle,satAlt=satAlt)

# Paris ground station
latParis = 48.857
# latParis = 0.
longParis = 2.352
altParis = 80.
# Matera ground station
latMLRO = 40.6486
longMLRO = 16.7046
altMLRO = 536.9


staParis = channel.GroundStation(latParis, longParis, altParis, 'Paris')
staMLRO = channel.GroundStation(latMLRO, longMLRO, altMLRO, 'MLRO')

# create the channels
downSatParis_1 = channel.SimpleDownlinkChannel(sat, staParis, wl_1, atmMod1urban)
downSatParis_2 = channel.SimpleDownlinkChannel(sat, staParis, wl_2, atmMod2urban)

downSatMLRO_1 = channel.SimpleDownlinkChannel(sat, staMLRO, wl_1, atmMod1rural)
downSatMLRO_2 = channel.SimpleDownlinkChannel(sat, staMLRO, wl_2, atmMod2rural)

# calculate the communication time to create the timeList
# Earth parameters (from Daniele's code)
Rt = 6.37e6     # Earth radius
M = 5.97e24     # Earth mass
G = 6.67e-11    # Gravitational constant

hs = sat.satAlt
Omega = np.sqrt( G*M / (Rt + hs)**3)

Tcom = 2 * np.arctan( np.sqrt(hs**2 + 2*Rt*hs) / Rt ) / Omega
# Tcom = 2 * np.arccos( Rt / (Rt+hs) ) / Omega


# create the timeList
dt = startTime = datetime.now()
endTime = startTime + timedelta(seconds=Tcom)
timeStep = timedelta(seconds = 10.)

timeList = []
while dt < endTime:
    timeList.append(dt)
    dt += timeStep

# calculate the satellite orbit
lenSatParis_1, tSatParis_1, elSatParis_1 = downSatParis_1.calculateChannelParameters(timeList)
lenSatParis_2, tSatParis_2, elSatParis_2 = downSatParis_2.calculateChannelParameters(timeList)

lenSatMLRO_1, tSatMLRO_1, elSatMLRO_1 = downSatMLRO_1.calculateChannelParameters(timeList)
lenSatMLRO_2, tSatMLRO_2, elSatMLRO_2 = downSatMLRO_2.calculateChannelParameters(timeList)


# # calculate the orbit parameters using the [Moll et al.] model.
# psi = downSatParis_1.groundStation.latitude

# deltaI = downSatParis_1.satellite.incAngle
# hs = downSatParis_1.satellite.satAlt

# deltaMin = np.arccos( np.cos(psi)*np.cos(deltaI) / np.sqrt(1 - ( np.cos(psi)*np.sin(deltaI) )**2) )
# tMin = timeList[ int(len(timeList)/2) ]

# # Earth parameters (from Daniele's code)
# Rt = 6.37e6     # Earth radius
# M = 5.97e24     # Earth mass
# G = 6.67e-11    # Gravitational constant

# Omega = np.sqrt( G*M / (Rt + hs)**3)

# relTime = np.array([(timeList[i] - tMin).total_seconds() for i in range(len(timeList))])
# delta = Omega * relTime + deltaMin

# Zc = np.arccos( np.sin(psi)*np.sin(delta) + np.cos(psi)*np.cos(delta)*np.cos(deltaI) )
# Z = np.arcsin( (Rt+hs)*np.sin(Zc) / np.sqrt( Rt**2 + (Rt+hs)**2 - 2*Rt*(Rt+hs)*np.cos(Zc)) )
# elevation = 90 - np.rad2deg(Z)

# channelLength = -Rt*np.cos(Z) + np.sqrt( (Rt*np.cos(Z))**2 + 2*Rt*hs + hs**2 )

# atmTrans = np.array([downSatParis_1.atm.calculateTransmittance(elevation[i]) for i in range(len(timeList))])


times = np.array([ (timeList[i]-timeList[0]).seconds  for i in range(len(timeList)) ])

# #%% Plot passage on Paris
# plt.figure(figsize=(18,6))

# indx = np.nonzero(elSatParis_1>20)[0]

# plt.subplot(131)
# plt.plot(times[indx],elSatParis_1[indx])
# # plt.plot(times/60,elevation,'b')
# plt.ylim([20,90])
# plt.ylabel('Elevation [degrees]')
# plt.xlabel('Passage time [sec]')
# plt.grid()

# plt.subplot(132)
# plt.plot(times[indx],lenSatParis_1[indx]/1000)
# # plt.plot(times/60,channelLength/1000,'b')
# plt.ylabel('Channel length [km]')
# plt.xlabel('Passage time [sec]')
# plt.grid()

# plt.subplot(133)
# plt.plot(times[indx],-10*np.log10(tSatParis_1[indx]))
# plt.plot(times[indx],-10*np.log10(tSatParis_2[indx]))
# # plt.plot(times/60,atmTrans,'b') 
# plt.ylabel('Atmospheric losses [dB]')
# plt.xlabel('Passage time [sec]')
# plt.legend(['795 nm', '1550 nm'])
# plt.grid()

# plt.suptitle('Passage at zenith on Paris - urban atmosphere')
# plt.tight_layout()

#%% Plot passage on Matera
plt.figure(figsize=(12,6))

indx = np.nonzero(elSatMLRO_1>20)[0]

plt.subplot(121)
plt.plot(times[indx],elSatMLRO_1[indx])
# plt.plot(times/60,elevation,'b')
plt.ylim([20,90])
plt.ylabel('Elevation [degrees]')
plt.xlabel('Passage time [sec]')
plt.grid()

plt.subplot(122)
plt.plot(times[indx],lenSatMLRO_1[indx]/1000)
# plt.plot(times/60,channelLength/1000,'b')
plt.ylabel('Channel length [km]')
plt.xlabel('Passage time [sec]')
plt.grid()

plt.suptitle('Satellite passage at zenith - altitude 1000 km')
plt.tight_layout()

plt.figure()
plt.plot(times[indx],-10*np.log10(tSatMLRO_1[indx]))
plt.plot(times[indx],-10*np.log10(tSatMLRO_2[indx]))
# plt.plot(times/60,atmTrans,'b') 
plt.ylabel('Atmospheric losses [dB]')
plt.xlabel('Passage time [sec]')
plt.legend(['795 nm', '1550 nm'])
plt.grid()

plt.suptitle('Atmospheric losses - rural atmosphere')
plt.tight_layout()

#%% sim-7 Atmospheric losses of the passage - 500 km

# wl_1 = 795e-9
# wl_2 = 1550e-9

txApRadius = 0.04
txDiv_1 = wl_1/(np.pi*txApRadius)
txDiv_2 = wl_2/(np.pi*txApRadius)

sigmaPoint = [1e-6, 5e-6, 10e-6]
Cn2 = 0

NavgPoints = 1000

TpassMLRO_1 = np.zeros((len(indx),len(RxRadius),len(sigmaPoint)))
TpassMLRO_2 = np.zeros((len(indx),len(RxRadius),len(sigmaPoint)))

timesIdx = times[indx]
lenSatMLROidx_1 = lenSatMLRO_1[indx]
elSatMLROidx_1 = elSatMLRO_1[indx]
tSatMLROidx_1 = tSatMLRO_1[indx]
tSatMLROidx_2 = tSatMLRO_2[indx]

calculate = True
if calculate:
    tStart = timer()
    for k in range(len(sigmaPoint)):
        for i in range(len(RxRadius)):
            for j in range(len(indx)):
                Tatm_1 = tSatMLROidx_1[j]
                Tatm_2 = tSatMLROidx_2[j]
                
                channelMLRO_1 = ls.FixedSatelliteLossModel(txDiv_1, sigmaPoint[k], RxRadius[i], Cn2, wl_1, Tatm=Tatm_1)
                channelMLRO_2 = ls.FixedSatelliteLossModel(txDiv_2, sigmaPoint[k], RxRadius[i], Cn2, wl_2, Tatm=Tatm_2)
                
                TpassMLRO_1[j,i,k] = np.mean(np.array([(1-channelMLRO_1._sample_loss_probability(lenSatMLROidx_1[j]/1000)) for k in range(NavgPoints)]))
                TpassMLRO_2[j,i,k] = np.mean(np.array([(1-channelMLRO_2._sample_loss_probability(lenSatMLROidx_1[j]/1000)) for k in range(NavgPoints)]))
    tEnd = timer()
    print('Simulation time =',tEnd-tStart,'sec.')
    
#%% plot of the results
ymin = 10
ymax = 50


style = ('-',':','--')
lineLegStyle = []
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4','#ff7f0e','#2ca02c','#d62728']) 


plt.figure(figsize=(12,6))

for k in range(len(sigmaPoint)):
    # plt.gca().set_prop_cycle(None)
        
    plt.subplot(121)
    plt.plot(timesIdx, -10*np.log10(TpassMLRO_1[:,:,k]), style[k])
    plt.ylim([ymin,ymax])
    plt.ylabel('Average losses [dB]')
    plt.grid('on')
    plt.title('795 nm')
    plt.legend(['200 mm','400 mm', '800 mm', '1500 mm'],title='Rx diameter',loc = 'lower right')
    
    plt.subplot(122)
    line = plt.plot(timesIdx, -10*np.log10(TpassMLRO_2[:,:,k]), style[k])
    lineLegStyle.append(line[0])
    plt.ylim([ymin,ymax])
    plt.grid('on')
    plt.title('1550 nm')
    plt.legend(lineLegStyle,[r'$1 \, \mu rad$',r'$5 \, \mu rad$',r'$10 \, \mu rad$'], title='Pointing error', loc='lower right')

plt.suptitle(r'Losses for a satellite passage at zenith - altitude 500 km')
plt.tight_layout()
