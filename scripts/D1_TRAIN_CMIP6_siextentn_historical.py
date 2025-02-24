#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 19 2024

@author: hoffmanl
"""

#set up environment
#------------------------------------------------------
#------------------------------------------------------

#system
#------------------
import sys
import os
import csv
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset

#data processing
#------------------
#scipy
from scipy import stats, odr
from scipy.io import netcdf
from scipy.stats import norm
import h5py
import math 

#other
from datetime import datetime
from datetime import timedelta

#plotting
#------------------
#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable

#colorbars
import cmocean 

#import functions
#------------------
sys.path.append('/Users/hoffmanl/Documents/scripts/functions/')
from functions_general import ncdisp
from functions_general import movmean


#model file paths
#------------------------------------------------------
#------------------------------------------------------
filepath = '/Users/hoffmanl/coriolis/siextentn_SImon_'
models = ['ACCESS-CM2_historical','ACCESS-ESM1-5_historical','CanESM5_historical','IPSL-CM6A-LR_historical','MIROC6_historical','MRI-ESM2-0_historical','CESM2-WACCM_historical','HadGEM3-GC31-LL_historical','HadGEM-GC31-MM_historical']
sic_name = ['ACCESS-CM2','ACCESS-ESM1-5','CanESM5','IPSL-CM6A-LR','MIROC6','MRI-ESM2-0','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM-GC31-MM']
model_path_tail = '_185001-201412.nc'
g = ['gn','gn','gn','gn','gn','gm','gn','gn','gn']
ensemble_numbers = [10,40,25,33,50,10,3,50,4]
f = [1,1,1,1,1,1,1,3,3]
p = [1,1,2,1,1,1,1,1,1]
lati = ['latitude','latitude','latitude','nav_lat','latitude','latitude','lat','latitude','latitude']
loni = ['longitude','longitude','longitude','nav_lon','longitude','longitude','lon','longitude','longitude']


#SIE: loop through models, spatial mean, ensemble mean, anomaly
#mount 'siextentn' on server to coriolis on local machine 
#sshfs -o idmap=user coriolis%gwcism:/cofast/lhoffman/cmip6/SImon/historical/siextentn ~/coriolis -o cache=no
#------------------------------------------------------
ensemble_mean = []
ensemble_stdev = []
sie_anomaly = []
sie_anomaly_norm = []
siextent_i = []
for i in range(1):
    ensemble_number = ensemble_numbers[i]+1
    
    siei = []
    for j in range(1,ensemble_number):
        
        #file
        #------------------
        ensemble_member = ['_r{}i1p{}f{}_'.format(j,p[i],f[i])]
        print(ensemble_member[0])
        
        loadpath = filepath+models[i]+ensemble_member[0]+g[i]+model_path_tail
        print(loadpath)
        #ncdisp(loadpath)
           
        #save vars
        #------------------
        dataset = nc.Dataset(loadpath,'r')
        ei = dataset.variables['siextentn']
        
        siextent = np.reshape(np.array(ei),[1980,])

        #set fill value to NaN
        threshold = 1E20
        mask = siextent > threshold
        siextent[mask] = np.nan

        if i == 0:
            #set time
            days_since = np.array(dataset.variables['time'])
            time0 = "1850-01-01"
            time0_parsed = datetime.strptime(time0, "%Y-%m-%d")
            time = [time0_parsed+timedelta(days=int(days)) for days in days_since]
        
        #a. separate into time-frames
        #yearly mean
        dates = np.array(time)
        years = np.array([date.year for date in dates])
        unique_years = np.unique(years)
        sie_yearly_mean = np.array([np.mean(siextent[years==year],axis=0) for year in unique_years])

        #months
        months = np.array([date.month for date in dates])
        unique_months = np.unique(months)

        #monthly sie
        sie_mon = []
        time_mon = []
        for k in range(1,13):
            siem = siextent[months==k]
            timem = dates[months==k]
            sie_mon.append(siem)
            time_mon.append(timem)
        
        sie_monthly = np.transpose(np.array(sie_mon))
        time_monthly = np.array(time_mon)
        
        #seasonal sie        
        #winter (JFM)
        selected_data = siextent[np.isin(months,[1,2,3])]
        selected_years = years[np.isin(months,[1,2,3])]
        selected_time = dates[np.isin(months,[1,2,3])]
        sie_yearly_JFM = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
        time_JFM = dates[months==1]
        
        #spring (AMJ)
        selected_data = siextent[np.isin(months,[4,5,6])]
        selected_years = years[np.isin(months,[4,5,6])]
        selected_time = dates[np.isin(months,[4,5,6])]
        sie_yearly_AMJ = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
        time_AMJ = dates[months==1]
        
        #summer (JAS)
        selected_data = siextent[np.isin(months,[7,8,9])]
        selected_years = years[np.isin(months,[7,8,9])]
        selected_time = dates[np.isin(months,[7,8,9])]
        sie_yearly_JAS = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
        time_JAS = dates[months==1]
        
        #fall (OND)
        selected_data = siextent[np.isin(months,[10,11,12])]
        selected_years = years[np.isin(months,[10,11,12])]
        selected_time = dates[np.isin(months,[10,11,12])]
        sie_yearly_OND = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
        time_OND = dates[months==1]

        
        
        #b. moving mean
        data = np.concatenate([sie_yearly_mean[:,np.newaxis],sie_yearly_JFM[:,np.newaxis],sie_yearly_AMJ[:,np.newaxis],sie_yearly_JAS[:,np.newaxis],sie_yearly_OND[:,np.newaxis],sie_monthly],axis=1) 
        nd = np.shape(data)[1]
        for k in range(1,11):
            outernp = []
            for l in range(nd):
                data_series = pd.Series(data[:,l]) 
                data_moving_mean = data_series.rolling(window=k,axis=0).mean()
                data_movmean = np.array(data_moving_mean)
                outernp.append(data_moving_mean.values[:, np.newaxis]) 
            
            outernp = np.concatenate(outernp, axis=1)
            
            if k == 1:
                sie_movmean = outernp[:, :, np.newaxis]
            else:
                sie_movmean = np.concatenate((sie_movmean, outernp[:, :, np.newaxis]), axis=2)
        
        #save
        siei.append(sie_movmean)
        
    days_since = np.array(dataset.variables['time'])
    
    # c. take spatial mean
    sie = np.array(siei)
    
    # d. remove multi-member mean
    epsilon = 1e-10
    ensemble_mean_i = np.nanmean(sie,axis=0)
    ensemble_stdev_i = np.nanstd(sie,axis=0)
    sie_anomaly_i_norm = np.divide((sie-ensemble_mean_i[np.newaxis,:,:,:]),ensemble_stdev_i[np.newaxis,:,:,:]+ epsilon) #normalized
    sie_anomaly_i = (sie-ensemble_mean_i[np.newaxis,:]) #remove mean


    #save
    siextent_i.append(sie)
    sie_anomaly_norm.append(sie_anomaly_i_norm)
    sie_anomaly.append(sie_anomaly_i)
    ensemble_mean.append(ensemble_mean_i)
    ensemble_stdev.append(ensemble_stdev_i)


#variables --> numpy arrays
sie_ensemble_mean = np.array(ensemble_mean)
sie_ensemble_stdev = np.array(ensemble_stdev)

for i in range(6):
    siei = np.array(siextent_i[i])
    sieai = np.array(sie_anomaly[i])
    sieani = np.array(sie_anomaly_norm[i])
    if i == 0:
        sie = siei
        siea = sieai
        siean = sieani
    else: 
        sie = np.append(sie,siei,axis=0)
        siea = np.append(siea,sieai,axis=0)
        siean = np.append(siean,sieani,axis=0)
    
siextent = sie    
sie_ensemble_anomaly = siea
sie_ensemble_anomaly_normalized = siean   

'''
import matplotlib.pyplot as plt
import numpy as np
jj = np.concatenate((np.ones(5),2*np.ones(5)),axis=0).astype(int)
kk = np.concatenate((np.arange(5),np.arange(5)),axis=0).astype(int)

#time frames, tf = 17
for i in range(13,14):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    #moving mean, T = 10
    for j in range(1):
        siep = sie[:,i,:,j]
        
        #ensemble member
        for h in range(10):
            axs[jj[h]-1,kk[h]].plot(np.arange(1850,2015),sie_ensemble_anomaly_normalized[h,i,:,j])
        
'''
'''

# d. save to .nc file
# RMMM = Remove multi-member mean ; SM = spatial mean ; NORM = normalzied to zero mean and one standard deviation ; T = moving mean ; TFmon = time frames, monthly
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d2/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('nem',siextent.shape[0]) #no. emsemble members
    file.createDimension('ntf',siextent.shape[1]) #no. time frames
    file.createDimension('ny',siextent.shape[2]) #no. years
    file.createDimension('nT',siextent.shape[3]) #no. moving means
    file.createDimension('nm',sie_ensemble_mean.shape[0]) #no. models

    

    #create variables
    years = file.createVariable('unique_years','f4',('ny',)) 
    sie = file.createVariable('siextent','f4',('nem','ntf','ny','nT'))
    sie_em = file.createVariable('sie_ensemble_mean','f4',('nm','ntf','ny','nT'))
    sie_es = file.createVariable('sie_ensemble_stdev','f4',('nm','ntf','ny','nT'))
    sie_rmmm = file.createVariable('sie_ensemble_anomaly','f4',('nem','ntf','ny','nT'))
    sie_norm = file.createVariable('sie_ensemble_anomaly_normalized','f4',('nem','ntf','ny','nT'))

    #write data to variables
    years[:] = unique_years  
    sie[:] = siextent
    sie_em[:] = sie_ensemble_mean
    sie_es[:] = sie_ensemble_stdev
    sie_rmmm[:] = sie_ensemble_anomaly
    sie_norm[:] = sie_ensemble_anomaly_normalized
'''