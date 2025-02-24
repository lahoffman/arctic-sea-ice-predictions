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

#historical
#------------------------------------------------------
#sithick,siconc
#----------
filepath_thick_hist = '/Users/hoffmanl/coriolis/historical/sithick/sithick_SImon_'
filepath_siconc_hist = '/Users/hoffmanl/coriolis/historical/siconc/siconc_SImon_'
models_hist = ['ACCESS-CM2_historical','ACCESS-ESM1-5_historical','CanESM5_historical','IPSL-CM6A-LR_historical','MIROC6_historical','MRI-ESM2-0_historical','CESM2-WACCM_historical','HadGEM3-GC31-LL_historical','HadGEM-GC31-MM_historical']
sic_name = ['ACCESS-CM2','ACCESS-ESM1-5','CanESM5','IPSL-CM6A-LR','MIROC6','MRI-ESM2-0','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM-GC31-MM']
model_path_tail_hist = '_185001-201412.nc'
g = ['gn','gn','gn','gn','gn','gm','gn','gn','gn']
ensemble_numbers = [10,40,25,33,50,10,3,50,4]
f = [1,1,1,1,1,1,1,3,3]
p = [1,1,2,1,1,1,1,1,1]
lati = ['latitude','latitude','latitude','nav_lat','latitude','latitude','lat','latitude','latitude']
loni = ['longitude','longitude','longitude','nav_lon','longitude','longitude','lon','longitude','longitude']

#areacello
#----------
filepatha_hist = '/Users/hoffmanl/coriolis/historical/areacello/areacello_Ofx_'
filepathtaila = 'gn.nc'
modelsa_hist = ['ACCESS-CM2_historical','ACCESS-ESM1-5_historical','CanESM5_historical','IPSL-CM6A-LR_historical','MIROC6_historical','MRI-ESM2-0_historical','CESM2-WACCM_historical','HadGEM3-GC31-LL_hist-1950','HadGEM-GC31-MM_hist-1950']


#ssp585
#------------------------------------------------------
#sithick, siconc
#----------
filepath_thick_ssp = '/Users/hoffmanl/coriolis/ssp585/sithick/sithick_SImon_'
filepath_siconc_ssp = '/Users/hoffmanl/coriolis/ssp585/siconc/siconc_SImon_'
models_ssp = ['ACCESS-CM2_ssp585','ACCESS-ESM1-5_ssp585','CanESM5_ssp585','IPSL-CM6A-LR_ssp585','MIROC6_ssp585','MRI-ESM2-0_ssp585','CESM2-WACCM_ssp585','HadGEM3-GC31-LL_ssp585','HadGEM-GC31-MM_ssp585']
sic_name = ['ACCESS-CM2','ACCESS-ESM1-5','CanESM5','IPSL-CM6A-LR','MIROC6','MRI-ESM2-0','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM-GC31-MM']
model_path_tail_ssp = '_201501-210012.nc'
g = ['gn','gn','gn','gn','gn','gn','gn','gn','gn']
ensemble_numbers_ssp = [10,40,25,7,50,5,3,4,4]
f = [1,1,1,1,1,1,1,3,3]
p = [1,1,2,1,1,1,1,1,1]
ipsl = [1,2,3,4,6,14,33]
lati = ['latitude','latitude','latitude','nav_lat','latitude','latitude','lat','latitude','latitude']
loni = ['longitude','longitude','longitude','nav_lon','longitude','longitude','lon','longitude','longitude']


#areacello
#----------
filepatha_ssp = '/Users/hoffmanl/coriolis/ssp585/areacello/areacello_Ofx_'
filepathtaila = 'gn.nc'
modelsa_ssp = ['ACCESS-CM2_ssp585','ACCESS-ESM1-5_ssp585','CanESM5_ssp585','IPSL-CM6A-LR_ssp585','MIROC6_ssp585','MRI-ESM2-0_ssp585','CESM2-WACCM_ssp585','HadGEM3-GC31-LL_ssp585','HadGEM-GC31-MM_ssp585']
#------------------------------------------------------
#------------------------------------------------------

#SIE: loop through models, spatial mean, ensemble mean, anomaly
#mount 'siextentn' on server to coriolis on local machine 
#sshfs -o idmap=user coriolis%gwcism:/cofast/lhoffman/cmip6/SImon ~/coriolis -o cache=no
#------------------------------------------------------
ensemble_mean = []
ensemble_stdev = []
area_anomaly = []
siat_i = []
for i in range(6):
    ensemble_number = ensemble_numbers_ssp[i]+1
    
    siei = []
    for j in range(1,ensemble_number):
        
        #file
        #------------------
        if i == 3:
            jj = ipsl[j-1]
            ensemble_member = ['_r{}i1p{}f{}_'.format(jj,p[i],f[i])]
        else:
            ensemble_member = ['_r{}i1p{}f{}_'.format(j,p[i],f[i])]
            print(ensemble_member[0])

        #------------------------------------------------------
        #------------------------------------------------------
        #historical
        #------------------------------------------------------
        #------------------------------------------------------
        loadpatht = filepath_thick_hist+models_hist[i]+ensemble_member[0]+g[i]+model_path_tail_hist
        print(loadpatht)
        #ncdisp(loadpath)
            
        loadpathc = filepath_siconc_hist+models_hist[i]+ensemble_member[0]+g[i]+model_path_tail_hist
        loadpath_hist = loadpathc
        print(loadpathc)
        #ncdisp(loadpathc)
        
        ensemble_membera = ['_r1i1p1f1_']
        loadpatha = filepatha_hist+modelsa_hist[i]+ensemble_membera[0]+filepathtaila
        print(loadpatha)
        #ncdisp(loadpatha)
          
        #save vars, SIT
        #------------------
        dataset = nc.Dataset(loadpatht,'r')
        latt = dataset.variables[lati[i]]
        lonn = dataset.variables[loni[i]]
        ti = dataset.variables['sithick']
        
        sithick = np.array(ti)
        lat = np.array(latt)
        lon = np.array(lonn)
        
    
        #set fill value to NaN
        threshold = 101
        mask = sithick > threshold
        sithick[mask] = np.nan
        
        #set non-Arctic sit to NaN
        mask = lat < 0
        if np.size(mask) < 1000:
            ny = np.shape(sithick)[2]
            maskt1= np.transpose(np.tile(mask,[ny,1]))
            maskt = np.tile(maskt1,[1980,1,1])
        else:
            maskt = np.tile(mask,[1980,1,1])
        sithick[maskt] = np.nan
        

        if i == 0:
            #set time
            days_since = np.array(dataset.variables['time'])
            time0 = "1850-01-01"
            time0_parsed = datetime.strptime(time0, "%Y-%m-%d")
            time = [time0_parsed+timedelta(days=int(days)) for days in days_since]
        
        
        #save vars, areacello
        #------------------
        datasetC = nc.Dataset(loadpathc,'r')
        datasetA = nc.Dataset(loadpatha,'r')
        
        latc = datasetA.variables[lati[i]]
        lonc = datasetA.variables[loni[i]]           
        aci = datasetA.variables['areacello']
        
        latC = np.array(latc)
        lonC = np.array(lonc)
        areacello = np.array(aci)
        

        if i == 4:
            ci = datasetC.variables['sic']
        else:
            ci = datasetC.variables['siconc']     
        
        siconc = np.array(ci)
        latC = np.array(latc)
        lonC = np.array(lonc)
        areacello = np.tile(np.array(aci),[1980,1,1])
          
        #set fill value to NaN
        threshold = 101
        mask = siconc > threshold
        siconc[mask] = np.nan
        
        #set non-Arctic sic to NaN
        mask = latC < 0
        if np.size(mask) < 1000:
            ny = np.shape(siconc)[2]
            maskt1= np.transpose(np.tile(mask,[ny,1]))
            maskt = np.tile(maskt1,[1980,1,1])
        else:
            maskt = np.tile(mask,[1980,1,1])
        siconc[maskt] = np.nan
        
        #calculate siv = siconc * areacello * sithick
        #------------------
        sivol = siconc*sithick/100
          
        
        #calculate area (sit > threshold) 
        #------------------
        siv_threshold = 1.25
        area_threshi = []
        for l in range(1980):
            sivl = sivol[l,:,:]
            siv_gt_threshold_mask = sivl > siv_threshold
            areal = areacello[l,:,:]
            area_siv_thresholdi = np.sum(areal[siv_gt_threshold_mask])
            area_threshi.append(area_siv_thresholdi)
        
        area_siv_threshold_hist = np.array(area_threshi)
        #------------------------------------------------------
        #------------------------------------------------------
        
        
        
        #------------------------------------------------------
        #------------------------------------------------------
        #ssp585
        #------------------------------------------------------
        #------------------------------------------------------
        loadpatht = filepath_thick_ssp+models_ssp[i]+ensemble_member[0]+g[i]+model_path_tail_ssp
        print(loadpatht)
        #ncdisp(loadpath)
            
        loadpathc = filepath_siconc_ssp+models_ssp[i]+ensemble_member[0]+g[i]+model_path_tail_ssp
        print(loadpathc)
        loadpath_ssp = loadpathc
        #ncdisp(loadpathc)
        
        ensemble_membera = ['_r1i1p1f1_']
        loadpatha = filepatha_ssp+modelsa_ssp[i]+ensemble_membera[0]+filepathtaila
        print(loadpatha)
        #ncdisp(loadpatha)
          
        #save vars, SIT
        #------------------
        dataset = nc.Dataset(loadpatht,'r')
        latt = dataset.variables[lati[i]]
        lonn = dataset.variables[loni[i]]
        ti = dataset.variables['sithick']
        
        sithick = np.array(ti)
        lat = np.array(latt)
        lon = np.array(lonn)
        
    
        #set fill value to NaN
        threshold = 101
        mask = sithick > threshold
        sithick[mask] = np.nan
        
        #set non-Arctic sit to NaN
        mask = lat < 0
        if np.size(mask) < 1000:
            ny = np.shape(sithick)[2]
            maskt1= np.transpose(np.tile(mask,[ny,1]))
            maskt = np.tile(maskt1,[1032,1,1])
        else:
            maskt = np.tile(mask,[1032,1,1])
        sithick[maskt] = np.nan
        

        if i == 0:
            #set time
            days_since = np.array(dataset.variables['time'])
            time0 = "2015-01-01"
            time0_parsed = datetime.strptime(time0, "%Y-%m-%d")
            time = [time0_parsed+timedelta(days=int(days)) for days in days_since]
        
        
        #save vars, areacello
        #------------------
        datasetC = nc.Dataset(loadpathc,'r')
        datasetA = nc.Dataset(loadpatha,'r')
        
        latc = datasetA.variables[lati[i]]
        lonc = datasetA.variables[loni[i]]           
        aci = datasetA.variables['areacello']
        
        latC = np.array(latc)
        lonC = np.array(lonc)
        areacello = np.array(aci)
        

        if i == 7:
            ci = datasetC.variables['sic']
        else:
            ci = datasetC.variables['siconc']     
        
        siconc = np.array(ci)
        latC = np.array(latc)
        lonC = np.array(lonc)
        areacello = np.tile(np.array(aci),[1032,1,1])
          
        #set fill value to NaN
        threshold = 101
        mask = siconc > threshold
        siconc[mask] = np.nan
        
        #set non-Arctic sic to NaN
        mask = latC < 0
        if np.size(mask) < 1000:
            ny = np.shape(siconc)[2]
            maskt1= np.transpose(np.tile(mask,[ny,1]))
            maskt = np.tile(maskt1,[1032,1,1])
        else:
            maskt = np.tile(mask,[1032,1,1])
        siconc[maskt] = np.nan
        
        #calculate siv = siconc * areacello * sithick
        #------------------
        sivol = siconc*sithick/100
          
        
        #calculate area (sit > threshold) 
        #------------------
        siv_threshold = 1.25
        area_threshi = []
        for l in range(1032):
            sivl = sivol[l,:,:]
            siv_gt_threshold_mask = sivl > siv_threshold
            areal = areacello[l,:,:]
            area_siv_thresholdi = np.sum(areal[siv_gt_threshold_mask])
            area_threshi.append(area_siv_thresholdi)
        
        area_siv_threshold_ssp585 = np.array(area_threshi)
        
        siextent = np.concatenate((area_siv_threshold_hist,area_siv_threshold_ssp585),axis=0)
        #------------------------------------------------------
        #------------------------------------------------------
        
        #set time
        if i == 0:
            dataset = nc.Dataset(loadpath_hist,'r')
            days_since_hist = np.array(dataset.variables['time'])
            dataset = nc.Dataset(loadpath_ssp,'r')
            days_since_ssp = np.array(dataset.variables['time'])
            
            days_since = np.concatenate((days_since_hist,days_since_ssp),axis=0)
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
            outer = ([])
            for l in range(nd):
                data_series = pd.Series(data[:,l]) 
                data_moving_mean = data_series.rolling(window=k,axis=0).mean()
                data_movmean = np.array(data_moving_mean)
                outer.append(data_movmean)
                outernp = np.array(outer)[:,:,np.newaxis]
            
            if k == 1:
                sie_movmean = outernp
            else:
                sie_movmean = np.append(sie_movmean,outernp,axis=2)
        
        #save
        siei.append(sie_movmean)
        
    days_since = np.array(dataset.variables['time'])
    
    # c. take spatial mean
    area = np.array(siei)
    
    # d. remove multi-member mean
    ensemble_mean_i = np.nanmean(area,axis=0)
    ensemble_stdev_i = np.nanstd(area,axis=0)
    area_anomaly_i = (area-ensemble_mean_i[np.newaxis,:,:,:]) #remove mean


    #save
    siat_i.append(area)
    area_anomaly.append(area_anomaly_i)
    ensemble_mean.append(ensemble_mean_i)
    ensemble_stdev.append(ensemble_stdev_i)


    
#variables --> numpy arrays
area_ensemble_mean = np.array(ensemble_mean)
area_ensemble_stdev = np.array(ensemble_stdev)

for i in range(6):
    siei = np.array(siat_i[i])
    sieai = np.array(area_anomaly[i])

    if i == 0:
        sie = siei
        siea = sieai

    else: 
        sie = np.append(sie,siei,axis=0)
        siea = np.append(siea,sieai,axis=0)

    
siat = sie    
area_ensemble_anomaly = siea

'''
# d. save to .nc file
# RMMM = Remove multi-member mean ; SM = spatial mean ; NORM = normalzied to zero mean and one standard deviation ; T = moving mean ; TFmon = time frames, monthly
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siat_1p25m_SImon_models_concat_hist_ssp585_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM_mon2sep_gn_185001-210012.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('nem',siat.shape[0]) #no. emsemble members
    file.createDimension('ntf',siat.shape[1]) #no. time frames
    file.createDimension('ny',siat.shape[2]) #no. years
    file.createDimension('nT',siat.shape[3]) #no. moving means
    file.createDimension('nm',area_ensemble_mean.shape[0]) #no. models

    
    #create variables
    area = file.createVariable('siat','f4',('nem','ntf','ny','nT'))
    area_em = file.createVariable('area_ensemble_mean','f4',('nm','ntf','ny','nT'))
    area_es = file.createVariable('area_ensemble_stdev','f4',('nm','ntf','ny','nT'))
    area_rmmm = file.createVariable('area_ensemble_anomaly','f4',('nem','ntf','ny','nT'))


    #write data to variables
    area[:] = siat
    area_em[:] = area_ensemble_mean
    area_es[:] = area_ensemble_stdev
    area_rmmm[:] = area_ensemble_anomaly
 
 
'''