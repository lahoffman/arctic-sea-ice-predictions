#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 Jan 2025

@author: hoffmanl

Summary: 
OBS: montly, moving mean
CMIP6 [historical + ssp585]: temporal subset to match obs, weighted mean of ensemble mean, 
remove bias (mean and trend) from observations --> forced & residual

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

#plotting
#------------------
#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

#colorbars
import cmocean 

#maps: cartopy
import cartopy.crs as ccrs
import cartopy.feature as cft
from cartopy import config
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)

#---------------------------------------------------------------------
#least squares function
#------------------
def least_squares(E,Y):
    """
    Calculates m using the formula m = inv(E'*E)*E'*Y.

    Parameters:
    - E: Design matrix
    - Y: Response vector

    Returns:
    - m: Resulting vector
    """
    
    E_transpose = np.transpose(E)
    E_transpose_E = np.dot(E_transpose,E)
    try:
        E_transpose_E_inv = np.linalg.inv(E_transpose_E)
        m = np.dot(np.dot(E_transpose_E_inv,E_transpose),Y)
        return m
    except np.linalg.LinAlgError:
        #Handle the case wehre matrix inversion is not possible
        print("Matrix inversion failed. Check the invertibility of E'*E.")
        return None
#---------------------------------------------------------------------  


#load PIOMAS data
#---------------------
loadpath = '/Users/hoffmanl/Documents/data/piomas/sivol_r1i1p1_mon_197901-202412_GCCS360x120.nc'
dataset =  nc.Dataset(loadpath,'r')
sivol = dataset.variables['sivol']
time = dataset.variables['time']
lon = dataset.variables['longitude']
lat = dataset.variables['latitude']
areacell = dataset.variables['areacello']

years = np.arange(1979,2025)
months = np.arange(1,13)
yearmon = np.tile(years,[12,1])
indexym = np.reshape(time,[12,46])
monthsf = np.transpose(np.tile(np.arange(1,13),[1,46]))

start_date = '1979-01'
end_date = '2025-01'
date_range = pd.date_range(start=start_date, end=end_date, freq='M')

years =np.reshape(yearmon.astype(int),(552,))
months = np.reshape(monthsf.astype(int),(552,))
time= [datetime(year, month, 1) for year, month in zip(years, months)]

#calculate area (sit > threshold) 
#------------------
siv_threshold = 1.25
area_threshi = []
for l in range(552):
    sivl = sivol[l,:,:]
    siv_gt_threshold_mask = sivl > siv_threshold
    areal = np.array(areacell)
    area_siv_thresholdi = np.sum(areal[siv_gt_threshold_mask])
    area_threshi.append(area_siv_thresholdi)

area_siv_threshold = np.array(area_threshi)
sivol = area_siv_threshold


#a. separate into desired time frames
#------------------
#yearly mean
dates = np.array(time)
years = np.array([date.year for date in dates])
unique_years = np.unique(years)
sivol_yearly_mean = np.array([np.nanmean(sivol[years==year],axis=0) for year in unique_years])

#months
months = np.array([date.month for date in dates])
unique_months = np.unique(months)

#monthly sie
sie_mon = []
time_mon = []
for k in range(1,13):
    siem = sivol[months==k]
    timem = dates[months==k]
    sie_mon.append(siem)
    time_mon.append(timem)

sivol_monthly = np.transpose(np.array(sie_mon))
time_monthly = np.array(time_mon)

#seasonal sie        
#winter (JFM)
selected_data = sivol[np.isin(months,[1,2,3])]
selected_years = years[np.isin(months,[1,2,3])]
selected_time = dates[np.isin(months,[1,2,3])]
sivol_yearly_JFM = np.array([np.nanmean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_JFM = dates[months==1]

#spring (AMJ)
selected_data = sivol[np.isin(months,[4,5,6])]
selected_years = years[np.isin(months,[4,5,6])]
selected_time = dates[np.isin(months,[4,5,6])]
sivol_yearly_AMJ = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_AMJ = dates[months==4]

#summer (JAS)
selected_data = sivol[np.isin(months,[7,8,9])]
selected_years = years[np.isin(months,[7,8,9])]
selected_time = dates[np.isin(months,[7,8,9])]
sivol_yearly_JAS = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_JAS = dates[months==7]

#fall (OND)
selected_data = sivol[np.isin(months,[10,11,12])]
selected_years = years[np.isin(months,[10,11,12])]
selected_time = dates[np.isin(months,[10,11,12])]
sivol_yearly_OND = np.array([np.nanmean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_OND = dates[months==10]

sivol_seasonal = np.concatenate([sivol_yearly_JFM[:,np.newaxis],sivol_yearly_AMJ[:,np.newaxis],sivol_yearly_JAS[:,np.newaxis],sivol_yearly_OND[:,np.newaxis]],axis=1) 


#b. take moving mean for T = 1:10 years
#------------------
data = np.concatenate([sivol_yearly_mean[:,np.newaxis],sivol_yearly_JFM[:,np.newaxis],sivol_yearly_AMJ[:,np.newaxis],sivol_yearly_JAS[:,np.newaxis],sivol_yearly_OND[:,np.newaxis],sivol_monthly],axis=1) 
nd = np.shape(data)[1]

for i in range(1,11):
    outer = ([])
    
    for j in range(nd):
        data_series = pd.Series(data[:,j]) 
        data_moving_mean = data_series.rolling(window=i,axis=0).mean()
        data_movmean = np.array(data_moving_mean)
        outer.append(data_movmean)
        outernp = np.array(outer)[:,:,np.newaxis]
    
    if i == 1:
        siat_movmean = outernp
    else:
        siat_movmean = np.append(siat_movmean,outernp,axis=2)
area_obs = siat_movmean

#extended moving mean for normalization case,T = 1-15 years
for i in range(1,16):
    outer = ([]) 
    for j in range(nd):
        data_series = pd.Series(data[:,j]) 
        data_moving_mean = data_series.rolling(window=i,axis=0).mean()
        data_movmean = np.array(data_moving_mean)
        outer.append(data_movmean)
        outernp = np.array(outer)[:,:,np.newaxis]  
    if i == 1:
        siat_movmean = outernp
    else:
        siat_movmean = np.append(siat_movmean,outernp,axis=2)
area_movmean_ext = siat_movmean


#reshape
#------------------
outeri = []
for i in range(10):
    inneri = []
    for j in range(nd):
        siato = area_obs[j,:,i]
        inneri.append(siato)
    outeri.append(inneri)
siat_nonstandardized = np.array(outeri)  

#sie obs [sie_nonstandardized [10,17,45]]
siat_obs = siat_nonstandardized
siat_mean = np.nanmean(siat_obs,axis=2)
siat_std = np.nanstd(siat_obs,axis=2)

#sie models [siextent [211,17,165,10]]
#------------------
model_name = ['ACCESS-CM2','ACCESS-ESM1-5','CanESM5','IPSL-CM6A-LR','MIROC6','MRI-ESM2-0','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM-GC31-MM']
ensemble_numbers = [10,40,25,7,50,5,3,4,4]
ensemble_numbers_cum = np.concatenate(([0,],np.cumsum(ensemble_numbers)),axis=0)


loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siat_1p25m_SImon_models_concat_hist_ssp585_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM_mon2sep_gn_185001-210012.nc'
dataset =  nc.Dataset(loadpath,'r')
siat_model_ssp585 = np.array(dataset.variables['area_sitthresh'])
siat_ensemble_mean_ssp585 = np.array(dataset.variables['area_ensemble_mean'])

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#(1) FORCED, RESIDUAL
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#time range 1978-2024
#---------------------------------------------------------------------
model_time = np.arange(1850,2100)
obs_time = np.arange(1979,2025)
mt1979 = np.where(model_time == 1979)[0]
mt2024 = np.where(model_time== 2024)[0]+1
mt2014 = np.where(obs_time ==2014)[0]

siat_ensemble_mean_7924 = siat_ensemble_mean_ssp585[:,:,mt1979[0,]:mt2024[0,],:]

#method: weighted ensemble mean
#relative variance = relative variance(mean(model ensembles))
#------------------------------------------------------------------------
#------------------------------------------------------------------------

#weighted by number of ensemble members
ensemble_numbers = [10,40,25,7,50,5] #,3,4,4]
ens_cum = np.cumsum(ensemble_numbers)
nf = siat_ensemble_mean_7924.shape[2]

#weighted by number of ensemble members
ensemble_numbers_rep = np.transpose(np.tile(ensemble_numbers[:6],(10,nf,17,1)))
siat_model_mean_weighted = np.nansum((siat_ensemble_mean_7924*ensemble_numbers_rep/ens_cum[5]),axis=0)


#save FIT
siatm_newj_detrended = []
siatm_newj = []
siatr_newj = []
#moving means
for j in range(10):
    siatm_newk_detrended= []
    siatm_newk = []
    siatr_newk= []
    #time frames
    for k in range(17):

        #observations
        siato = siat_obs[j,k,:]
        
        #fit
        siatm_new =  siat_model_mean_weighted[k,:,j]
        
        #demean (i.e. subtract obs temporal mean from forced)
        obs_mean = np.nanmean(siato)
        ens_mean = np.nanmean(siatm_new)
        siatm_new_demeaned = siatm_new - ens_mean + obs_mean
        
        
        #detrend (i.e. forced-LE+LO, LE = ensemble linear trend, LO = obs linear trend) 
        # Perform a linear fit on ensemble, LE = y_fit_E
        coefficientsE = np.polyfit(np.arange(1979,2025), siatm_new_demeaned, 1)
        polynomial = np.poly1d(coefficientsE)
        x_fit = np.arange(1979,2025)
        y_fit_E = polynomial(x_fit)
        
        # Perform a linear fit on obs, LO = y_fit_O
        dy = siato
        dx = np.arange(1979,2025)
        
        #remove nans
        dy = dy[~np.isnan(dx)]
        dx = dx[~np.isnan(dx)]
        dx = dx[~np.isnan(dy)]
        dy = dy[~np.isnan(dy)]
        xn = dx
        yn = dy
        
        coefficientsO = np.polyfit(xn, yn, 1)
        polynomial = np.poly1d(coefficientsO)
        x_fit = np.arange(1979,2025)
        y_fit_O = polynomial(x_fit)
        
        #forced (demeaned & detrended ensemble mean)
        siatm_new_detrended = siatm_new_demeaned-y_fit_E+y_fit_O
        
        
        #residual = observations - concatenated ensemble mean
        siatr_new = siato-siatm_new_detrended


        siatm_newk_detrended.append(siatm_new_detrended)
        siatm_newk.append(siatm_new)
        siatr_newk.append(siatr_new)
    siatm_newj_detrended.append(siatm_newk_detrended)
    siatm_newj.append(siatm_newk)
    siatr_newj.append(siatr_newk)


#[nT,tf,t]
residual_mean_weighted = np.array(siatr_newj)
fit_mean_weighted = np.array(siatm_newj_detrended)


#relative variance, T = 1 year moving mean
res1 = residual_mean_weighted[0,13,:][:,np.newaxis]
fi1 = fit_mean_weighted[0,13,:][:,np.newaxis]
prop = np.concatenate([res1,fi1],axis=1)
change_duration = np.arange(0,46)
nch = change_duration.shape[0]

outervar = []
for k in range(2):
    innervar = []        
    for i in range(nch):

        prop0 = prop[0:-(i+1),k]
        propf = prop[i+1:,k]
        propdelta = propf-prop0
        vardelta = np.nanvar(propdelta)
        innervar.append(vardelta)
        
    outervar.append(innervar)
    variance = np.array(outervar)
    relvar = np.divide(variance[0,:],np.sum(variance,axis=0))
     
relative_variancei = np.array(relvar)
nyear = np.arange(46)


#interannual variability: forced vs. residual [MEAN OVER MODELS]
#define figure properties
#fit & residual: [nm,nT,tf,t]
#------------------
ylab = 'siat' 
dateform = '%Y' # i.e, %Y 
s1 = 15 #figure size, xdim
s2 = 30 #figure size, ydim
n1 = 4 #no. subplots, xdim
n2 = 1 #no.subplots, ydim
time_sic = obs_time
tfr = 13 #13 = september 

fig, ax = plt.subplots(n1,n2, figsize=(s1,s2))

#T = 1 year MM
datai = siat_obs[0,tfr,:]
fiti = fit_mean_weighted[0,tfr,:]
resi = residual_mean_weighted[0,tfr,:]

#T = 5 year MM
datai5 = siat_obs[4,tfr,:]
fiti5 = fit_mean_weighted[4,tfr,:]
resi5 = residual_mean_weighted[4,tfr,:]

#T = 10 year MM
datai10 = siat_obs[9,tfr,:]
fiti10 = fit_mean_weighted[9,tfr,:]
resi10 = residual_mean_weighted[9,tfr,:]

relative_variance = relative_variancei

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#plot: annual, 5-year, and 10-year mean from monthly mean
#------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

ax[0].plot(time_sic,datai,linewidth=3,color='red',label='1 year')
ax[0].plot(time_sic,datai5,linewidth=3,color='purple',label='5 year')
ax[0].plot(time_sic,datai10,linewidth=3,color='blue',label='10 year')
ax[0].set_ylabel('SIAt [$\mathregular{km^2}$]',fontsize=25)
ax[0].set_xlim([time_sic[0],time_sic[-1]])
ax[0].tick_params(axis='both', labelsize=20)
ax[0].legend(fontsize=25)
ax[0].set_title('(a)',fontsize=25)
        
ax[1].plot(time_sic,fiti,linewidth=3,color='red',label='weighted cmip ensemble mean')
ax[1].plot(time_sic[mt2014[0]:],fiti[mt2014[0]:],linewidth=2,linestyle='--',color='white')
ax[1].plot(time_sic,fiti5,linewidth=3,color='purple')
ax[1].plot(time_sic[mt2014[0]:],fiti5[mt2014[0]:],linewidth=2,linestyle='--',color='white')
ax[1].plot(time_sic,fiti10,linewidth=3,color='blue')
ax[1].plot(time_sic[mt2014[0]:],fiti10[mt2014[0]:],linewidth=2,linestyle='--',color='white')
ax[1].set_ylabel('forced [$\mathregular{km^2}$]',fontsize=25)
ax[1].set_xlim([time_sic[0],time_sic[-1]])
ax[1].tick_params(axis='both', labelsize=20)
ax[1].set_title('(b)',fontsize=25)

ax[2].plot(time_sic,np.zeros(46,),color='black')
ax[2].plot(time_sic,resi,linewidth=3,color='red')
ax[2].plot(time_sic,resi5,linewidth=3,color='purple')
ax[2].plot(time_sic,resi10,linewidth=3,color='blue')
ax[2].set_ylabel('residual [$\mathregular{km^2}$]',fontsize=25)
ax[2].set_xlim([time_sic[0],time_sic[-1]])
ax[2].tick_params(axis='both', labelsize=20)
ax[2].set_title('(c)',fontsize=25)

ax[3].plot(nyear,relative_variance,linewidth=3,color='black',label='')
ax[3].fill_between(nyear[0:41],relative_variance[0:41], 1, color='salmon', alpha=0.3, where=(relative_variance[0:41] > 0), label='forced')
ax[3].fill_between(nyear[0:41],relative_variance[0:41], 0, color='skyblue', alpha=0.3, where=(relative_variance[0:41]  > 0), label='residual')
ax[3].tick_params(axis='both', labelsize=20)
ax[3].set_xlabel('change duration [years]',fontsize=25)
ax[3].legend(fontsize=25)  
ax[3].set_ylim([0,1])
ax[3].set_xlim([1,40])
ax[3].set_title('(d)',fontsize=25)
ax[3].set_ylabel('relative variance of residual',fontsize=25)


'''
#plot forced & obs on same axis
fig, ax = plt.subplots(3,1, figsize=(s1,s2))
ax[0].plot(time_sic,datai,linewidth=3,color='red',label='obs')
ax[0].plot(time_sic,fiti,linewidth=3,color='black',label='fit')
ax[0].set_ylabel('sea ice extent [$\mathregular{10^6   km^2}$]',fontsize=25)
ax[0].set_xlim([time_sic[0],time_sic[-1]])
ax[0].tick_params(axis='both', labelsize=20)
ax[0].legend(fontsize=25)
ax[0].set_title('(a)',fontsize=25)
        
ax[1].plot(time_sic,datai5,linewidth=3,color='purple',label='5 year')
ax[1].plot(time_sic,fiti5,linewidth=3,color='black')
ax[1].set_ylabel('sea ice extent [$\mathregular{10^6   km^2}$]',fontsize=25)
ax[1].set_xlim([time_sic[0],time_sic[-1]])
ax[1].tick_params(axis='both', labelsize=20)
ax[1].set_title('(b)',fontsize=25)

ax[2].plot(time_sic,datai10,linewidth=3,color='blue',label='10 year')
ax[2].plot(time_sic,fiti10,linewidth=3,color='black')
ax[2].set_ylabel('sea ice extent [$\mathregular{10^6   km^2}$]',fontsize=25)
ax[2].set_xlim([time_sic[0],time_sic[-1]])
ax[2].tick_params(axis='both', labelsize=20)
ax[2].set_title('(c)',fontsize=25)
'''

'''
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#save data
#------------------------------------------------------------------------
#------------------------------------------------------------------------

filepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siat_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
with nc.Dataset(filepath,'w') as file:
    #create dimensions
    file.createDimension('nd',siat_obs.shape[1]) #time frames, tf = 17
    file.createDimension('nT',siat_obs.shape[0]) #mov mean, T = 10
    file.createDimension('nt',siat_obs.shape[2]) #years, t = 45
    file.createDimension('mt',6) #ensemble members, m = 6
    
    
    #create variables
    siato = file.createVariable('siat_obs','f4',('nT','nd','nt'))
    fitm = file.createVariable('fit_mean_weighted','f4',('nT','nd','nt'))
    resm = file.createVariable('residual_mean_weighted','f4',('nT','nd','nt'))


    #write data to variables 
    siato[:] = siat_obs
    fitm[:] = fit_mean_weighted
    resm[:] = residual_mean_weighted


'''