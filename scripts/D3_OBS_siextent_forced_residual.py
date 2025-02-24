#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:08:57 2024

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

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#I. OBSERVATIONS
#---------------------------------------------------------------------
#---------------------------------------------------------------------
timeframe = ('y','JFM','AMJ','JAS','NOD','J','F','M','A','M','J','J','A','S','O','N','D')
#**** edit lines 113-123 &  254 when loading new dataset

#load siextent data
#------------------------------------------------------
#------------------------------------------------------
file_path = '/Users/hoffmanl/Documents/data/nsidc/siextent/Sea_Ice_Index_Monthly_Data_with_Statistics_G02135_v3.0.xlsx'
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
string_tail = '-NH'

current_date = datetime.now()
current_year = current_date.year
current_year = 2025
current_month = current_date.month
current_month = 12
day = current_date.day
yearrange = np.arange(1978,current_year)
nn = current_year-1978

siextenti = []
time = []

for i in range(12):
    sie = np.full((nn,1), np.nan)
    years = np.full((nn,1), np.nan)
    sheet = months[i]+string_tail
    data = pd.read_excel(file_path,sheet_name=sheet)
    
    timei = np.array(data[['Unnamed: 1']])[9:]
    siei = np.array(data[['Unnamed: 5']])[9:]
    
    if i < 10:  
        start = 1979
    else:
        start = 1978
    
    
    end = current_year-1
   
    startyearindex = np.where(yearrange==start)[0][0]
    endyearindex = np.where(yearrange==end)[0][0]    
    sie[startyearindex:endyearindex+1,]=siei[:,]
    years[startyearindex:endyearindex+1,]=timei[:,]
    
    #siey = sie[1:-1]
    #yeary = years[1:-1]
    
    siey = sie
    yeary = years
    
    siextenti.append(siey)
    time.append(yeary)

siextent = np.reshape(np.array(siextenti),[12,nn])
timeall = np.reshape(np.array(time),[12,nn])

yearsf = timeall.flatten('F')
monthsf = np.transpose(np.tile(np.arange(1,13),[1,nn]))
ny = yearsf.shape[0]

years =np.reshape(yearsf.astype(int),(ny,))
years[:10] = 1978
years[ny-current_month+1:] = current_year-1
months = np.reshape(monthsf.astype(int),(ny,))
time= [datetime(year, month, 1) for year, month in zip(years, months)]

sie_yearly_meani = np.nanmean(siextent,axis=0)
sie=siextent.flatten('F')
year = np.arange(1979,current_year+1)

#fill in Dec 1987 and Jan 1988 with linear interpolation using SEP87-MAR88
tint = np.arange(1,8)
sieint = sie[116:123]
dx = tint
dy = sieint
dx2 = tint
 
#remove NaNs
dy = dy[~np.isnan(dx)]
dx = dx[~np.isnan(dx)]
 
dx = dx[~np.isnan(dy)]
dy = dy[~np.isnan(dy)]

x = dx
y = dy

# Perform a quadratic fit (degree = 2)
coefficients = np.polyfit(x, y, 2)
polynomial = np.poly1d(coefficients)

# Generate x values for plotting the fitted curve
x_fit = dx2
y_fit = polynomial(x_fit)

'''
# Plotting
xti = {'SEP','OCT','NOV','DEC','JAN','FEB','MAR'}
plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_fit, y_fit, label='Quadratic Fit')
plt.scatter([4,5],y_fit[3:5],color='purple',label = 'Extrapolated Data')
plt.ylabel('sie')
plt.xticks(x_fit,xti)
plt.legend()
plt.title('Quadratic Fit to fill missing data for Dec 1987 and Jan 1988')
plt.show()
'''

#replace sie[107:108] with extrapolated values
sie[119] = y_fit[3]
sie[120] = y_fit[4]

#a. separate into desired time frames
#------------------
#yearly mean
dates = np.array(time)
years = np.array([date.year for date in dates])
unique_years = np.unique(years)
sie_yearly_mean = np.array([np.nanmean(sie[years==year],axis=0) for year in unique_years])

#months
months = np.array([date.month for date in dates])
unique_months = np.unique(months)

#monthly sie
sie_mon = []
time_mon = []
for k in range(1,13):
    siem = sie[months==k]
    timem = dates[months==k]
    sie_mon.append(siem)
    time_mon.append(timem)

sie_monthly = np.transpose(np.array(sie_mon))
time_monthly = np.array(time_mon)

#seasonal sie        
#winter (JFM)
selected_data = sie[np.isin(months,[1,2,3])]
selected_years = years[np.isin(months,[1,2,3])]
selected_time = dates[np.isin(months,[1,2,3])]
sie_yearly_JFM = np.array([np.nanmean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_JFM = dates[months==1]

#spring (AMJ)
selected_data = sie[np.isin(months,[4,5,6])]
selected_years = years[np.isin(months,[4,5,6])]
selected_time = dates[np.isin(months,[4,5,6])]
sie_yearly_AMJ = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_AMJ = dates[months==4]

#summer (JAS)
selected_data = sie[np.isin(months,[7,8,9])]
selected_years = years[np.isin(months,[7,8,9])]
selected_time = dates[np.isin(months,[7,8,9])]
sie_yearly_JAS = np.array([np.mean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_JAS = dates[months==7]

#fall (OND)
selected_data = sie[np.isin(months,[10,11,12])]
selected_years = years[np.isin(months,[10,11,12])]
selected_time = dates[np.isin(months,[10,11,12])]
sie_yearly_OND = np.array([np.nanmean(selected_data[selected_years == year],axis=0) for year in unique_years])
time_OND = dates[months==10]

sie_seasonal = np.concatenate([sie_yearly_JFM[:,np.newaxis],sie_yearly_AMJ[:,np.newaxis],sie_yearly_JAS[:,np.newaxis],sie_yearly_OND[:,np.newaxis]],axis=1) 

#b. take moving mean for T = 1:10 years
#------------------
data = np.concatenate([sie_yearly_mean[:,np.newaxis],sie_yearly_JFM[:,np.newaxis],sie_yearly_AMJ[:,np.newaxis],sie_yearly_JAS[:,np.newaxis],sie_yearly_OND[:,np.newaxis],sie_monthly],axis=1) 
nd = np.shape(data)[1]

#set NaNs for appropriate means in 1978 and 2024; 1==NaN
i98 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
i24 = [1,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1]
i24 = [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]

sieoj = []
for j in range(17):
    sieo = data[:,j]
    if i98[j] == 1:
        sieo[0] = np.nan 
    if i24[j] == 1:
        sieo[-1] = np.nan
    sieoj.append(sieo)
data = np.array(sieoj)


for i in range(1,11):
    outer = ([])   
    for j in range(nd):
        data_series = pd.Series(data[j,:]) 
        data_moving_mean = data_series.rolling(window=i,axis=0).mean()
        data_movmean = np.array(data_moving_mean)
        outer.append(data_movmean)
        outernp = np.array(outer)[:,:,np.newaxis]  
    if i == 1:
        sie_movmean = outernp
    else:
        sie_movmean = np.append(sie_movmean,outernp,axis=2)
sie_obs = sie_movmean


#moving mean for normalization case,T = 1-15 years
data = np.array(sieoj)
for i in range(1,16):
    outer = ([]) 
    for j in range(nd):
        data_series = pd.Series(data[j,:]) 
        data_moving_mean = data_series.rolling(window=i,axis=0).mean()
        data_movmean = np.array(data_moving_mean)
        outer.append(data_movmean)
        outernp = np.array(outer)[:,:,np.newaxis]  
    if i == 1:
        sie_movmean = outernp
    else:
        sie_movmean = np.append(sie_movmean,outernp,axis=2)
sie_movmean_ext = sie_movmean


#reshape
#------------------
sie_stand = []
outeri = []
for i in range(10):
    inner_sie = []
    inneri = []
    for j in range(nd):
        sieo = sie_obs[j,:,i]
        inneri.append(sieo)
    outeri.append(inneri)
sie_nonstandardized = np.array(outeri)  


#sie obs [sie_nonstandardized [10,17,45]]
sie_obs = sie_nonstandardized
sie_mean = np.nanmean(sie_obs,axis=2)
sie_std = np.nanstd(sie_obs,axis=2)


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#I. CMIP6 TRAINING
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#sie models [siextent [211,17,165,10]]
#------------------
model_name = ['ACCESS-CM2','ACCESS-ESM1-5','CanESM5','IPSL-CM6A-LR','MIROC6','MRI-ESM2-0','CESM2-WACCM','HadGEM3-GC31-LL','HadGEM-GC31-MM']
ensemble_numbers = [10,40,25,7,50,5,3,4,4]
ensemble_numbers_cum = np.concatenate(([0,],np.cumsum(ensemble_numbers)),axis=0)

loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextent_SImon_models_concat_hist_ssp585_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM_mon2sep_gn_185001-210012.nc'
dataset =  nc.Dataset(loadpath,'r')
sie_model_ssp585 = np.array(dataset.variables['siextent'])
sie_ensemble_mean_ssp585 = np.array(dataset.variables['sie_ensemble_mean'])


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#(1) FORCED, RESIDUAL
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#time range 1978-2024
#---------------------------------------------------------------------
model_time = np.arange(1850,2100)
obs_time = np.arange(1978,2025)
mt1978 = np.where(model_time == 1978)[0]
mt2024 = np.where(model_time== 2024)[0]+1
mt2014 = np.where(obs_time ==2014)[0]

sie_ensemble_mean_7824 = sie_ensemble_mean_ssp585[:,:,mt1978[0,]:mt2024[0,],:]



#method: weighted ensemble mean
#relative variance = relative variance(mean(model ensembles))
#------------------------------------------------------------------------
#------------------------------------------------------------------------

#weighted by number of ensemble members
ensemble_numbers = [10,40,25,7,50,5] #,3,4,4]
ens_cum = np.cumsum(ensemble_numbers)
nf = sie_ensemble_mean_7824.shape[2]

#weighted by number of ensemble members
ensemble_numbers_rep = np.transpose(np.tile(ensemble_numbers[:6],(10,nf,17,1)))
sie_model_mean_weighted = np.nansum((sie_ensemble_mean_7824*ensemble_numbers_rep/ens_cum[5]),axis=0)


#save FIT
siem_newj_detrended = []
siem_newj = []
sier_newj = []
#moving means
for j in range(10):
    siem_newk_detrended= []
    siem_newk = []
    sier_newk= []
    #time frames
    for k in range(17):

        #observations
        sieo = sie_obs[j,k,:]
        
        #fit
        siem_new =  sie_model_mean_weighted[k,:,j]
        
        #demean (i.e. subtract obs temporal mean from forced)
        obs_mean = np.nanmean(sieo)
        ens_mean = np.nanmean(siem_new)
        siem_new_demeaned = siem_new - ens_mean + obs_mean
        
        
        #detrend (i.e. forced-LE+LO, LE = ensemble linear trend, LO = obs linear trend) 
        # Perform a linear fit on ensemble, LE = y_fit_E
        coefficientsE = np.polyfit(np.arange(1978,2025), siem_new_demeaned, 1)
        polynomial = np.poly1d(coefficientsE)
        x_fit = np.arange(1978,2025)
        y_fit_E = polynomial(x_fit)
        
        # Perform a linear fit on obs, LO = y_fit_O
        dy = sieo
        dx = np.arange(1978,2025)
        
        #remove nans
        dy = dy[~np.isnan(dx)]
        dx = dx[~np.isnan(dx)]
        dx = dx[~np.isnan(dy)]
        dy = dy[~np.isnan(dy)]
        xn = dx
        yn = dy
        
        coefficientsO = np.polyfit(xn, yn, 1)
        polynomial = np.poly1d(coefficientsO)
        x_fit = np.arange(1978,2025)
        y_fit_O = polynomial(x_fit)
        
        #forced (demeaned & detrended ensemble mean)
        siem_new_detrended = siem_new_demeaned-y_fit_E+y_fit_O
        
        
        #residual = observations - concatenated ensemble mean
        sier_new = sieo-siem_new_detrended


        siem_newk_detrended.append(siem_new_detrended)
        siem_newk.append(siem_new)
        sier_newk.append(sier_new)
    siem_newj_detrended.append(siem_newk_detrended)
    siem_newj.append(siem_newk)
    sier_newj.append(sier_newk)


#[nT,tf,t]
residual_mean_weighted = np.array(sier_newj)
fit_mean_weighted = np.array(siem_newj_detrended)


#relative variance, T = 1 year moving mean
res1 = residual_mean_weighted[0,13,:][:,np.newaxis]
fi1 = fit_mean_weighted[0,13,:][:,np.newaxis]
prop = np.concatenate([res1,fi1],axis=1)
change_duration = np.arange(0,45)
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
nyear = np.arange(45)


#interannual variability: forced vs. residual [MEAN OVER MODELS]
#define figure properties
#fit & residual: [nm,nT,tf,t]
#------------------
ylab = 'sie' 
dateform = '%Y' # i.e, %Y 
s1 = 15 #figure size, xdim
s2 = 30 #figure size, ydim
n1 = 4 #no. subplots, xdim
n2 = 1 #no.subplots, ydim
time_sic = obs_time
tfr = 13 #13 = september 

fig, ax = plt.subplots(n1,n2, figsize=(s1,s2))

#T = 1 year MM
datai = sie_obs[0,tfr,:]
fiti = fit_mean_weighted[0,tfr,:]
resi = residual_mean_weighted[0,tfr,:]

#T = 5 year MM
datai5 = sie_obs[4,tfr,:]
fiti5 = fit_mean_weighted[4,tfr,:]
resi5 = residual_mean_weighted[4,tfr,:]

#T = 10 year MM
datai10 = sie_obs[9,tfr,:]
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
ax[0].set_ylabel('sea ice extent [$\mathregular{10^6   km^2}$]',fontsize=25)
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
ax[1].set_ylabel('forced [$\mathregular{10^6   km^2}$]',fontsize=25)
ax[1].set_xlim([time_sic[0],time_sic[-1]])
ax[1].tick_params(axis='both', labelsize=20)
ax[1].set_title('(b)',fontsize=25)

ax[2].plot(time_sic,np.zeros(47,),color='black')
ax[2].plot(time_sic,resi,linewidth=3,color='red')
ax[2].plot(time_sic,resi5,linewidth=3,color='purple')
ax[2].plot(time_sic,resi10,linewidth=3,color='blue')
ax[2].set_ylabel('residual [$\mathregular{10^6   km^2}$]',fontsize=25)
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

filepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
with nc.Dataset(filepath,'w') as file:
    #create dimensions
    file.createDimension('nd',sie_obs.shape[1]) #time frames, tf = 17
    file.createDimension('nT',sie_obs.shape[0]) #mov mean, T = 10
    file.createDimension('nt',sie_obs.shape[2]) #years, t = 45
    file.createDimension('mt',6) #ensemble members, m = 6
    
    
    #create variables
    siem = file.createVariable('sie_mean','f4',('nT','nd'))
    sies = file.createVariable('sie_std','f4',('nT','nd'))
    sieo = file.createVariable('sie_obs','f4',('nT','nd','nt'))
    fitm = file.createVariable('fit_mean_weighted','f4',('nT','nd','nt'))
    resm = file.createVariable('residual_mean_weighted','f4',('nT','nd','nt'))


    #write data to variables 
    siem[:] = sie_mean
    sies[:] = sie_std
    sieo[:] = sie_obs
    fitm[:] = fit_mean_weighted
    resm[:] = residual_mean_weighted


'''