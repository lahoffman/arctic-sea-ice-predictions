#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:26:59 2024

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec

#colorbars
import cmocean 

#import functions
#------------------
sys.path.append('/Users/hoffmanl/Documents/scripts/functions/')
from functions_general import ncdisp
from functions_general import movmean
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
#least squares / linear trend
#------------------------------------------------------
#------------------------------------------------------
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
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
#LINEAR TREND IN OBSERVATIONS
#------------------------------------------------------
#------------------------------------------------------

#load siextent data, raw observations
#------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202312_vX.nc'
dataset =  nc.Dataset(load_path,'r')
sie_observed = np.array(dataset.variables['sie_obs'])
unique_years = np.arange(1978,2025)


#least squares fit, time vs. raw sie --> linear trend
#end fit year index ; TO only trained through 2014, so only do linear fit through 2014
trend_end = 2014
ntt = np.array(np.where(unique_years==trend_end))[0,0]
fit = []
fit22 = []
residual = []
error = []

for i in range(10):
    fit_in = []
    fit2_in = []
    residual_in = []
    error_in = []
    for j in range(17):
        #define inputs and outputs
        dx = unique_years[0:ntt]
        dx2 = np.arange(1979,2034)[:,np.newaxis]
        dy = sie_observed[i,j,0:ntt]
        
        #standardize: zero mean, unit standard deviation
        #dy = np.divide((dyi-np.nanmean(dyi)),np.nanstd(dyi))
        
        #remove NaNs
        dy = dy[~np.isnan(dx)]
        dx = dx[~np.isnan(dx)]
        
        dx = dx[~np.isnan(dy)]
        dy = dy[~np.isnan(dy)]
        
        dx = dx[:, np.newaxis]
        dy = dy[:, np.newaxis]
        
        nt = dx.shape[0]
        nt2 = dx2.shape[0]
        
        #define kernal, ouput
        E = np.concatenate([np.ones([nt,1]),dx],axis=1)
        E2 = np.concatenate([np.ones([nt2,1]),dx2],axis=1)
        Y = dy
        
        #run model
        m = least_squares(E,Y)
        
        #fit data to model
        fiti = np.dot(E,m)
        fit2 = np.dot(E2,m)
  
        #residual
        residuali = Y-fiti
    
        #error
        errori = np.nanstd(residuali)
        
        
        uf = unique_years[0:ntt].shape[0]
        sf = fiti.shape[0]
        nf = uf-sf      
        fit_app = np.full([uf,1], np.nan)
        res_app = np.full([uf,1], np.nan)
        
        uf2 = dx2.shape[0]
        sf2 = fit2.shape[0]
        nf2 = uf2-sf2
        fit2_app = np.full([uf2,1],np.nan)
        
        #for nans in movmean datasets
        if sf < uf:
            fit_app[nf:,] = fiti
            res_app[nf:,] = residuali
        else:
            fit_app = fiti
            res_app = residuali
            
        if sf2 < uf2:
            fit2_app[nf2:,] = fit2
        else:
            fit2_app = fit2
            
            
        fit_in.append(fit_app)
        residual_in.append(res_app)
        error_in.append(errori)
        fit2_in.append(fit2_app)
        
    residual.append(residual_in)
    fit.append(fit_in)
    error.append(error_in)
    fit22.append(fit2_in)
        

residual1 = np.array(residual)
fit1 = np.array(fit)
error1 = np.array(error)
fit2np = np.array(fit22)

#fit for 1979-2014
residual_linear_fit = np.reshape(residual1,[10,17,uf])
linear_fit = np.reshape(fit1,[10,17,uf])
error = np.tile(error1[:,:,np.newaxis],[1,1,uf2])
#error_ext = np.tile(error1[:,:,np.newaxis],[1,1,53])

#extended fit for 1979-2034
timeextend = np.arange(1979,2034)
linear_fit_ext = np.reshape(fit22,[10,17,uf2])
linear_fitp1std = linear_fit_ext+error
linear_fitp2std = linear_fit_ext+2*error
linear_fitm1std = linear_fit_ext-error
linear_fitm2std = linear_fit_ext-2*error
time = unique_years
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
#STANDARDIZATION
#------------------------------------------------------
#------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
transfer_operator = dataset.variables['transfer_operator']
transfer_bins = dataset.variables['transfer_bins']
tb = np.array(transfer_bins)
miu_train = dataset.variables['miu_tr']
sigma_train = dataset.variables['sigma_tr']


#------------------------------------------------------
#------------------------------------------------------
#LOAD TEST DATA: OBSERVATIONS
#------------------------------------------------------
#------------------------------------------------------

#residual from linear trend
load_path = '/Users/hoffmanl/Documents/data/transfer_operator/methodB3_sie/D_siextentn_test_data_transfer_operator_obs_residual_fit_ext_197901202312_vX.nc'
dataset =  nc.Dataset(load_path,'r')
fitext_obs = np.array(dataset.variables['fitext']) #linear only

#[MMM, demeaned, detrended] residual from cmip6 historical (1979-2014) + residual cmip6 ssp585 (2015-2024)
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
sie_observed = np.array(dataset.variables['sie_obs'])
residual_obs = np.array(dataset.variables['residual_mean_weighted']) #[10,17,45]
fit_obs = np.array(dataset.variables['fit_mean_weighted']) #[10,17,45]

    
#set residual to one unit standard deviation
outer = []
outeri = []
outerstd = []
outermean = []
for i in range(17):
    inner = []
    inneri = []
    innerstd = []
    innermean = []
    for j in range(10):
        te = residual_obs[j,i,:] 
        test = sigma_train[0,i,0,j]
        tem = miu_train[0,i,0,j]
        ted = np.divide((te-tem),test)
        
        tei = residual_obs[j,i,:]       
        inner.append(ted) #standardized
        inneri.append(tei) #non-standardized
        innerstd.append(test)
        innermean.append(tem)
    outer.append(inner)
    outeri.append(inneri)
    outerstd.append(innerstd)
    outermean.append(innermean)
residual_standardized = np.array(outer)
residual_std = np.array(outerstd)
residual_mean = np.array(outermean)
sie_mean = residual_mean.T
sie_std = residual_std.T

test_residual = residual_standardized[:,np.newaxis,:,:] #2002-2020
test_original = sie_observed[:,np.newaxis,:,:]

#reshape
outer = []
outerog = []
for i in range(17):
    inner = []
    innerog = []
    for j in range(46):
        te = residual_standardized[i,:,j+1]
        tog = sie_observed[:,i,j+1]
        
        inner.append(te)
        innerog.append(tog)
    outer.append(inner)
    outerog.append(innerog)
sie_obs = np.array(outer)    
sie_original = np.array(outerog)

sie_monthly = sie_original[5:,:,:]
sie_seasonal = sie_original[1:5,:,:]
sie_yearly = sie_original[0,:,:]
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
#LOAD TRAINING DATA: CMIP6 
#------------------------------------------------------
#------------------------------------------------------
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset_tr =  nc.Dataset(loadpath,'r')
siextent_model = np.array(dataset_tr.variables['siextent'])

load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
dataset_tr =  nc.Dataset(load_path,'r')
sie_training = np.array(dataset_tr.variables['train_standardized'])
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# LOAD 2DNN PREDICTIONS
#------------------------------------------------------
#------------------------------------------------------
loadpath_nn2d = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_2D_siearea2sie_bins_performance.nc'
dataset =  nc.Dataset(loadpath_nn2d,'r')
predictionl = dataset.variables['nn_prediction_mean_obs']
stdevl = dataset.variables['nn_prediction_err_obs']
sie_obs_nn = dataset.variables['nn_input_obs']
#------------------------------------------------------
#------------------------------------------------------



#------------------------------------------------------
#------------------------------------------------------
#DEFINE PREDICTION TIMES
#------------------------------------------------------
#------------------------------------------------------
years = []
for i in range(10):
    yearsi = np.arange(1979+i+1,2024+i+1)
    years.append(yearsi)
    
yearspred = np.array(years)

tf = [0,0,0,0,0,0,0,0,1,1,1,1]

yp = []
yi = []
for i in range(12):
    if tf[i] == 1:
        yearpred = yearspred
        yearinput = yearspred-1

    else:
        yearpred = yearspred
        yearinput = yearspred
        
    yp.append(yearpred)
    yi.append(yearinput)
yearpred = np.array(yp)[:,:,:]
yearinput = np.array(yi)[:,:,:]
years_fit = np.arange(1978,2025)
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# PREDICTION TIME SERIES
#------------------------------------------------------
#------------------------------------------------------
def find_all_indices(array, value):
    indices = [i for i, x in enumerate(array) if x == value]
    return indices

#unstandardize prediction
#----------------------------------
prediction_residual = np.array(predictionl)
predi = []
#time frames
for i in range(5,17):
    predj = []
    #moving mean
    for j in range(10):
        predh = []
        #lag time
        for h in range(10):
            siem = residual_mean[i,j]
            sies = residual_std[i,j]
            yearspred = yearpred[i-5,h,:]
            npp = np.array(yearspred.shape[0])
            predk = []
            for k in range(npp):
                yf = find_all_indices(years_fit,yearspred[k])
                if yf != []:
                    fit = np.reshape(fit_obs[j,13,yf],[1,])
                else:
                    fit = np.full((1,), np.nan)
                predo = np.multiply(prediction_residual[i-5,j,h,k],sies)+siem+fit 
                predon = np.array(predo)
                predk.append(predon)
            predh.append(predk)
        predj.append(predh)
    predi.append(predj)

sie_pred = np.array(predi)[:,:,:,:,0]
#pred_unstandardized = pred           

#one year pred, one year moving mean, #prediction of september from each time frame and year
prediction_residual = np.array(predictionl)
sie_pred_residual = np.array(predictionl)[:,0,0,:] 


#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------ 
#------------------------------------------------------ 
# SAVE: PREDICTION & NSIDC, 1989-2024
#------------------------------------------------------ 
#------------------------------------------------------ 
sie_prediction_residual = prediction_residual[:,0,0,:]
sie_prediction = sie_pred[:,0,0,:]
sie_sep_residual_nsidc = test_residual[13,0,0,2:]
sie_sep_nsidc = sie_monthly[8,1:,0]

savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/prediction_full_detrend_2DNN.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',sie_prediction_residual.shape[0]) #no. time frames
    file.createDimension('nt',sie_prediction_residual.shape[1]) #years
  

    #create variables
    #NN performance
    spr = file.createVariable('sie_prediction_residual','f4',('tf','nt')) 
    sp = file.createVariable('sie_prediction','f4',('tf','nt'))
    ssrn = file.createVariable('sie_sep_residual_nsidc','f4',('nt'))  
    ssn = file.createVariable('sie_sep_nsidc','f4',('nt'))

    #write data to variables
    spr[:] = sie_prediction_residual
    sp[:] = sie_prediction
    ssrn[:] = sie_sep_residual_nsidc
    ssn[:] = sie_sep_nsidc


#------------------------------------------------------ 
#------------------------------------------------------ 

'''
#------------------------------------------------------
#------------------------------------------------------
#EXTREME EVENT PREDICTION
#PLOT: SIC prediction in YEAR for different hindcast times
# 1-YEAR MOVING MEAN
#------------------------------------------------------
#------------------------------------------------------ 
# Create a figure and gridspec layout
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(48, 30))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
kl = np.concatenate((np.arange(0,2)[:,np.newaxis],np.arange(2,4)[:,np.newaxis],np.arange(4,6)[:,np.newaxis]),axis=1)

dataextremej = []
climmeanj = []
climstdp1extremej = []
climstdp2extremej = []
climstdm1extremej = []
climstdm2extremej = []
predj = []
errorj = []
for j in range(2):
    
    ps = 13 #september, predicted time frame
    yearsinterest = [2012,2020]
    #yearsinterest = [2007,2019]
    #---------------------------------
    #*********************************
    #TOGGLE TO DESIRED INTEREST
    pp = 6 #starting time frame
    year_of_interest = yearsinterest[j]
    #*********************************
    #---------------------------------
    
    titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    title_letters = ['(a)','(d)','(b)','(e)','(c)','(f)']
    labelk  = ['1-year mean','2-year mean','5-year mean']
    prediction_array = np.array(predictionl)
    error_array = np.array(stdevl)
    #median_array = np.array(medianl)
    #cilow_array = np.array(cilowl)
    #cihigh_array= np.array(cihighl)
    text_large = 50
    text_small = 30
    
    #index of prediction year
    inputyears = np.arange(1980,2025)
    predyears = np.arange(1980,2025)
    fityears = np.arange(1980,2025)
    linfityears = np.arange(1979,2034)
    timedata = np.array(np.where(inputyears==year_of_interest))[0,0]
    timefitX = np.array(np.where(fityears==year_of_interest))[0,0]
    timelinfitX = np.array(np.where(linfityears==year_of_interest))[0,0]

    #index of prediction year for time before now; index of previous year for time now (i.e. unknown 'data')
    if(timefitX<46):
        timefit = timefitX 
        timelinfit = timelinfitX
    else:
        timefit = 45
        timelinfit = 45
    
    #lagtime based on starting timeframe
    if tf[pp] == 0:
        lagtime = np.arange(0,10)
    else:
        lagtime = np.arange(1,11)
 
        
    initialize = timefitX-lagtime
    unique_years_extend = np.arange(1978,2033)

    predki = []
    errki = []
    #medki = []
    #cilki = []
    #cihki = []
    
    dataextremek = []
    climmeank = []
    climstdp1extremek = []
    climstdp2extremek = []
    climstdm1extremek = []
    climstdm2extremek = []
    for k in range(3):
        predi = []
        erri = []
        #medi = []
        #cili = []
        #cihi = []
        
        dataextremei = []
        climmeani = []
        climstdp1extremei = []
        climstdp2extremei = []
        climstdm1extremei = []
        climstdm2extremei = []
        for i in range(10):
            
            #title
            titles = f'{title_letters[kl[j,k]]} {yearsinterest[j]}, {labelk[k]}' 
            #prediction time index
            yp = yearpred[pp,i,:]
            timepred = np.array(np.where(yp==year_of_interest))[0,0]
        
            mov_mean = [0,1,4] #index of n+1 - year average
            mov_meani = mov_mean[k]
            mov_mean_label = mov_meani+1
        
            fitk = fit_obs[mov_meani,ps,2:] 
            data = sie_original[ps,1:,mov_meani]
            prevsep= np.round(data[timefit],decimals=2)
            predk = (prediction_array[pp,mov_meani,i,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timefit] #predk = prediction based on TO applied to residual s from obs    
            errork = error_array[pp,mov_meani,i,timepred]*residual_std[ps,mov_meani]
            #medk = (median_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timeX2] #predk = prediction based on TO applied to residual s from obs 
            #cilk = (cilow_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timeX2] #predk = prediction based on TO applied to residual s from obs 
            #cihk = (cihigh_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timeX2] #predk = prediction based on TO applied to residual s from obs 

            #linear fit statistics 
            clim_mean = linear_fit_ext[mov_meani,ps,timelinfitX]
            clim_mean_all = clim_mean
            clim_stdp1_all = linear_fitp1std[mov_meani,ps,timelinfitX]
            clim_stdp2_all = linear_fitp2std[mov_meani,ps,timelinfitX]
            clim_stdm1_all = linear_fitm1std[mov_meani,ps,timelinfitX]
            clim_stdm2_all = linear_fitm2std[mov_meani,ps,timelinfitX]
            axtick = np.array([clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all])
            #axtick2 = np.array([clim_stdm3_all,clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all,clim_stdp3_all])


            ax = axes[kl[j,k]] 
            
            #if statement so legend only on first set
            if i == 0:
                line = ax.plot(np.arange(-1,12),data[timedata]*np.ones(13),linewidth=5,color='blue',label='observations')
                line2 = ax.plot(np.arange(-1,12),clim_mean*np.ones(13),linewidth=5,color='gray',label='linear trend')
                line3 = ax.plot(np.arange(-1,12),clim_mean*np.ones(13),linewidth=3,color='gray',label='linear standard deviation')
                errorbars = ax.errorbar(lagtime[i],predk,errork,color='red',fmt='o',markersize=30,capsize=10,label='prediction')
                ax.plot(np.arange(-1,12),clim_mean*np.ones([13,]),linewidth=1,color='lightgray',zorder=1,label='_nolegend_')
                #ax.text(0.55,0.01,f'MEAN: {np.round(predk,decimals=2)} \nSTDEV: {np.round(errork,decimals=2)} \nMEDIAN: {np.round(medk,decimals=2)}  \nLOW ERR BOUND: {np.round(cilk,decimals=2)}  \nHIGH ERR BOUND: {np.round(cihk,decimals=2)} \nLINEAR: {np.round(clim_mean,decimals=2)} \nSEP 2023: {4.37} ' ,fontsize=30,ha='left', va='bottom', transform=ax.transAxes)
               
            else:   
                line = ax.plot(np.arange(-1,12),data[timedata]*np.ones(13),linewidth=5,color='blue',label='_nolegend_')
                line2 = ax.plot(np.arange(-1,12),clim_mean*np.ones(13),linewidth=5,color='gray',label='_nolegend_')
                errorbars = ax.errorbar(lagtime[i],predk,errork,color='red',fmt='o',markersize=30,capsize=10,label='_nolegend_')
                ax.plot(np.arange(-1,12),clim_mean*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
                
            ax.plot(np.arange(-1,12),clim_stdp1_all*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
            ax.plot(np.arange(-1,12),clim_stdp2_all*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
            ax.plot(np.arange(-1,12),clim_stdm1_all*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
            ax.plot(np.arange(-1,12),clim_stdm2_all*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
            #ax.set_title(f'Hindcast for {mov_mean_label}-year average', fontsize=20)
            ax.grid(axis='x')
            ax.tick_params(axis='both', labelsize=text_small)
            ax.set_xlim([0,11])
            ax.set_xticks(np.arange(0,11))
            ax.set_title(titles,fontsize=text_large)
            #ax.set_ylim([4,7])
            
            dataextremei.append(data[timefit])
            predi.append(predk)
            erri.append(errork)
            climmeani.append(clim_mean)
            climstdp1extremei.append(clim_stdp1_all)
            climstdp2extremei.append(clim_stdp2_all)
            climstdm1extremei.append(clim_stdm1_all)
            climstdm2extremei.append(clim_stdm2_all)
        dataextremek.append(dataextremei)
        predki.append(predi)
        errki.append(erri)
        climmeank.append(climmeani)
        climstdp1extremek.append(climstdp1extremei)
        climstdp2extremek.append(climstdp2extremei)
        climstdm1extremek.append(climstdm1extremei)
        climstdm2extremek.append(climstdm2extremei)
    dataextremej.append(dataextremek)
    predj.append(predki)
    errorj.append(errki)
    climmeanj.append(climmeank)
    climstdp1extremej.append(climstdp1extremek)
    climstdp2extremej.append(climstdp2extremek)
    climstdm1extremej.append(climstdm1extremek)
    climstdm2extremej.append(climstdm2extremek)
         
    ax3.set_ylabel('sea ice extent [$\mathregular{10^6   km^2}$]', fontsize=text_large)
    fig.text(0.4, 0.08,'hindcast lag, N [years]', fontsize=text_large)
    #ax1.legend(fontsize=30) #,loc='lower left')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,fontsize=text_large)
    fig.subplots_adjust(right=0.85)  # Adjust the right margin to make room for the colorbar
    #fig.suptitle(f'PREDICTION OF SEPTEMBER {year_of_interest} SIE, INITIALIZED {titles[pp]} {year_of_interest}-N YEARS', fontsize=30)
#------------------------------------------------------
#------------------------------------------------------ 


#------------------------------------------------------ 
#------------------------------------------------------ 
# SAVE: EXTREME EVENT PREDICTION, 2DNN
#------------------------------------------------------ 
#------------------------------------------------------ 
datann = np.array(dataextremej)
prednn = np.array(predj)
errnn= np.array(errorj)
climmeannn = np.array(climmeanj)
climstdp1nn = np.array(climstdp1extremej)
climstdp2nn = np.array(climstdp2extremej)
climstdm1nn = np.array(climstdm1extremej)
climstdm2nn = np.array(climstdm2extremej)

savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_2DNN_julyinit_2012_2020.nc'
#savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_2DNN_julyinit_2007_2019.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',datann.shape[0]) #no. time frames
    file.createDimension('ntau',datann.shape[2]) #hindcast lag
    file.createDimension('nT',datann.shape[1]) #no. moving means
  

    #create variables
    #NN performance
    dto = file.createVariable('datann','f4',('tf','nT','ntau')) 
    pto = file.createVariable('prednn','f4',('tf','nT','ntau'))
    eto = file.createVariable('errnn','f4',('tf','nT','ntau'))  
    cmto = file.createVariable('climmeannn','f4',('tf','nT','ntau'))
    csp1to = file.createVariable('climstdp1nn','f4',('tf','nT','ntau'))
    csp2to = file.createVariable('climstdp2nn','f4',('tf','nT','ntau'))
    csm1to = file.createVariable('climstdm1nn','f4',('tf','nT','ntau'))
    csm2to = file.createVariable('climstdm2nn','f4',('tf','nT','ntau'))
    
    #write data to variables
    dto[:] = datann
    pto[:] = prednn
    eto[:] = errnn
    cmto[:] = climmeannn
    csp1to[:] = climstdp1nn
    csp2to[:] = climstdp2nn
    csm1to[:] = climstdm1nn
    csm2to[:] = climstdm2nn

#------------------------------------------------------ 
#------------------------------------------------------ 
'''



'''
#------------------------------------------------------
#------------------------------------------------------ 
#PLOT: HINDCAST TIME SERIES FOR RILES
# prediction must be initialized before 2014
#***NOTE: first point on this plot is the same as the first on the next plot
#------------------------------------------------------
#------------------------------------------------------ 
from matplotlib.cm import ScalarMappable

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(60,40))
label_large = 50
label_small = 30

# Create a figure and gridspec layout
for j in range(2):   
    
    ps = 13 #september, predicted time frame
    
    #---------------------------------
    #*********************************
    #TOGGLE TO DESIRED INTEREST
    pp = 6 #starting time frame
    year_of_interest = 2011
    #*********************************
    #---------------------------------
 
    
    
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    kl = np.concatenate((np.arange(0,2)[:,np.newaxis],np.arange(2,4)[:,np.newaxis],np.arange(4,6)[:,np.newaxis]),axis=1)
    titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    title_letter = ['(a)','(d)','(b)','(e)','(c)','(f)']
    time = np.arange(1980,2025)
    prediction_residual = np.array(predictionl)
    error_residual = np.array(stdevl)
    

    timek = []
    timeextendk = []
    timeinputk = []
    timedurpredk = []
    datakk = []
    data_residualk = []
    data_residual_predtimek = []
    prediction_residualk = []
    error_residualk = []
    data_residual_initk = []
    fitkok = []
    linfitkk = []
    linfitkp1k = []
    linfitkp2k = []
    linfitkm1k = []
    linfitkm2k = []
    predkk = []
    errorkk = []
    data_initk = []
    for k in range(3):
        
        timei = []
        timeextendi = []
        timeinputi = []
        timedurpredi = []
        dataki = []
        data_residuali = []
        data_residual_predtimei = []
        prediction_residuali = []
        error_residuali = []
        data_residual_initi = []
        fitkoi = []
        linfitki = []
        linfitkp1i = []
        linfitkp2i = []
        linfitkm1i = []
        linfitkm2i = []
        predki = []
        errorki = []
        data_initi = []
        for i in range(10):
        
            
            mov_mean = [0,1,4] #index of n+1 - year average
            mov_meani = mov_mean[k]
            mov_mean_label = mov_meani+1
            ii = 0
            
            #prediction time index
            yp = yearpred[pp,0,:]
            timepred = np.array(np.where(yp==year_of_interest))[0,0]
            if timepred+11<46:
                timeindexpred = yp[timepred:timepred+11]
            else:
                timeindexpred = timeextend[timepred:timepred+11]-1
       
            #fit time indices
            yearfit = np.arange(1980,2025)
            yi = yearfit
            timefitstart = np.array(np.where(yi==year_of_interest))[0,0]
            timefit = np.arange(timefitstart,timefitstart+10)
             
            #fit
            fitobs = fit_obs[mov_meani,ps,2:] #fit from cmip6
            fitlin = linear_fit_ext[mov_meani,ps,:] #extended fit from linear, non-standardized
            
            #use fit to obs if we have the data, otherwise use the extended linear fit
            if timefit[i] < 46:
                fitik = fitobs
            else: 
                fitik = fitlin
                
            #data residual time indices
            yeardata = np.arange(1980,2025)
            yd = yeardata
            timedatastart = np.array(np.where(yd==year_of_interest))[0,0]
            timedataresidual = np.arange(timedatastart,timedatastart+10)
            
            #data residual init time indices
            yeardata = yearinput[pp,0,:]
            yd = yeardata
            timedatastart = np.array(np.where(yd==year_of_interest))[0,0]
            timedataresidualinit = np.arange(timedatastart,timedatastart+10)
            
            #data time indices
            yeardata = np.arange(1979,2025)
            ydo = yeardata
            timedataostart = np.array(np.where(ydo==year_of_interest))[0,0]
            timedatai = np.arange(timedataostart,timedataostart+10)
            timedata = np.arange(1979,2025)
            
            #sie: observed, predicted
            data = sie_original[ps,:,mov_meani]
            data_init = sie_original[pp+5,:,mov_meani]
            pred_unstandardized = (prediction_residual[pp,mov_meani,i,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitik[timefit[i]]     
            err_unstandardized = error_residual[pp,mov_meani,i,timepred]*residual_std[ps,mov_meani] #+residual_mean[ps,mov_meani] #error on predictions made by TO
            
            #linear fit
            linfitk = linear_fit_ext[mov_meani,ps,:] #linear fit to obs (residual + forced)
            linfitkp1 = linear_fitp1std[mov_meani,ps,:] #+1*sigma linear fit to obs
            linfitkp2 = linear_fitp2std[mov_meani,ps,:] #+2*sigma linear fit to obs
            linfitkm1 = linear_fitm1std[mov_meani,ps,:] #-1*sigma linear fit to obs
            linfitkm2 = linear_fitm2std[mov_meani,ps,:] #-2*sigma linear fit to obs
            
            #residual
            data_residual = test_residual[ps,0,mov_meani,2:]
            data_residual_init = test_residual[pp+5,0,mov_meani,2:]
            pred_residual = prediction_residual[pp,mov_meani,i,timepred]
            err_residual = error_residual[pp,mov_meani,i,timepred]
            
            #figure properties
            tlet = title_letter[kl[j,k]]
            titles = f' {tlet} Hindcast for {mov_mean_label}-year average'
            
            if j == 0:
                #figure titles and axes labels        
                ax = axes[kl[j,k]] 

                #plot residual, obs
                ax.plot(time,data_residual,linewidth=5,color='black',label='_nolegend_')               
               
                #plot residual, obs for prediction time
                ax.plot(time[timedataresidual[0]:timedataresidual[0]+10],data_residual[timedataresidual[0]:timedataresidual[0]+10,],linewidth=6,color='blue',label='_nolegend_')         
                
                #plot linear fit
                ax.plot(timeextend,np.zeros([55,1]),linewidth=5,color='grey',label='_nolegend_')
               
                #plot prediction
                ax.errorbar(timeindexpred[i],pred_residual,err_residual,color='red',fmt='o',markersize=30,capsize=10,label='_nolegend_')
                ax.set_xlim([time[1],2024])
                   
                #plot input sie
                ax.scatter(yearinput[pp,0,timedataresidual[0]],data_residual_init[timedataresidual[0],],marker='s',s=1000,color='blue',label='_nolegend_')
               
                #plot linear fit   
                ax.plot(timeextend,np.ones([55,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,-1*np.ones([55,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,2*np.ones([55,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,-2*np.ones([55,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,-3*np.ones([55,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.set_xlim([time[1],2024])
               
                if i == 0:
                    ax.set_xlim([time[1],2024])
                    
                ax.set_title(titles, fontsize=label_large)
                ax.grid(axis='x')
                ax.tick_params(axis='both', labelsize=label_small)
                ax.set_xlim([time[1],2024])
                #ax1.legend(fontsize=label_large)
                
            elif j == 1:
                #figure titles and axes labels        
                ax = axes[kl[j,k]] 
                
                #if i for legend on/off
                if i == 0:
                    #plot forced (i.e. fit)
                    ax.plot(time[:],fitobs,linewidth=5,color='orchid',label='forced, obs')
            
                    #plot forced + residual, obs
                    ax.plot(timedata,data,linewidth=5,color='black',label='forced+residual, obs')          
                    
                    #plot forced + residual, obs for prediction time
                    ax.plot(timedata[timedatai[0]:timedatai[0]+10],data[timedatai[0]:timedatai[0]+10,],linewidth=6,color='blue',label='_nolegend_')               
                    
                    #plot linear fit
                    ax.plot(timeextend,linfitk,linewidth=5,color='grey',label='linear fit, obs')
                    
                    #plot prediction
                    ax.errorbar(timeindexpred[i],pred_unstandardized,err_unstandardized,color='red',fmt='o',markersize=25,capsize=10,label='prediction, TO')
                                    
                    #plot input sie
                    ax.scatter(time[timedatai[0]],data_init[timedatai[0],]+10,marker='s',s=1000,color='blue',label='model input')
                    ax.set_ylim([2,9])
                
                else:
                    #plot fit
                    ax.plot(timeextend[:-1],fitlin[:-1],linewidth=5,color='orchid',label='_nolegend_')
                    
                    #plot forced + residual, obs
                    ax.plot(timedata,data,linewidth=5,color='black',label='_nolegend_')               
                   
                    #plot forced + residual, obs for prediction time
                    ax.plot(timedata[timedatai[0]:timedatai[0]+10],data[timedatai[0]:timedatai[0]+10,],linewidth=6,color='blue',label='_nolegend_')         
                    
                    #plot linear fit
                    ax.plot(timeextend,linfitk,linewidth=5,color='grey',label='_nolegend_')
                   
                    #plot prediction
                    ax.errorbar(timeindexpred[i],pred_unstandardized,err_unstandardized,color='red',fmt='o',markersize=25,capsize=10,label='_nolegend_')
                    ax.set_xlim([time[1],2024])
                    ax.set_ylim([2,9])
                   
                #plot linear fit   
                ax.plot(timeextend,linfitkp1,linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,linfitkp2,linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,linfitkm1,linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,linfitkm2,linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.set_xlim([time[1],2024])
               
                if i == 0:
                    ax.set_xlim([time[1],2024])
                    
                ax.set_title(titles, fontsize=label_large)
                ax.grid(axis='x')
                ax.tick_params(axis='both', labelsize=label_small)
                ax.set_xlim([time[1],2024])
            


            ax3.set_xlim([time[1],2024])
            ax3.set_ylabel('sea ice extent [$\mathregular{10^6 km^2}$]', fontsize=label_large)
            #ax2.legend(fontsize=label_large)
            fig.subplots_adjust(top=0.85,right=0.9,hspace=0.2,wspace=0.1)  # Adjust the right margin to make room for the colorbar
            handles, labels = ax2.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=5,fontsize=label_large)
            
            
            timei.append(time)
            timeextendi.append(timeextend)
            timeinputi.append(time[timedataresidual[0]:timedataresidual[0]+10])
            timedurpredi.append(timeindexpred[i])
            data_residuali.append(data_residual)
            data_residual_predtimei.append(data_residual[timedataresidual[0]:timedataresidual[0]+10,])
            prediction_residuali.append(pred_residual)
            error_residuali.append(err_residual)
            data_residual_initi.append(data_residual_init[timedataresidual[0],])
            fitkoi.append(fitobs)
            linfitki.append(linfitk)
            linfitkp1i.append(linfitkp1)
            linfitkp2i.append(linfitkp2)
            linfitkm1i.append(linfitkm1)
            linfitkm2i.append(linfitkm2)
            predki.append(pred_unstandardized)
            errorki.append(err_unstandardized)
            dataki.append(data)
        timek.append(timei)
        timeextendk.append(timeextendi)
        timeinputk.append(timeinputi)
        timedurpredk.append(timedurpredi)
        data_residualk.append(data_residuali)
        data_residual_predtimek.append(data_residual_predtimei)
        prediction_residualk.append(prediction_residuali)
        error_residualk.append(error_residuali)
        data_residual_initk.append(data_residual_initi)
        fitkok.append(fitkoi)
        linfitkk.append(linfitki)
        linfitkp1k.append(linfitkp1i)
        linfitkp2k.append(linfitkp2i)
        linfitkm1k.append(linfitkm1i)
        linfitkm2k.append(linfitkm2i)
        predkk.append(predki)
        errorkk.append(errorki)
        datakk.append(dataki)
#------------------------------------------------------
#------------------------------------------------------ 


#------------------------------------------------------ 
#------------------------------------------------------ 
# SAVE: RILES HINDCAST PREDICTION, 2DNN
#------------------------------------------------------ 
#------------------------------------------------------ 
timenn = np.array(timek)
timeextendnn = np.array(timeextendk)
timeinputnn = np.array(timeinputk)
timedurprednn = np.array(timedurpredk)
data_residualnn = np.array(data_residualk)
data_residual_predtimenn = np.array(data_residual_predtimek)
prediction_residualnn = np.array(prediction_residualk)
error_residualnn = np.array(error_residualk)
data_residual_initnn = np.array(data_residual_initk)
fitnn = np.array(fitkok)
linfitnn = np.array(linfitkk)
linfitp1nn = np.array(linfitkp1k)
linfitp2nn = np.array(linfitkp2k)
linfitm1nn = np.array(linfitkm1k)
linfitm2nn = np.array(linfitkm2k)
predknn = np.array(predkk)
errorknn = np.array(errorkk)
dataknn = np.array(datakk)

frames = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
yearstr = str(year_of_interest)
savelead = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_2DNN_'
savetail = '.nc'
savepath = savelead+frames[pp]+yearstr+savetail
 
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('T',timenn.shape[0]) #no. moving means
    file.createDimension('ntau',timenn.shape[1]) #hindcast lag
    file.createDimension('nt',timenn.shape[2]) #no. timesteps
    file.createDimension('nt1',fitnn.shape[2]) #no. timesteps
    file.createDimension('nte',linfitnn.shape[2]) #no. timesteps
    
    #create variables
    #NN performance
    tnn = file.createVariable('timenn','f4',('T','ntau','nt')) 
    tenn = file.createVariable('timeextendnn','f4',('T','ntau','nte')) 
    tinn = file.createVariable('timeinputnn','f4',('T','ntau','ntau')) 
    tdnn = file.createVariable('timedurprednn','f4',('T','ntau')) 
    
    dnn = file.createVariable('dataknn','f4',('T','ntau','nt'))
    drnn = file.createVariable('data_residualnn','f4',('T','ntau','nt'))
    drptnn = file.createVariable('data_residual_predtimenn','f4',('T','ntau','ntau'))
    drinn = file.createVariable('data_residual_initnn','f4',('T','ntau'))
    prnn = file.createVariable('prediction_residualnn','f4',('T','ntau'))
    ernn = file.createVariable('error_residualnn','f4',('T','ntau'))  
    pnn = file.createVariable('predknn','f4',('T','ntau'))
    enn = file.createVariable('errorknn','f4',('T','ntau'))  
    fnn = file.createVariable('fitnn','f4',('T','ntau','nt1'))  
    
    lnn = file.createVariable('linfitnn','f4',('T','ntau','nte'))
    lp1nn = file.createVariable('linfitp1nn','f4',('T','ntau','nte'))
    lp2nn = file.createVariable('linfitp2nn','f4',('T','ntau','nte'))
    lm1nn = file.createVariable('linfitm1nn','f4',('T','ntau','nte'))
    lm2nn = file.createVariable('linfitm2nn','f4',('T','ntau','nte'))
    
    #write data to variables
    tnn[:]  = timenn 
    tenn[:]  = timeextendnn
    tinn[:]  = timeinputnn 
    tdnn[:]  = timedurprednn
    
    drnn[:]  = data_residualnn
    drptnn[:]  = data_residual_predtimenn
    drinn[:]  = data_residual_initnn
    prnn[:]  = prediction_residualnn
    ernn[:]  = error_residualnn
    pnn[:]  = predknn
    enn[:]  = errorknn 
    fnn[:]  = fitnn 
    
    lnn[:]  = linfitnn
    lp1nn[:]  = linfitp1nn
    lp2nn[:]  = linfitp2nn
    lm1nn[:]  = linfitm1nn
    lm2nn[:]  = linfitm2nn
    
#------------------------------------------------------ 
#------------------------------------------------------ 
'''

'''
#------------------------------------------------------
#------------------------------------------------------ 
#PLOT: HINDCAST TIME SERIES, JJA INITIALIZED
#------------------------------------------------------
#------------------------------------------------------ 

def rmse(y_true_obs,y_pred_obs):
    nmt = y_true_obs.shape[0]
    return np.sqrt(np.divide(np.nansum(np.square(y_pred_obs-y_true_obs)),nmt))

def acc(y_true,y_pred):
    return (np.nansum((y_true-np.nanmean(y_true))*(y_pred-np.nanmean(y_pred))))/((np.sqrt(np.nansum(np.square(y_true-np.nanmean(y_true)))))*(np.sqrt(np.nansum(np.square(y_pred-np.nanmean(y_pred))))))
       
#------------------------------------
#JJAS Initialized
#------------------------------------
fig = plt.figure(figsize=(80,60))
label_large = 100
label_small = 80

plt.subplot(2,2,1)
pmon=4
timep = yearpred[pmon,0,:]
accmay = np.round(acc(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
rmsemay = np.round(rmse(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
accmaydetrended = np.round(acc(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
rmsemaydetrended = np.round(rmse(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
plt.plot(np.arange(1979,2025),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5,label='observations')
plt.plot(timep,sie_pred[pmon,0,0,:],color= 'r', linewidth=6.0, marker='o', markersize=5,label='prediction')
plt.title('(a) June 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
plt.ylabel('Sea Ice Extent (M km$^2$)',fontsize=label_large)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
#plt.legend(fontsize=label_small,loc='upper right')
accmayformatted = f'{accmay:0.2f}'
rmsemayformatted = f'{rmsemay:0.2f}'
accmaydetrendedformatted = f'{accmaydetrended:0.2f}'
rmsemaydetrendedformatted = f'{rmsemaydetrended:0.2f}'
fig.text(0.15, 0.58,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.15, 0.56,f'{accmayformatted} ({accmaydetrendedformatted})       {rmsemayformatted} ({rmsemaydetrendedformatted})', fontsize=label_small)


plt.subplot(2,2,2)
pmon=5
timep = yearpred[pmon,0,:]
accjun = np.round(acc(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
rmsejun = np.round(rmse(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
accjundetrended = np.round(acc(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
rmsejundetrended = np.round(rmse(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
plt.plot(np.arange(1979,2025),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5,label='observations')
plt.plot(timep,sie_pred[pmon,0,0,:],color= 'r', linewidth=6.0, marker='o', markersize=5,label='prediction')
plt.title('(b) July 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
plt.legend(fontsize=label_small,loc='upper right')
accjunformatted = f'{accjun:0.2f}'
rmsejunformatted = f'{rmsejun:0.2f}'
accjundetrendedformatted = f'{accjundetrended:0.2f}'
rmsejundetrendedformatted = f'{rmsejundetrended:0.2f}'
fig.text(0.58, 0.58,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.58, 0.56,f'{accjunformatted} ({accjundetrendedformatted})       {rmsejunformatted} ({rmsejundetrendedformatted})', fontsize=label_small)

plt.subplot(2,2,3)
pmon=6
accjul = np.round(acc(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
rmsejul = np.round(rmse(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
accjuldetrended = np.round(acc(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
rmsejuldetrended = np.round(rmse(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
timep = yearpred[pmon,0,:]
plt.plot(np.arange(1979,2025,),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5)
plt.plot(timep,sie_pred[pmon,0,0,:],color= 'r', linewidth=6.0, marker='o', markersize=5)
plt.title('(c) August 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
plt.ylabel('Sea Ice Extent (M km$^2$)',fontsize=label_large)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
accjulformatted = f'{accjul:0.2f}'
rmsejulformatted = f'{rmsejul:0.2f}'
accjuldetrendedformatted = f'{accjuldetrended:0.2f}'
rmsejuldetrendedformatted = f'{rmsejuldetrended:0.2f}'
fig.text(0.15, 0.15,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.15, 0.13,f'{accjulformatted} ({accjuldetrendedformatted})       {rmsejulformatted} ({rmsejuldetrendedformatted})', fontsize=label_small)

plt.subplot(2,2,4)
pmon = 7
accaug = np.round(acc(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
rmseaug = np.round(rmse(sie_monthly[8,1:,0],sie_pred[pmon,0,0,:]),2)
accaugdetrended = np.round(acc(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
rmseaugdetrended = np.round(rmse(test_residual[13,0,0,2:],prediction_residual[pmon,0,0,:]),2)
timep = yearpred[pmon,0,:]
plt.plot(np.arange(1979,2025,),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5)
plt.plot(timep,sie_pred[pmon,0,0,:],color= 'r', linewidth=6.0, marker='o', markersize=5)
plt.title('(d) September 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
#plt.ylabel('Sea Ice Extent (M km$^2$)',fontsize=label_large)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
accaugformatted = f'{accaug:0.2f}'
rmseaugformatted = f'{rmseaug:0.2f}'
accaugdetrendedformatted = f'{accaugdetrended:0.2f}'
rmseaugdetrendedformatted = f'{rmseaugdetrended:0.2f}'
fig.text(0.58, 0.15,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.58, 0.13,f'{accaugformatted} ({accaugdetrendedformatted})       {rmseaugformatted} ({rmseaugdetrendedformatted})', fontsize=label_small)
#------------------------------------------------------
#------------------------------------------------------ 
'''

'''
#------------------------------------------------------ 
#------------------------------------------------------ 
# SAVE: HISTORICAL JJAS PREDICTION, 2DNN
#------------------------------------------------------ 
#------------------------------------------------------ 
acci = np.array([accmay,accjun,accjul,accaug])
rmsei = np.array([rmsemay,rmsejun,rmsejul,rmseaug])
accdetrended = np.array([accmaydetrended,accjundetrended,accjuldetrended,accaugdetrended])
rmsedetrended = np.array([rmsemay,rmsejundetrended,rmsejuldetrended,rmseaugdetrended])
timem = np.arange(1978,2025)
siemsep = sie_monthly[8,:,0]
siepred = sie_pred[4:8,0,0,:]
timep = yearpred[4:8,0,:]

savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_historicalJJAS_2DNN.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',acci.shape[0]) #no. time frames
    file.createDimension('nt',timep.shape[1]) #hindcast lag
    file.createDimension('nt1',siemsep.shape[0]) #hindcast lag
    file.createDimension('ntm',timem.shape[0]) #hindcast lag
  

    #create variables
    #NN performance
    ac = file.createVariable('acci','f4',('tf')) 
    rm = file.createVariable('rmsei','f4',('tf'))
    acd = file.createVariable('accdetrended','f4',('tf'))  
    rmd = file.createVariable('rmsedetrended','f4',('tf'))
    tp = file.createVariable('timep','f4',('tf','nt'))
    tm = file.createVariable('timem','f4',('ntm'))
    sms = file.createVariable('siemsep','f4',('nt1'))
    sp = file.createVariable('siepred','f4',('tf','nt'))
    
    #write data to variables
    ac[:] = acci
    rm[:] = rmsei
    acd[:] = accdetrended 
    rmd[:] = rmsedetrended
    tp[:] = timep
    tm[:] = timem
    sms[:] = siemsep
    sp[:] = siepred

#------------------------------------------------------ 
#------------------------------------------------------ 
'''