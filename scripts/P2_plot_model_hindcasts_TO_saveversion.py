#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:13:34 2024

@author: hoffmanl
"""


#------------------------------------------------------
#------------------------------------------------------
#change the following lines depending on the TO
#TO time frame:
#linear vs. GMT: 
#------------------------------------------------------
#------------------------------------------------------


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
        dx2 = np.arange(1978,2034)[:,np.newaxis]
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
timeextend = np.arange(1978,2034)
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
#LOAD TRANSFER OPERATOR
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
sie_obs_te = np.array(dataset.variables['residual_mean_weighted']) #[10,17,45]
fit_obs = np.array(dataset.variables['fit_mean_weighted']) #[10,17,45]
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])
    
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
        te = sie_obs_te[j,i,:] 
        test = sigma_train[0,i,0,j]
        tem = miu_train[0,i,0,j]
        ted = np.divide((te-tem),test)
        
        tei = sie_obs_te[j,i,:]       
        inner.append(ted) #standardized
        inneri.append(tei) #non-standardized
        innerstd.append(test)
        innermean.append(tem)
    outer.append(inner)
    outeri.append(inneri)
    outerstd.append(innerstd)
    outermean.append(innermean)
sie_obsi = np.array(outer)
residual_std = np.array(outerstd)
residual_mean = np.array(outermean)

test_data = sie_obsi[:,np.newaxis,:,:] #1978-2024
test_original = sie_observed[:,np.newaxis,:,:]

#reshape
outer = []
outerog = []
for i in range(17):
    inner = []
    innerog = []
    for j in range(46):
        te = sie_obsi[i,:,j+1]
        tog = sie_observed[:,i,j+1]
        
        inner.append(te)
        innerog.append(tog)
    outer.append(inner)
    outerog.append(innerog)
sie_obs = np.array(outer)    
sie_original = np.array(outerog)

sie_monthly = sie_original[5:,:,:] #1979-2024
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
#BINNING
#------------------------------------------------------
#------------------------------------------------------
#which bin does the initial condition fall in
#find bin indices
#------------------
def find_bin_index(value, bin_boundaries):
    # Iterate through bin boundaries
    for i in range(len(bin_boundaries) - 1):
        # Check if the value falls within the current bin boundaries
        if bin_boundaries[i] <= value < bin_boundaries[i + 1]:
            return i  # Return the index of the bin
    # If the value is outside all bins, return None or -1 (depending on preference)
    return np.nan

#assign time series to bins
#------------------
def assign_time_series_to_bins(time_series, bin_boundaries):
    bin_indices = []
    # Iterate through the time series
    for value in time_series:
        # Find the bin index for the current time step
        bin_index = find_bin_index(value, bin_boundaries)
        bin_indices.append(bin_index)
    return bin_indices
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
#PREDICTED POSSIBILITIES
#find the middle of the bins
#------------------------------------------------------
#------------------------------------------------------
predicted_poss = []
for k in range(1):
    bin_means = []
    transferbin = transfer_bins[:,k]
    for i in range(len(transferbin)-1):
        bin_mean = (transferbin[i]+transferbin[i+1])/2
        bin_means.append(bin_mean)
    predicted_possibilities = np.array(bin_means)
    predicted_possibilities[0,] = transferbin[1,]-(transferbin[2,]-transferbin[1,])/2
    predicted_possibilities[21,] = transferbin[21,]+(transferbin[2,]-transferbin[1,])/2
    predicted_poss.append(predicted_possibilities)      
xi = np.array(predicted_possibilities)
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# TO PREDICTION
#make predictions based on the residuals calculated from the observations
#------------------------------------------------------
#------------------------------------------------------
stdevl = []
predictionl = []
binsl = []
probabilityl = []
medianl = []
cilowl = []
cihighl = []
for l in range(10):
    stdevi = []
    predictioni = []
    binsi= []
    probabilityi = []
    mediani = []
    cilowi = []
    cihighi = []
    for i in range(17):
        
        stdevj = []
        binsj = []
        predictionj = []
        probabilityj = []
        medianj = []
        cilowj = []
        cihighj = []
        for j in range(10):
            transferop = transfer_operator[i,j,l,:,:]
            transferbins = transfer_bins[:,i]
                      
            sie = np.reshape(test_data[i,:,j,:],[47,])
            
            sie_bins = assign_time_series_to_bins(sie,transferbins)
            nt = sie.shape[0]
            
            stdev = []
            prediction = []
            probability = []
            mediank = []
            cilowk = []
            cihighk = []
            for k in range(nt):            
                bi = sie_bins[k]
                if ~np.isnan(bi):
                    prob_k = transferop[bi,:]
                else: 
                    prob_k = np.full([22,], np.nan)
                
                #prediction is expected value
                predictionk = np.sum(xi*prob_k) 
                stdevk = np.sqrt(np.sum(np.multiply(np.square(xi-predictionk),prob_k),axis=0))
                
                #median is xi where CDF = 0.5
                cdf = np.cumsum(prob_k)
                cdf05 = np.abs(0.5-cdf)
                cdf0025 = np.abs(0.025-cdf)
                cdf0975 = np.abs(0.975-cdf)
                
                median_index = np.argmin(cdf05)
                median = xi[median_index]
                
                cilow_index = np.argmin(cdf0025)
                ci_low = xi[cilow_index]
                
                cihigh_index = np.argmin(cdf0975)
                ci_high = xi[cihigh_index]
                
                mediank.append(median)
                cilowk.append(ci_low)
                cihighk.append(ci_high)
                stdev.append(stdevk)
                prediction.append(predictionk)
                probability.append(prob_k)
            medianj.append(mediank)
            cilowj.append(cilowk)
            cihighj.append(cihighk)
            stdevj.append(stdev)
            predictionj.append(prediction)
            probabilityj.append(probability)
            binsj.append(sie_bins)
        mediani.append(medianj)
        cilowi.append(cilowj)
        cihighi.append(cihighj)
        stdevi.append(stdevj)
        predictioni.append(predictionj)
        probabilityi.append(probabilityj)
        binsi.append(binsj)
    medianl.append(mediani)
    cilowl.append(cilowi)
    cihighl.append(cihighi)
    stdevl.append(stdevi)
    predictionl.append(predictioni)
    probabilityl.append(probabilityi)
    binsl.append(binsi)
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
#DEFINE PREDICTION TIMES
#------------------------------------------------------
#------------------------------------------------------
years = []
for i in range(10):
    yearsi = np.arange(1980+i,2025+i)
    years.append(yearsi)
    
yearspred = np.array(years)


yearsinput = np.arange(1979,2025)
years = np.arange(1980,2025)

tf = [1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1]

yp = []
yi = []
for i in range(17):
    if tf[i] == 1:
        yearpred = yearspred
        yearinput = yearspred-1

    else:
        yearpred = yearspred
        yearinput = yearspred
        
    yp.append(yearpred)
    yi.append(yearinput)
yearpred = np.array(yp)
yearinput = np.array(yi)
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
prediction_residual = np.array(predictionl)[:,:,:,2:]
predi = []
#time frames
for i in range(17):
    predj = []
    #moving mean
    for j in range(10):
        predh = []
        #lag time
        for h in range(10):
            siem = residual_mean[i,j]
            sies = residual_std[i,j]
            yearspred = yearpred[i,h,:]
            npp = np.array(yearspred.shape[0])
            predk = []
            for k in range(npp):
                yf = find_all_indices(years_fit,yearspred[k])
                if yf != []:
                    fit = np.reshape(fit_obs[j,13,yf],[1,])
                else:
                    fit = np.full((1,), np.nan)
                predo = np.multiply(prediction_residual[h,i,j,k],sies)+siem+fit 
                predon = np.array(predo)
                predk.append(predon)
            predh.append(predk)
        predj.append(predh)
    predi.append(predj)

pred = np.array(predi)[:,:,:,:,0]


#------------------------------------------------------
#------------------------------------------------------


'''
#------------------------------------------------------
#------------------------------------------------------
#EXTREME EVENTS
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
    #yearsinterest = [2012,2020]
    yearsinterest = [2007,2019]
    #---------------------------------
    #*********************************
    #TOGGLE TO DESIRED INTEREST
    pp = 11 #starting time frame
    year_of_interest = yearsinterest[j]
    #*********************************
    #---------------------------------
    
    titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    title_letters = ['(a)','(d)','(b)','(e)','(c)','(f)']
    labelk  = ['1-year mean','2-year mean','5-year mean']
    prediction_array = np.array(predictionl)[:,:,:,2:]
    error_array = np.array(stdevl)[:,:,:,2:]
    median_array = np.array(medianl)
    cilow_array = np.array(cilowl)
    cihigh_array= np.array(cihighl)
    text_large = 50
    text_small = 30
    
    
    #index of prediction year
    inputyears = np.arange(1980,2025)
    predyears = np.arange(1980,2025)
    fityears = np.arange(1980,2025)
    linfityears = np.arange(1978,2034)
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
    medki = []
    errki = []
    cilki = []
    cihki = []
    
    dataextremek = []
    climmeank = []
    climstdp1extremek = []
    climstdp2extremek = []
    climstdm1extremek = []
    climstdm2extremek = []
    for k in range(3):
        predi = []
        erri = []
        medi = []
        cili = []
        cihi = []
        
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
        
            fitk = fit_obs[mov_meani,ps,2:] #1978-2024
            data = sie_original[ps,1:,mov_meani] #1979-2024
            prevsep= np.round(data[timefit],decimals=2)
            predk = (prediction_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timefit] #predk = prediction based on TO applied to residual s from obs    
            errork = error_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]
            #medk = (median_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timeX2+2] #predk = prediction based on TO applied to residual s from obs 
            #cilk = (cilow_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timeX2+2] #predk = prediction based on TO applied to residual s from obs 
            #cihk = (cihigh_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitk[timeX2+2] #predk = prediction based on TO applied to residual s from obs 

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
# SAVE: EXTREME EVENT PREDICTION, TO
#------------------------------------------------------ 
#------------------------------------------------------ 
datato = np.array(dataextremej)
predto = np.array(predj)
errto = np.array(errorj)
climmeanto = np.array(climmeanj)
climstdp1to = np.array(climstdp1extremej)
climstdp2to = np.array(climstdp2extremej)
climstdm1to = np.array(climstdm1extremej)
climstdm2to = np.array(climstdm2extremej)

#savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_TO_julyinit_2012_2020.nc'
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_TO_julyinit_2007_2019.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',datato.shape[0]) #no. time frames
    file.createDimension('ntau',datato.shape[2]) #hindcast lag
    file.createDimension('nT',datato.shape[1]) #no. moving means
  

    #create variables
    #NN performance
    dto = file.createVariable('datato','f4',('tf','nT','ntau')) 
    pto = file.createVariable('predto','f4',('tf','nT','ntau'))
    eto = file.createVariable('errto','f4',('tf','nT','ntau'))  
    cmto = file.createVariable('climmeanto','f4',('tf','nT','ntau'))
    csp1to = file.createVariable('climstdp1to','f4',('tf','nT','ntau'))
    csp2to = file.createVariable('climstdp2to','f4',('tf','nT','ntau'))
    csm1to = file.createVariable('climstdm1to','f4',('tf','nT','ntau'))
    csm2to = file.createVariable('climstdm2to','f4',('tf','nT','ntau'))
    
    #write data to variables
    dto[:] = datato
    pto[:] = predto
    eto[:] = errto
    cmto[:] = climmeanto
    csp1to[:] = climstdp1to
    csp2to[:] = climstdp2to
    csm1to[:] = climstdm1to
    csm2to[:] = climstdm2to

#------------------------------------------------------ 
#------------------------------------------------------ 
'''



'''



#------------------------------------------------------
#------------------------------------------------------ 
#PLOT: HINDCAST TIME SERIES FOR RILES
# prediction must be initialized before 2014
#note: first point on this plot is the same as the first on the next plot
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
    pp = 11 #starting time frame
    year_of_interest = 2010
    #*********************************
    #---------------------------------
 
    
    
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    kl = np.concatenate((np.arange(0,2)[:,np.newaxis],np.arange(2,4)[:,np.newaxis],np.arange(4,6)[:,np.newaxis]),axis=1)
    titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    title_letter = ['(a)','(d)','(b)','(e)','(c)','(f)']
    time = np.arange(1980,2025)
    prediction_array = np.array(predictionl)[:,:,:,2:]
    error_array = np.array(stdevl)[:,:,:,2:]
    

    timek = []
    timeextendk = []
    timeinputk = []
    timedurpredk = []
    datak = []
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
        datai = []
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
                timedurpred = yp[timepred:timepred+11]
            else:
                timedurpred = timeextend[timepred:timepred+11]-1
       
            #fit time indices
            yi = np.arange(1980,2025)
            timeinputstart = np.array(np.where(yi==year_of_interest))[0,0]
            timeinput = np.arange(timeinputstart,timeinputstart+10)
            
            
            #fit
            fitko = fit_obs[mov_meani,ps,2:] #fit from cmip6
            fitk = linear_fit_ext[mov_meani,ps,:] #extended fit from linear, non-standardized
            
            #use fit to obs if we have the data, otherwise use the extended linear fit
            if timeinput[i] < 46:
                fitik = fitko
            else: 
                fitik = fitk
                
            data = sie_original[ps,1:,mov_meani]
            data_init = sie_original[pp,1:,mov_meani]
            predk = (prediction_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani]+residual_mean[ps,mov_meani])+fitik[timeinput[i]]     
            errork = error_array[i,pp,mov_meani,timepred]*residual_std[ps,mov_meani] #+residual_mean[ps,mov_meani] #error on predictions made by TO
            
    
            linfitk = linear_fit_ext[mov_meani,ps,:] #linear fit to obs (residual + forced)
            linfitkp1 = linear_fitp1std[mov_meani,ps,:] #+1*sigma linear fit to obs
            linfitkp2 = linear_fitp2std[mov_meani,ps,:] #+2*sigma linear fit to obs
            linfitkm1 = linear_fitm1std[mov_meani,ps,:] #-1*sigma linear fit to obs
            linfitkm2 = linear_fitm2std[mov_meani,ps,:] #-2*sigma linear fit to obs
            
            #residual
            data_residual = test_data[ps,0,mov_meani,2:]
            data_residual_init = test_data[pp,0,mov_meani,2:]
            pred_residual = prediction_array[i,pp,mov_meani,timepred]
            error_residual = error_array[i,pp,mov_meani,timepred]
            
            #figure properties
            tlet = title_letter[kl[j,k]]
            titles = f' {tlet} Hindcast for {mov_mean_label}-year average'
            
            if j == 0:
                #figure titles and axes labels        
                ax = axes[kl[j,k]] 

                #plot forced + residual, obs
                ax.plot(time,data_residual,linewidth=5,color='black',label='_nolegend_')               
               
                #plot forced + residual, obs for prediction time
                ax.plot(time[timeinput[0]:timeinput[0]+10],data_residual[timeinput[0]:timeinput[0]+10,],linewidth=6,color='blue',label='_nolegend_')         
                
                #plot linear fit
                ax.plot(timeextend,np.zeros([56,1]),linewidth=5,color='grey',label='_nolegend_')
               
                #plot prediction
                ax.errorbar(timedurpred[i],pred_residual,error_residual,color='red',fmt='o',markersize=30,capsize=10,label='_nolegend_')
                ax.set_xlim([time[1],2024])
                   
                #plot input sie
                ax.scatter(time[timeinput[0]],data_residual_init[timeinput[0],],marker='s',s=1000,color='blue',label='_nolegend_')
               
                #plot linear fit   
                ax.plot(timeextend,np.ones([56,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,-1*np.ones([56,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,2*np.ones([56,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,-2*np.ones([56,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.plot(timeextend,-3*np.ones([56,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                ax.set_xlim([time[1],2023])
               
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
                    ax.plot(time[:-1],fitko[:-1],linewidth=5,color='orchid',label='forced, obs')
            
                    #plot forced + residual, obs
                    ax.plot(time,data,linewidth=5,color='black',label='forced+residual, obs')          
                    
                    #plot forced + residual, obs for prediction time
                    ax.plot(time[timeinput[0]:timeinput[0]+10],data[timeinput[0]:timeinput[0]+10,],linewidth=6,color='blue',label='_nolegend_')               
                    
                    #plot linear fit
                    ax.plot(timeextend,linfitk,linewidth=5,color='grey',label='linear fit, obs')
                    
                    #plot prediction
                    ax.errorbar(timedurpred[i],predk,errork,color='red',fmt='o',markersize=25,capsize=10,label='prediction, TO')
                                    
                    #plot input sie
                    #ax.scatter(time[timeinput[0]],data_init[timeinput[0]+10,],marker='s',s=1000,color='blue',label='model input')
                    ax.set_ylim([2,9])
                
                else:
                    #plot fit
                    ax.plot(timeextend[:-1],fitk[:-1],linewidth=5,color='orchid',label='_nolegend_')
                    
                    #plot forced + residual, obs
                    ax.plot(time,data,linewidth=5,color='black',label='_nolegend_')               
                   
                    #plot forced + residual, obs for prediction time
                    ax.plot(time[timeinput[0]:timeinput[0]+10],data[timeinput[0]:timeinput[0]+10,],linewidth=6,color='blue',label='_nolegend_')         
                    
                    #plot linear fit
                    ax.plot(timeextend,linfitk,linewidth=5,color='grey',label='_nolegend_')
                   
                    #plot prediction
                    ax.errorbar(timedurpred[i],predk,errork,color='red',fmt='o',markersize=25,capsize=10,label='_nolegend_')
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
            timeinputi.append(time[timeinput[0]:timeinput[0]+10])
            timedurpredi.append(timedurpred[i])
            datai.append(data)
            data_residuali.append(data_residual)
            data_residual_predtimei.append(data_residual[timeinput[0]:timeinput[0]+10,])
            prediction_residuali.append(pred_residual)
            error_residuali.append(error_residual)
            data_residual_initi.append(data_residual_init[timeinput[0],])
            fitkoi.append(fitko[:-1])
            linfitki.append(linfitk)
            linfitkp1i.append(linfitkp1)
            linfitkp2i.append(linfitkp2)
            linfitkm1i.append(linfitkm1)
            linfitkm2i.append(linfitkm2)
            predki.append(predk)
            errorki.append(errork)
        timek.append(timei)
        timeextendk.append(timeextendi)
        timeinputk.append(timeinputi)
        timedurpredk.append(timedurpredi)
        datak.append(datai)
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
        
#------------------------------------------------------
#------------------------------------------------------ 


#------------------------------------------------------ 
#------------------------------------------------------ 
# SAVE: RILES HINDCAST PREDICTION, TO
#------------------------------------------------------ 
#------------------------------------------------------ 
timeto = np.array(timek)
timeextendto = np.array(timeextendk)
timeinputto = np.array(timeinputk)
timedurpredto = np.array(timedurpredk)
data_residualto = np.array(data_residualk)
data_residual_predtimeto = np.array(data_residual_predtimek)
prediction_residualto = np.array(prediction_residualk)
error_residualto = np.array(error_residualk)
data_residual_initto = np.array(data_residual_initk)
fitto = np.array(fitkok)
linfitto = np.array(linfitkk)
linfitp1to = np.array(linfitkp1k)
linfitp2to = np.array(linfitkp2k)
linfitm1to = np.array(linfitkm1k)
linfitm2to = np.array(linfitkm2k)
predkto = np.array(predkk)
errorkto = np.array(errorkk)
datakto = np.array(datak)

frames = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
yearstr = str(year_of_interest)
savelead = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_TO_'
savetail = '.nc'
savepath = savelead+frames[pp]+yearstr+savetail

with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('T',timeto.shape[0]) #no. moving means
    file.createDimension('ntau',timeto.shape[1]) #hindcast lag
    file.createDimension('nt',timeto.shape[2]) #no. timesteps
    file.createDimension('nt1',fitto.shape[2]) #no. timesteps
    file.createDimension('nte',linfitto.shape[2]) #no. timesteps
    
    #create variables
    #NN performance
    tto = file.createVariable('timeto','f4',('T','ntau','nt')) 
    teto = file.createVariable('timeextendto','f4',('T','ntau','nte')) 
    tito = file.createVariable('timeinputto','f4',('T','ntau','ntau')) 
    tdto = file.createVariable('timedurpredto','f4',('T','ntau')) 
    
    dto = file.createVariable('datakto','f4',('T','ntau','nt'))
    drto = file.createVariable('data_residualto','f4',('T','ntau','nt'))
    drptto = file.createVariable('data_residual_predtimeto','f4',('T','ntau','ntau'))
    drito = file.createVariable('data_residual_initto','f4',('T','ntau'))
    prto = file.createVariable('prediction_residualto','f4',('T','ntau'))
    erto = file.createVariable('error_residualto','f4',('T','ntau'))  
    pto = file.createVariable('predkto','f4',('T','ntau'))
    eto = file.createVariable('errorkto','f4',('T','ntau'))  
    fto = file.createVariable('fitto','f4',('T','ntau','nt1'))  
    
    lto = file.createVariable('linfitto','f4',('T','ntau','nte'))
    lp1to = file.createVariable('linfitp1to','f4',('T','ntau','nte'))
    lp2to = file.createVariable('linfitp2to','f4',('T','ntau','nte'))
    lm1to = file.createVariable('linfitm1to','f4',('T','ntau','nte'))
    lm2to = file.createVariable('linfitm2to','f4',('T','ntau','nte'))
    
    #write data to variables
    tto[:]  = timeto 
    teto[:]  = timeextendto 
    tito[:]  = timeinputto 
    tdto[:]  = timedurpredto
    
    dto[:] = datakto
    drto[:]  = data_residualto
    drptto[:]  = data_residual_predtimeto
    drito[:]  = data_residual_initto
    prto[:]  = prediction_residualto
    erto[:]  = error_residualto
    pto[:]  = predkto
    eto[:]  = errorkto 
    fto[:]  = fitto 
    
    lto[:]  = linfitto
    lp1to[:]  = linfitp1to
    lp2to[:]  = linfitp2to
    lm1to[:]  = linfitm1to
    lm2to[:]  = linfitm2to
    
#------------------------------------------------------ 
#------------------------------------------------------ 
'''



#------------------------------------------------------
#------------------------------------------------------ 
#PLOT: HINDCAST TIME SERIES, JJA INITIALIZED
#------------------------------------------------------
#------------------------------------------------------ 
#------------------------------------------------------
#------------------------------------------------------
#DEFINE PREDICTION TIMES
#------------------------------------------------------
#------------------------------------------------------
years = []
for i in range(10):
    yearsi = np.arange(1979+i,2025+i)
    years.append(yearsi)
    
yearspred = np.array(years)


tf = [1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1]

yp = []
for i in range(17):
    if tf[i] == 1:
        yearpred = yearspred
    else:
        yearpred = yearspred
    yp.append(yearpred)
yearpred = np.array(yp)

#------------------------------------------------------
#------------------------------------------------------

#one year pred, one year moving mean, #prediction of september from each time frame and year
prediction_residual = np.array(predictionl)
sie_pred_residual = np.array(predictionl)[0,:,0,:] 
t0 = [0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0]
fiti = []
for i in range(17):
    fit1i = fit_obs[0,13,:]
    '''
    if t0[i] == 1:
        fit1i = fit_obs[0,13,:-1]
    else: 
        fit1i = fit_obs[0,13,1:]
    '''
    fiti.append(fit1i)
    
fit1 = np.array(fiti)
siestd = np.transpose(np.tile(residual_std[:,0],[47,1]))
siemean = np.transpose(np.tile(residual_mean[:,0],[47,1]))
sie_pred = np.multiply(sie_pred_residual,siestd)+siemean+fit1
sie_pred = sie_pred[:,1:]
sie_original1 = test_original[0,0,:,:]

#persistence, full signal
persistence_predi = []
for i in range(17):
    residual_mon = test_data[i,0,0,:]
    fiti = fit_obs[0,13,:]
    sie_pred_psi = np.multiply(residual_mon,siestd[i,:])+siemean[i,:]+fiti
    persistence_predi.append(sie_pred_psi)

persistence_prediction = np.array(persistence_predi)[:,1:]




#------------------------------------------------------ 
#------------------------------------------------------ 
# SAVE: PREDICTION & NSIDC, 1989-2024
#------------------------------------------------------ 
#------------------------------------------------------ 
siep =  np.multiply(sie_pred_residual,siestd)+siemean+fit1
sie_prediction_residual = prediction_residual[0,5:,0,1:]
sie_prediction = siep[5:,1:]
sie_sep_residual_nsidc = test_data[13,0,0,1:]
sie_sep_nsidc = sie_monthly[8,:,0]
sie_persistence_full_pred = persistence_prediction[5:,:]

savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/prediction_full_detrend_1DTO.nc'
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
    pp = file.createVariable('sie_persistence_full_pred','f4',('tf','nt'))

    #write data to variables
    spr[:] = sie_prediction_residual
    sp[:] = sie_prediction
    ssrn[:] = sie_sep_residual_nsidc
    ssn[:] = sie_sep_nsidc
    pp[:] = sie_persistence_full_pred


#------------------------------------------------------ 
#------------------------------------------------------ 

'''
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
pmon=9
timep = yearpred[pmon,0,:]
accmay = np.round(acc(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
rmsemay = np.round(rmse(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
accmaydetrended = np.round(acc(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
rmsemaydetrended = np.round(rmse(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
accpsmay = np.round(acc(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
rmsepsmay = np.round(rmse(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
accpsmaydetrended = np.round(acc(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
rmsepsmaydetrended = np.round(rmse(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
plt.plot(np.arange(1979,2025),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5,label='observations')
plt.plot(timep[:],sie_pred[pmon,:],color= 'r', linewidth=6.0, marker='o', markersize=5,label='prediction')
plt.plot(timep[:],persistence_prediction[pmon,:],color= 'grey', linewidth=6.0, marker='o', markersize=5,label='persistence')
plt.title('(a) June 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
plt.ylabel('Sea Ice Extent (M km$^2$)',fontsize=label_large)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
#plt.legend(fontsize=label_small,loc='upper right')
fig.text(0.15, 0.58,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.15, 0.56,f'{accmay} ({accmaydetrended})       {rmsemay} ({rmsemaydetrended})', fontsize=label_small)
fig.text(0.15, 0.54,f'{accpsmay:.2f} ({accpsmaydetrended:.2f})       {rmsepsmay:.2f} ({rmsepsmaydetrended:.2f})', fontsize=label_small)


plt.subplot(2,2,2)
pmon=10
timep = yearpred[pmon,0,:]
accjun = np.round(acc(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
rmsejun = np.round(rmse(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
accjundetrended = np.round(acc(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
rmsejundetrended = np.round(rmse(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
accpsjune = np.round(acc(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
rmsepsjune = np.round(rmse(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
accpsjunedetrended = np.round(acc(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
rmsepsjunedetrended = np.round(rmse(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
plt.plot(np.arange(1979,2025),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5,label='observations')
plt.plot(timep[:],sie_pred[pmon,:],color= 'r', linewidth=6.0, marker='o', markersize=5,label='prediction')
plt.plot(timep[:],persistence_prediction[pmon,:],color= 'grey', linewidth=6.0, marker='o', markersize=5,label='persistence')
plt.title('(b) July 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
plt.legend(fontsize=label_small,loc='upper right')
fig.text(0.58, 0.58,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.58, 0.56,f'{accjun} ({accjundetrended})       {rmsejun} ({rmsejundetrended})', fontsize=label_small)
fig.text(0.58, 0.54,f'{accpsjune:.2f} ({accpsjunedetrended:.2f})       {rmsepsjune:.2f} ({rmsepsjunedetrended:.2f})', fontsize=label_small)

plt.subplot(2,2,3)
pmon=11
accjul = np.round(acc(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
rmsejul = np.round(rmse(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
accjuldetrended = np.round(acc(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
rmsejuldetrended = np.round(rmse(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
accpsjuly = np.round(acc(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
rmsepsjuly = np.round(rmse(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
accpsjulydetrended = np.round(acc(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
rmsepsjulydetrended = np.round(rmse(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
timep = yearpred[pmon,0,:]
plt.plot(np.arange(1979,2025,),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5)
plt.plot(timep[:],sie_pred[pmon,:],color= 'r', linewidth=6.0, marker='o', markersize=5)
plt.plot(timep[:],persistence_prediction[pmon,:],color= 'grey', linewidth=6.0, marker='o', markersize=5,label='persistence')
plt.title('(c) August 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
plt.ylabel('Sea Ice Extent (M km$^2$)',fontsize=label_large)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
fig.text(0.15, 0.15,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.15, 0.13,f'{accjul} ({accjuldetrended})       {rmsejul} ({rmsejuldetrended})', fontsize=label_small)
fig.text(0.15, 0.11,f'{accpsjuly:.2f} ({accpsjulydetrended:.2f})       {rmsepsjuly:.2f} ({rmsepsjulydetrended:.2f})', fontsize=label_small)

plt.subplot(2,2,4)
pmon = 12
accaug = np.round(acc(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
rmseaug = np.round(rmse(sie_monthly[8,:,0],sie_pred[pmon,:]),2)
accaugdetrended = np.round(acc(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
rmseaugdetrended = np.round(rmse(test_data[13,0,0,:],prediction_residual[0,pmon,0,:]),2)
accpsaug = np.round(acc(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
rmsepsaug = np.round(rmse(sie_monthly[8,:,0],persistence_prediction[pmon,:]),2)
accpsaugdetrended = np.round(acc(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
rmsepsaugdetrended = np.round(rmse(test_data[13,0,0,:],test_data[pmon,0,0,:]),2)
timep = yearpred[pmon,0,:]
plt.plot(np.arange(1979,2025,),sie_monthly[8,:,0],color='k', linewidth=6.0, marker='o', markersize=5)
plt.plot(timep[:],sie_pred[pmon,:],color= 'r', linewidth=6.0, marker='o', markersize=5)
plt.plot(timep[:],persistence_prediction[pmon,:],color= 'grey', linewidth=6.0, marker='o', markersize=5,label='persistence')
plt.title('(d) September 1st Initialized',fontsize=label_large)
plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
#plt.ylabel('Sea Ice Extent (M km$^2$)',fontsize=label_large)
plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
fig.text(0.58, 0.15,'ACC (detrend) RMSE (detrend)', fontsize=label_small)
fig.text(0.58, 0.13,f'{accaug} ({accaugdetrended})       {rmseaug} ({rmseaugdetrended})', fontsize=label_small)
fig.text(0.58, 0.11,f'{accpsaug:.2f} ({accpsaugdetrended:.2f})       {rmsepsaug:.2f} ({rmsepsaugdetrended:.2f})', fontsize=label_small)

#------------------------------------------------------
#------------------------------------------------------ 
'''

'''
#------------------------------------------------------ 
#------------------------------------------------------ 
# SAVE: HISTORICAL JJAS PREDICTION, TO
#------------------------------------------------------ 
#------------------------------------------------------ 
acci = np.array([accmay,accjun,accjul,accaug])[:,np.newaxis]
rmsei = np.array([rmsemay,rmsejun,rmsejul,rmseaug])[:,np.newaxis]
accdetrended = np.array([accmaydetrended,accjundetrended,accjuldetrended,accaugdetrended])[:,np.newaxis]
rmsedetrended = np.array([rmsemaydetrended,rmsejundetrended,rmsejuldetrended,rmseaugdetrended])[:,np.newaxis]

accpsi = np.array([accpsmay,accpsjune,accpsjuly,accpsaug])[:,np.newaxis]
rmsepsi = np.array([rmsepsmay,rmsepsjune,rmsepsjuly,rmsepsaug])[:,np.newaxis]
accpsdetrended = np.array([accpsmaydetrended,accpsjunedetrended,accpsjulydetrended,accpsaugdetrended])[:,np.newaxis]
rmsepsdetrended = np.array([rmsepsmaydetrended,rmsepsjunedetrended,rmsepsjulydetrended,rmsepsaugdetrended])[:,np.newaxis]

timem = np.arange(1979,2025)[:,np.newaxis]
siemsep = sie_monthly[8,:,0][:,np.newaxis]
siepred = sie_pred[9:13,:]
timep = yearpred[9:13,0,:]

savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_historicalJJAS_TO.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',acci.shape[0]) #no. time frames
    file.createDimension('n1',acci.shape[1]) #no. time frames
    file.createDimension('nt',timep.shape[1]) #hindcast lag
    file.createDimension('nt1',siepred.shape[1]) #hindcast lag
  

    #create variables
    #NN performance
    ac = file.createVariable('acci','f4',('tf','n1')) 
    rm = file.createVariable('rmsei','f4',('tf','n1'))
    acd = file.createVariable('accdetrended','f4',('tf','n1'))  
    rmd = file.createVariable('rmsedetrended','f4',('tf','n1'))
    acps = file.createVariable('accpsi','f4',('tf','n1')) 
    rmps = file.createVariable('rmsepsi','f4',('tf','n1'))
    acdps = file.createVariable('accpsdetrended','f4',('tf','n1'))  
    rmdps = file.createVariable('rmsepsdetrended','f4',('tf','n1'))
    tp = file.createVariable('timep','f4',('tf','nt'))
    tm = file.createVariable('timem','f4',('nt','n1'))
    sms = file.createVariable('siemsep','f4',('nt','n1'))
    sp = file.createVariable('siepred','f4',('tf','nt1'))
    
    #write data to variables
    ac[:] = acci
    rm[:] = rmsei
    acd[:] = accdetrended 
    rmd[:] = rmsedetrended
    acps[:] = accpsi
    rmps[:] = rmsepsi
    acdps[:] = accpsdetrended 
    rmdps[:] = rmsepsdetrended
    tp[:] = timep
    tm[:] = timem
    sms[:] = siemsep
    sp[:] = siepred

#------------------------------------------------------ 
#------------------------------------------------------ 
'''
