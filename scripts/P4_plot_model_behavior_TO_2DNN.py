#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:20:29 2024

@author: hoffmanl
"""


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
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib import gridspec

#colorbars
#------------------
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
#LOAD TRANSFER OPERATOR
#------------------------------------------------------
#------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
transfer_operator = dataset.variables['transfer_operator']
transfer_bins = dataset.variables['transfer_bins']
train_data = dataset.variables['train_standardized']
miu_train = dataset.variables['miu_tr']
sigma_train = dataset.variables['sigma_tr']

tb = np.array(transfer_bins)


#reshape training data
sieouter = []
for i in range(17):
    sieinner = []
    for j in range(148):
        siei = train_data[j,i,:,:]
        sieinner.append(siei)
    sieouter.append(sieinner)
sie_m = np.array(sieouter)
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
#BIN INDICES
#------------------------------------------------------
#------------------------------------------------------
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
#PREDICTED POSSIBILITIES: BIN VALUES
#since transferbin is the bin edges, find the middle of the bins
#------------------------------------------------------
#------------------------------------------------------
predicted_poss = []
for k in range(17):
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
# PREDICTIONS FROM INITIAL CONDITIONS
#------------------------------------------------------
#------------------------------------------------------

initial_conditions = np.arange(-3,3.05,0.05)

#make predictions using inputs
#------------------
stdevl = []
predictionl = []
binsl = []
probabilityl = []
for l in range(10):
    stdevi = []
    predictioni = []
    binsi= []
    probabilityi = []
    for i in range(17):
        
        stdevj = []
        binsj = []
        predictionj = []
        probabilityj = []
        for j in range(10):
            transferop = transfer_operator[i,j,l,:,:]
            transferbins = transfer_bins[:,i]
            sic = initial_conditions
            
            sic_bins = assign_time_series_to_bins(sic,transferbins)
            nt = sic.shape[0]
            
            stdev = []
            prediction = []
            probability = []
            for k in range(nt):            
                bi = sic_bins[k]
                if ~np.isnan(bi):
                    prob_k = transferop[bi,:]
                else: 
                    prob_k = np.full([22,], np.nan)
                
                #prediction is expected value
                predictionk = np.sum(np.multiply(xi,prob_k)) 
                stdevk = np.sqrt(np.sum(np.multiply(np.square(xi-predictionk),prob_k),axis=0))
                
                stdev.append(stdevk)
                prediction.append(predictionk)
                probability.append(prob_k)
            stdevj.append(stdev)
            predictionj.append(prediction)
            probabilityj.append(probability)
            binsj.append(sic_bins)
        stdevi.append(stdevj)
        predictioni.append(predictionj)
        probabilityi.append(probabilityj)
        binsi.append(binsj)
    stdevl.append(stdevi)
    predictionl.append(predictioni)
    probabilityl.append(probabilityi)
    binsl.append(binsi)
#------------------------------------------------------
#------------------------------------------------------ 


#------------------------------------------------------------
#------------------------------------------------------------
# I. TRAINING DATA: CMIP6 data 
#------------------------------------------------------------
#------------------------------------------------------------

time_frames = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
start_mo = 11
end_mo = 12

# a. SIEXTENTN
#------------------------------------------------------------
#load_path = '/Users/hoffmanl/Documents/data/mip46/sie_sit_siv_SM_RMMM_NORMZMOS_TSS_MM_ARCTIC.nc'
loadpath_sie = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset_sie =  nc.Dataset(loadpath_sie,'r')
years = dataset_sie.variables['unique_years']
sie_tr = np.array(dataset_sie.variables['sie_ensemble_anomaly'])
tij = np.arange(1,11)
tmj = np.arange(0,10)

siet = []
for i in range(17):
    siei = sie_tr[:,i,:,:]
    siet.append(siei)
sie_training = np.array(siet)


# b. AREA(SIVOL > THRESHOLD)
#------------------------------------------------------------
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siat_1p25m_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM_mon2sep_gn_185001-201412.nc'
dataset =  nc.Dataset(loadpath,'r')
area_tr = np.array(dataset.variables['area_ensemble_anomaly'])

areat = []
for i in range(17):
    areai = area_tr[:,i,:,:]
    areat.append(areai)
area_training = np.array(areat)
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#TRAIN-VALIDATE-TEST SPLIT BY MODEL
#TVT Split: 70-15-15
#------------------------------------------------------------
#------------------------------------------------------------
ensemble_numbers = [10,40,25,33,50,10,3,50,4]
ensemble_gt_10 = [1,2,3,4] #model index for ensemble members > 10
ensemble_cumsum = np.cumsum(ensemble_numbers)
ensemble_no_index = np.int64(np.concatenate((np.zeros(1,),ensemble_cumsum[:-1]),axis=0))
ts = ensemble_no_index[ensemble_gt_10]
te = ts+5

sie = np.concatenate((sie_training[:,0:ts[0],:,:],sie_training[:,te[0]:ts[1],:,:],sie_training[:,te[1]:ts[2],:,:],sie_training[:,te[2]:ts[3],:,:],sie_training[:,te[3]:,:,:]),axis=1)
area = np.concatenate((area_training[:,0:ts[0],:,:],area_training[:,te[0]:ts[1],:,:],area_training[:,te[1]:ts[2],:,:],area_training[:,te[2]:ts[3],:,:],area_training[:,te[3]:,:,:]),axis=1)

siete = np.concatenate((sie_training[:,ts[0]:te[0],:,:],sie_training[:,ts[1]:te[1],:,:],sie_training[:,ts[2]:te[2],:,:],sie_training[:,ts[3]:te[3],:,:]),axis=1)
areate = np.concatenate((area_training[:,ts[0]:te[0],:,:],area_training[:,ts[1]:te[1],:,:],area_training[:,ts[2]:te[2],:,:],area_training[:,ts[3]:te[3],:,:]),axis=1)
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
# STANDARDIZE
#------------------------------------------------------------
#------------------------------------------------------------
sigma_sie = np.nanstd(sie,axis=(1,2))[:,np.newaxis,np.newaxis,:]
miu_sie = np.nanmean(sie,axis=(1,2))[:,np.newaxis,np.newaxis,:]

train_standardized_sie = np.divide((sie-miu_sie),sigma_sie)
test_standardized_sie = np.divide((siete-miu_sie),sigma_sie)

sie_training = train_standardized_sie[5:,:,:,:]
sie_test = test_standardized_sie[5:,:,:,:]

sigma_siat = np.nanstd(area,axis=(1,2))[:,np.newaxis,np.newaxis,:]
miu_siat = np.nanmean(area,axis=(1,2))[:,np.newaxis,np.newaxis,:]

train_standardized_siat = np.divide((area-miu_siat),sigma_siat)
test_standardized_siat = np.divide((areate-miu_siat),sigma_siat)

area_training = train_standardized_siat[5:,:,:,:]
area_test = test_standardized_siat[5:,:,:,:]
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
# II. TEST DATA: OBSERVATIONS, 1979-2014
#------------------------------------------------------
#------------------------------------------------------

# a. NSIDC - SIEXTENTN
#------------------------------------------------------------
#residual from cmip6 ensemble mean (1979-2014) + ssp585 (2015-2024)
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
sie_observed = np.array(dataset.variables['sie_obs'])[:,:,1:]
sie_obs_te = np.array(dataset.variables['residual_mean_weighted'])[:,:,1:]
fit_obs = np.array(dataset.variables['fit_mean_weighted'])[:,:,1:]
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])

res_std = np.nanstd(sie_obs_te,axis=2)
res_mean = np.nanmean(sie_obs_te,axis=2)

# b. PIOMAS - AREA(SIVOL > THRESHOLD)
#------------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siat_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
area_observed = np.array(dataset.variables['siat_obs'])
area_obs_te = np.array(dataset.variables['residual_mean_weighted'])
area_fit_obs = np.array(dataset.variables['fit_mean_weighted'])


#standardize
#----------------------------------------
outer = []
outera = []
for i in range(17):
    inner = []
    innera = []
    for j in range(10):
        te = sie_obs_te[j,i,:]
        test = sigma_sie[i,0,0,j]
        tem = miu_sie[i,0,0,j]
        ted = np.divide(te-tem,test)       
        inner.append(ted)
        
        tea = area_obs_te[j,i,:]
        testa = sigma_siat[i,0,0,j]
        tema = miu_siat[i,0,0,j]
        teda = np.divide(tea-tema,testa)       
        innera.append(teda)
        
    outer.append(inner)
    outera.append(innera)
sie_obsi = np.array(outer)
area_obsi = np.array(outera)

#reshape test data 
#----------------------------------------
outer = []
outera = []
for i in range(17):
    inner = []
    innera = []
    for j in range(46):
        te = sie_obsi[i,:,j]
        inner.append(te)
        
        tea = area_obsi[i,:,j]
        innera.append(tea)
        
    outer.append(inner)
    outernp = np.array(outer)[np.newaxis,:,:]
    outera.append(innera)
    outernpa = np.array(outera)[np.newaxis,:,:]
    
sie_obs = np.array(outer)   
sie_test = sie_obs[5:,np.newaxis,:,:]

area_obs = np.array(outera)   
area_test = area_obs[5:,np.newaxis,:,:]
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------ 
# LOAD NN ICS PREDICTIONS
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_2D_siearea2sie_bins_ICs_2.nc'
dataset =  nc.Dataset(loadpath,'r')
probabilities_nn_ics = np.array(dataset.variables['model_prediction_ics'])

#bins
data = np.load("/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/model_output/bin_centers.npz")
bins = data["bins"]

initial_conditions = np.arange(-3,3.1,0.1)
#------------------------------------------------------
#------------------------------------------------------ 



#------------------------------------------------------ 
#------------------------------------------------------ 
# FIGURE 1: PROBABILITY THAT NN PREDICTS LESS THAN CLIM MEAN FOR ICS
# 1-, 2-, 5-year moving means, July-initialized models
# (a-c) Transfer Operator, 1D
# (d-f) Neural Network, 2D
# (g-j) 2D Histogram, CMIP6 data, SIAt vs. SIE
# (j-l) 2D Histogram, NSIDC obs, SIAt vs. SIE
#------------------------------------------------------ 
#------------------------------------------------------ 

years = np.arange(1979,2022)
from matplotlib.colors import TwoSlopeNorm
    
#figure properties
fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12)) = plt.subplots(3,4, figsize=(300,150))
axes = [ax1,ax5,ax9,ax2,ax6,ax10,ax3,ax7,ax11,ax4,ax8,ax12]
initial_conditions_to = np.arange(-3,3.05,0.05)
kj = np.reshape(np.arange(12),(4,3))
titles = ['yearly','JFM','AMJ','JAS','OND','J','F','M','A','M','J','J','A','S','O','N','D']
bg_color = ['cornflowerblue','lightskyblue','paleturquoise','cornsilk','moccasin','sandybrown']
title_letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
label_large = 150
label_small = 100
kk = [-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
kk = np.array(kk)
    
    
for j in range(4):    
    for k in range(3):   
        ax = axes[kj[j,k]]
        
        #mov_mean index
        mov_mean = [0,1,4] #index of n+1 - year average
        mov_meani = mov_mean[k]
        mov_mean_label = mov_meani+1
        tau = 0
    
        #fiture properties
        mov_mean_label = mov_meani+1
        tlet = title_letters[kj[j,k]]
        if k > 0:
            titlesto = f' {tlet} {mov_mean_label}-year average'
            titlesnn = f' {tlet} {mov_mean_label}-year average'
        else:
            titlesto = f' Probability TO SIE prediction > climatological mean \n {tlet} {mov_mean_label}-year average'
            titlesnn = f' Probability NN SIE prediction > climatological mean \n {tlet} {mov_mean_label}-year average'
        
        #probabilities TO
        prediction_array = np.array(predictionl)
        error_array = np.array(stdevl)
        bins = transfer_bins[:,j]
        bin_width = bins[21]-bins[20]
        probabilities = np.array(probabilityl)
        total_probability= np.sum(probabilities,axis=0)
        probabilitykto = probabilities[:,11,mov_meani,:,:] #11 = July, 7 = March
        pdfsto = probabilitykto*100
        
        #probabilities NN
        probabilityknn = probabilities_nn_ics[6,mov_meani,mov_meani,:,:,:] #6 = July, 2 = March
        pdfsnn = probabilityknn*100
        
        #climatology statistics [zero mean, unit standard deviation]
        pj = 8 #september
        sie_train = np.reshape(sie_training[pj,:,:,mov_meani],148*165,)
        clim_mean = np.nanmean(sie_train)
        
    
        #categorize pdf: high, low, moderate, etc.
        pdf = pdfsnn
        binsk = bins[1:]
        binsknn = np.concatenate(([-np.inf],binsk),axis=0)
        lownn = np.round(np.sum(pdfsnn[:,:,(binsknn < clim_mean)],axis=2),decimals=1)
        lowto = np.round(np.sum(pdfsto[:,:,(binsk < clim_mean)],axis=2),decimals=1)
        
        if j == 0:
            #contour: probability that prediction is less than clim mean for given IC
            contour = ax.contourf(initial_conditions_to,np.arange(1,11),lowto,levels=np.linspace(0, 100, 50),cmap = 'twilight_shifted',norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100))
            c95 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [95],colors = 'white',linewidths=8)
            c05 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [5],colors = 'white',linewidths=8)
            c90 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [90],colors = 'white',linewidths=6,linestyles='--')
            c10 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [10],colors = 'white',linewidths=6,linestyles='--')
            #c85 = ax.contour(initial_conditions,np.arange(1,11),low,levels = [85],colors = 'white',linewidths=4)
            #c15 = ax.contour(initial_conditions,np.arange(1,11),low,levels = [15],colors = 'white',linewidths=4)
            
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.12, 0.1, 0.35, 0.025])  # [left, bottom, width, height]
            cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
                #cbar = fig.colorbar(contour)
            cbar.set_label('probability (%)', fontsize=label_large)
            cbar.set_ticks([0,10,20,30,40,50,60,70,80,90,100])
            cbar.set_ticklabels(['0','10','20','30','40','50','60','70','80','90','100'])
            cbar.ax.tick_params(labelsize=label_large)
            #fig.set_title(f'Probability of sic LOWER than climatological mean (%) for {mov_mean_label}-year average', fontsize=label_large)
                    
            ax.plot(np.zeros(3),[1,5,10],color='black',label = '')
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_xlim([-3,3])
            ax.set_yticks(np.arange(0,11,2))
            ax9.set_xlabel('                                                                        initial condition [standardized SIE]', fontsize=label_large)
            ax5.set_ylabel('time lag [yr]', fontsize=label_large)
            ax.set_title(titlesto,fontsize=label_large)
            
        elif j == 1:
            #contour: probability that prediction is less than clim mean for given IC
            contour = ax.contourf(initial_conditions,initial_conditions,lownn,levels=np.linspace(0, 100, 50),cmap = 'twilight_shifted',norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100))
            c95 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [95],colors = 'white',linewidths=8)
            c05 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [5],colors = 'white',linewidths=8)
            c90 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [90],colors = 'white',linewidths=6,linestyles='--')
            c10 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [10],colors = 'white',linewidths=6,linestyles='--')
            #c85 = ax.contour(initial_conditions,np.arange(1,11),low,levels = [85],colors = 'white',linewidths=4)
            #c15 = ax.contour(initial_conditions,np.arange(1,11),low,levels = [15],colors = 'white',linewidths=4)
            
            fig.subplots_adjust(bottom=0.2)
            #cbar_ax = fig.add_axes([0.13, 0.1, 0.2, 0.025])  # [left, bottom, width, height]
            #cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
                #cbar = fig.colorbar(contour)
            #cbar.set_label('probability (%)', fontsize=label_large)
            #cbar.set_ticks([0,10,20,30,40,50,60,70,80,90,100])
            #cbar.set_ticklabels(['0','10','20','30','40','50','60','70','80','90','100'])
            #cbar.ax.tick_params(labelsize=label_large)
            #fig.set_title(f'Probability of sic LOWER than climatological mean (%) for {mov_mean_label}-year average', fontsize=label_large)
                    
            ax.plot(np.zeros(3),[-2.5,0,2.5],color='black',label = '')
            ax.plot([-3,0,3],np.zeros(3),color='black',label = '')
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_xlim([-3,3])
            ax.set_yticks(np.arange(-2,3,1))
            ax11.set_xlabel('                                                                  initial condition [standardized SIE]', fontsize=label_large)
            ax6.set_ylabel('initial condition [standardized SIAt]', fontsize=label_large)
            ax7.set_ylabel('initial condition [standardized SIAt]', fontsize=label_large)
            ax8.set_ylabel('initial condition [standardized SIAt]', fontsize=label_large)
            ax.set_title(titlesnn,fontsize=label_large)
            
        elif j == 2:
            tlet = title_letters[kj[j,k]]
            if k > 0:
                titles = f' {tlet} {mov_mean_label}-year average'
            else:
                titles = f' Joint Histogram, CMIP6 \n {tlet} {mov_mean_label}-year average'
            
            
            dataa = np.reshape(area_training[6,:,:,k],(148*165,1)).flatten()
            datas = np.reshape(sie_training[6,:,:,k],(148*165,1)).flatten()
            
            # Filter out NaN values
            mask = ~np.isnan(datas) & ~np.isnan(dataa)  # Mask valid data (no NaNs)
            x_valid = datas[mask]
            y_valid = dataa[mask]

            histtr = ax.hist2d(x_valid, y_valid, bins=30, cmap='Blues', vmin=0, vmax=600)
            ax.plot(np.zeros(3),[-4,0,4],color='black')
            ax.plot([-4,0,4],np.zeros(3),color='black')
            ax.set_title(titles,fontsize=label_large)
            ax.set_xlim((-3,3))
            ax.set_ylim((-2.5,2.5))
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_yticks(np.arange(-2,3,1))
            ax.set_xticks(np.arange(-3,3,1))

            # Add colorbar for counts
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.52, 0.1, 0.15, 0.025])  # [left, bottom, width, height]
            cbar = fig.colorbar(histtr[3], cax=cbar_ax, orientation='horizontal')
            cbar.set_label('counts', fontsize=label_large)
            cbar.set_ticks([0,100,200,300,400,500,600])
            cbar.set_ticklabels(['0','100','200','300','400','500','600'])
            cbar.ax.tick_params(labelsize=label_large)


            # Add labels and title
            #ax8.set_xlabel("standardized SIE", fontsize=label_large)
            #ax5.set_ylabel("standardized SIAt", fontsize=label_large)
            

        else:
            
            tlet = title_letters[kj[j,k]]
            if k > 0:
                titles = f' {tlet} {mov_mean_label}-year average'
            else:
                titles = f' Joint Histogram, observations \n {tlet} {mov_mean_label}-year average'
            
            dataa = np.reshape(area_test[8,:,:,k],(46,1)).flatten()
            datas = np.reshape(sie_test[8,:,:,k],(46,1)).flatten()
            
            # Filter out NaN values
            mask = ~np.isnan(datas) & ~np.isnan(dataa)  # Mask valid data (no NaNs)
            x_valid = datas[mask]
            y_valid = dataa[mask]

            histte = ax.hist2d(x_valid, y_valid, bins=10, cmap='Blues',vmin=0, vmax=3)
            
            ax.plot(np.zeros(3),[-4,0,4],color='black')
            ax.plot([-4,0,4],np.zeros(3),color='black')
            ax.set_title(titles,fontsize=label_large)
            ax.set_xlim((-3,3))
            ax.set_ylim((-2.5,2.5))
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_yticks(np.arange(-2,3,1))
            ax.set_xticks(np.arange(-3,3,1))

            # Add colorbar for counts
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.7, 0.1, 0.15, 0.025])  # [left, bottom, width, height]
            cbar = fig.colorbar(histte[3], cax=cbar_ax, orientation='horizontal')
            cbar.set_label('counts', fontsize=label_large)
            cbar.set_ticks([0,1,2,3])
            cbar.set_ticklabels(['0','1','2','3'])
            cbar.ax.tick_params(labelsize=label_large)


            # Add labels and title
            #ax9.set_xlabel("standardized SIE", fontsize=label_large)
            #ax6.set_ylabel("standardized SIAt", fontsize=label_large)
           
    
        fig.subplots_adjust(right=0.85,hspace=0.3)

 
plt.show
#------------------------------------------------------
#------------------------------------------------------ 
        


#------------------------------------------------------ 
#------------------------------------------------------ 
# FIGURE 2: PROBABILITY THAT NN PREDICTS LESS THAN CLIM MEAN FOR ICS
# July-initialized models
# (a) Transfer Operator, 1D
# (b) Neural Network, 2D
# (c) 2D Histogram, CMIP6 data, SIAt vs. SIE
# (d) 2D Histogram, NSIDC obs, SIAt vs. SIE
#------------------------------------------------------ 
#------------------------------------------------------ 

years = np.arange(1979,2022)
from matplotlib.colors import TwoSlopeNorm


    
#figure properties
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(200,200))
axes = [ax1,ax2,ax3,ax4]
initial_conditions_to = np.arange(-3,3.05,0.05)
kj = np.reshape(np.arange(12),(4,3))
titles = ['yearly','JFM','AMJ','JAS','OND','J','F','M','A','M','J','J','A','S','O','N','D']
bg_color = ['cornflowerblue','lightskyblue','paleturquoise','cornsilk','moccasin','sandybrown']
title_letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
label_large = 150
label_small = 100
kk = [-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
kk = np.array(kk)
    
    
for j in range(4):    
    for k in range(1):   
        ax = axes[j]
        
        #mov_mean index
        mov_mean = [0,1,4] #index of n+1 - year average
        mov_meani = mov_mean[k]
        mov_mean_label = mov_meani+1
        tau = 0
    
        #fiture properties
        mov_mean_label = mov_meani+1
         
        #probabilities TO
        prediction_array = np.array(predictionl)
        error_array = np.array(stdevl)
        bins = transfer_bins[:,j]
        bin_width = bins[21]-bins[20]
        probabilities = np.array(probabilityl)
        total_probability= np.sum(probabilities,axis=0)
        probabilitykto = probabilities[:,12,mov_meani,:,:]
        pdfsto = probabilitykto*100
        
        #probabilities NN
        probabilityknn = probabilities_nn_ics[7,mov_meani,mov_meani,:,:,:]
        pdfsnn = probabilityknn*100
        
        #climatology statistics [zero mean, unit standard deviation]
        pj = 8 #september
        sie_train = np.reshape(sie_training[pj,:,:,mov_meani],148*165,)
        clim_mean = np.nanmean(sie_train)
        
    
        #categorize pdf: high, low, moderate, etc.
        pdf = pdfsnn
        binsk = bins[1:]
        binsknn = np.concatenate(([-np.inf],binsk),axis=0)
        lownn = np.round(np.sum(pdfsnn[:,:,(binsknn < clim_mean)],axis=2),decimals=1)
        lowto = np.round(np.sum(pdfsto[:,:,(binsk < clim_mean)],axis=2),decimals=1)
        
        if j == 0:
            #contour: probability that prediction is less than clim mean for given IC
            contour = ax.contourf(initial_conditions_to,np.arange(1,11),lowto,levels=np.linspace(0, 100, 50),cmap = 'twilight_shifted',norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100))
            c95 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [95],colors = 'white',linewidths=8)
            c05 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [5],colors = 'white',linewidths=8)
            c90 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [90],colors = 'white',linewidths=6,linestyles='--')
            c10 = ax.contour(initial_conditions_to,np.arange(1,11),lowto,levels = [10],colors = 'white',linewidths=6,linestyles='--')

            fig.subplots_adjust(bottom=0)
            cbar_ax = fig.add_axes([0.15, 0.53, 0.7, 0.025])  # [left, bottom, width, height]
            cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
                #cbar = fig.colorbar(contour)
            cbar.set_label('probability (%)', fontsize=label_large)
            cbar.set_ticks([0,10,20,30,40,50,60,70,80,90,100])
            cbar.set_ticklabels(['0','10','20','30','40','50','60','70','80','90','100'])
            cbar.ax.tick_params(labelsize=label_large)
       
            ax.plot(np.zeros(3),[1,5,10],color='black',label = '')
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_xlim([-3,3])
            ax.set_ylim([1,10])
            ax.set_yticks(np.arange(1,11,2))
            ax1.set_xlabel('initial condition [standardized SIE]', fontsize=label_large)
            ax1.set_ylabel('time lag [yr]', fontsize=label_large)
            ax.set_title(f'(a) Probability TO SIE prediction > climatological mean',fontsize=label_large)
            
        elif j == 1:
            #contour: probability that prediction is less than clim mean for given IC
            contour = ax.contourf(initial_conditions,initial_conditions,lownn,levels=np.linspace(0, 100, 50),cmap = 'twilight_shifted',norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100))
            c95 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [95],colors = 'white',linewidths=8)
            c05 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [5],colors = 'white',linewidths=8)
            c90 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [90],colors = 'white',linewidths=6,linestyles='--')
            c10 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [10],colors = 'white',linewidths=6,linestyles='--')

            fig.subplots_adjust(bottom=0)
            ax.plot(np.zeros(3),[-2.5,0,2.5],color='black',label = '')
            ax.plot([-3,0,3],np.zeros(3),color='black',label = '')
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_xlim([-3,3])
            ax.set_yticks(np.arange(-2,3,1))
            ax2.set_xlabel('initial condition [standardized SIE]', fontsize=label_large)
            ax3.set_xlabel('initial condition [standardized SIE]', fontsize=label_large)
            ax4.set_xlabel('initial condition [standardized SIE]', fontsize=label_large)
            ax2.set_ylabel('initial condition [standardized SIAt]', fontsize=label_large)
            ax3.set_ylabel('initial condition [standardized SIAt]', fontsize=label_large)
            ax4.set_ylabel('initial condition [standardized SIAt]', fontsize=label_large)
            ax.set_title(f'(b) Probability NN SIE prediction > climatological mean',fontsize=label_large)
            
            
        elif j == 2:
            tlet = title_letters[kj[j,k]]
            if k > 0:
                titles = f' {tlet} {mov_mean_label}-year average'
            else:
                titles = f' Joint Histogram, CMIP6 \n {tlet} {mov_mean_label}-year average'
            
            
            dataa = np.reshape(area_training[6,:,:,k],(148*165,1)).flatten()
            datas = np.reshape(sie_training[6,:,:,k],(148*165,1)).flatten()
            
            # Filter out NaN values
            mask = ~np.isnan(datas) & ~np.isnan(dataa)  # Mask valid data (no NaNs)
            x_valid = datas[mask]
            y_valid = dataa[mask]

            histtr = ax.hist2d(x_valid, y_valid, bins=30, cmap='Blues', vmin=0, vmax=600)
            ax.plot(np.zeros(3),[-4,0,4],color='black')
            ax.plot([-4,0,4],np.zeros(3),color='black')
            ax.set_title(f'(c) Joint Histogram, CMIP6',fontsize=label_large)
            ax.set_xlim((-3,3))
            ax.set_ylim((-2.5,2.5))
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_yticks(np.arange(-2,3,1))
            ax.set_xticks(np.arange(-3,3,1))

            # Add colorbar for counts
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.125, 0.125, 0.32, 0.025])  # [left, bottom, width, height]
            cbar = fig.colorbar(histtr[3], cax=cbar_ax, orientation='horizontal')
            cbar.set_label('counts', fontsize=label_large)
            cbar.set_ticks([0,100,200,300,400,500,600])
            cbar.set_ticklabels(['0','100','200','300','400','500','600'])
            cbar.ax.tick_params(labelsize=label_large)

        else:
            dataa = np.reshape(area_test[8,:,:,k],(46,1)).flatten()
            datas = np.reshape(sie_test[8,:,:,k],(46,1)).flatten()
            
            # Filter out NaN values
            mask = ~np.isnan(datas) & ~np.isnan(dataa)  # Mask valid data (no NaNs)
            x_valid = datas[mask]
            y_valid = dataa[mask]

            histte = ax.hist2d(x_valid, y_valid, bins=10, cmap='Blues',vmin=0, vmax=3)
            
            ax.plot(np.zeros(3),[-4,0,4],color='black')
            ax.plot([-4,0,4],np.zeros(3),color='black')
            ax.set_title(f'(d) Joint Histogram, observations',fontsize=label_large)
            ax.set_xlim((-3,3))
            ax.set_ylim((-2.5,2.5))
            ax.tick_params(axis='both', labelsize=label_large)
            ax.set_yticks(np.arange(-2,3,1))
            ax.set_xticks(np.arange(-3,3,1))

            # Add colorbar for counts
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.525, 0.125, 0.32, 0.025])  # [left, bottom, width, height]
            cbar = fig.colorbar(histte[3], cax=cbar_ax, orientation='horizontal')
            cbar.set_label('counts', fontsize=label_large)
            cbar.set_ticks([0,1,2,3])
            cbar.set_ticklabels(['0','1','2','3'])
            cbar.ax.tick_params(labelsize=label_large)
    
        fig.subplots_adjust(right=0.85,hspace=0.3)

 
plt.show
#------------------------------------------------------
#------------------------------------------------------ 
        
        
        
#------------------------------------------------------ 
#------------------------------------------------------ 
# FIGURE 3: PROBABILITY THAT NN PREDICTS LESS THAN CLIM MEAN FOR ICS
# Neural Network, 2D, 1-year lag and averaging, JAN-DEC initialized
#------------------------------------------------------ 
#------------------------------------------------------ 

#figure properties
fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12)) = plt.subplots(3,4, figsize=(300,150))
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
initial_conditions_to = np.arange(-3,3.05,0.05)
kj = np.arange(12)
titles = ['yearly','JFM','AMJ','JAS','OND',' JAN',' FEB',' MAR',' APR',' MAY',' JUN',' JUL',' AUG',' SEP',' OCT',' NOV',' DEC']
bg_color = ['cornflowerblue','lightskyblue','paleturquoise','cornsilk','moccasin','sandybrown']
title_letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
label_large = 150
label_small = 100
kk = [-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
kk = np.array(kk)
  
for k in range(12):   
    ax = axes[k]
    tit_mon = titles[k+5]
    tit_let = title_letters[k]
    tit = tit_let+tit_mon
    
    #mov_mean index
    mov_mean = [0,1,4] #index of n+1 - year average
    mov_meani = mov_mean[0]
    mov_mean_label = mov_meani+1
    tau = 0

    #fiture properties
    mov_mean_label = mov_meani+1
        
    #probabilities TO
    prediction_array = np.array(predictionl)
    error_array = np.array(stdevl)
    bins = transfer_bins[:,j]
    bin_width = bins[21]-bins[20]
    probabilities = np.array(probabilityl)
    total_probability= np.sum(probabilities,axis=0)
    probabilitykto = probabilities[:,k+5,mov_meani,:,:]
    pdfsto = probabilitykto*100
    
    #probabilities NN
    probabilityknn = probabilities_nn_ics[k,mov_meani,mov_meani,:,:,:]
    pdfsnn = probabilityknn*100
    
    #climatology statistics [zero mean, unit standard deviation]
    pj = 8 #september
    sie_train = np.reshape(sie_training[pj,:,:,mov_meani],148*165,)
    clim_mean = np.nanmean(sie_train)
    

    #categorize pdf: high, low, moderate, etc.
    pdf = pdfsnn
    binsk = bins[1:]
    binsknn = np.concatenate(([-np.inf],binsk),axis=0)
    lownn = np.round(np.sum(pdfsnn[:,:,(binsknn < clim_mean)],axis=2),decimals=1)
    lowto = np.round(np.sum(pdfsto[:,:,(binsk < clim_mean)],axis=2),decimals=1)
    

    #contour: probability that prediction is less than clim mean for given IC
    contour = ax.contourf(initial_conditions,initial_conditions,lownn,levels=np.linspace(0, 100, 50),cmap = 'twilight_shifted',norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100))
    c95 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [95],colors = 'white',linewidths=8)
    c05 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [5],colors = 'white',linewidths=8)
    c90 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [90],colors = 'white',linewidths=6,linestyles='--')
    c10 = ax.contour(initial_conditions,initial_conditions,lownn,levels = [10],colors = 'white',linewidths=6,linestyles='--')

    fig.subplots_adjust(bottom=0)
    ax.plot(np.zeros(3),[-2.5,0,2.5],color='black',label = '')
    ax.plot([-3,0,3],np.zeros(3),color='black',label = '')
    ax.tick_params(axis='both', labelsize=label_large)
    ax.set_xlim([-3,3])
    ax.set_yticks(np.arange(-2,3,1))
    ax.set_xlabel('initial condition [standardized SIE]', fontsize=label_large)
    ax.set_ylabel('initial condition [standardized SIAt]', fontsize=label_large)
    ax.set_title(tit,fontsize=label_large)
    #ax2.set_title(f'Probability NN SIE prediction > climatological mean \n (b)',fontsize=label_large)
    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.12, 0.1, 0.7, 0.025])  # [left, bottom, width, height]
    cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
        #cbar = fig.colorbar(contour)
    cbar.set_label('probability (%)', fontsize=label_large)
    cbar.set_ticks([0,10,20,30,40,50,60,70,80,90,100])
    cbar.set_ticklabels(['0','10','20','30','40','50','60','70','80','90','100'])
    cbar.ax.tick_params(labelsize=label_large)
    #fig.set_title(f'Probability of sic LOWER than climatological mean (%) for {mov_mean_label}-year average', fontsize=label_large)
            

    fig.subplots_adjust(right=0.85,hspace=0.3)

 
plt.show
#------------------------------------------------------
#------------------------------------------------------ 