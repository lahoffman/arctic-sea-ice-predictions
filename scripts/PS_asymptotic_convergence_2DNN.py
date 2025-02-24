
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:50:23 2024

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
from matplotlib.colors import TwoSlopeNorm

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
# I. LOAD TRAINING DATA
#------------------------------------------------------
#------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
transfer_operator = dataset.variables['transfer_operator']
transfer_bins = dataset.variables['transfer_bins']
train_data = dataset.variables['train_standardized']
test_data = dataset.variables['test_standardized']
miu_train = dataset.variables['miu_tr']
sigma_train = dataset.variables['sigma_tr']
tb = np.array(transfer_bins)


#rearrange array so same shape as observations data
sietr = []
siete = []
for i in range(17):
    sietri = train_data[:,i,:,:]
    sietei = test_data[:,i,:,:]
    
    sietr.append(sietri)
    siete.append(sietei)
    
sie_training = np.array(sietr)
sie_testing = np.array(siete)
sie_model = sie_training
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# II. LOAD TEST DATA: NSIDC OBSERVATIONS
#------------------------------------------------------
#------------------------------------------------------
#residual from cmip6 (1979-2014) + moving mean (2015-2024)
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202312_vX.nc'
dataset =  nc.Dataset(load_path,'r')
sie_original = np.array(dataset.variables['sie_obs'])
sie_obs_te = np.array(dataset.variables['residual_mean_weighted']) #[10,17,45]
fit_obs = np.array(dataset.variables['fit_mean_weighted']) #[10,17,45]
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])


#set residual to one unit standard deviation
outer = []
outeri = []
for i in range(17):
    inner = []
    inneri = []
    for j in range(10):
        te = sie_obs_te[j,i,:] 
        test = sigma_train[0,i,0,j]
        tem = miu_train[0,i,0,j]
        ted = np.divide((te-tem),test)
        
        tei = sie_obs_te[j,i,:]       
        inner.append(ted) #standardized
        inneri.append(tei) #non-standardized
    outer.append(inner)
    outeri.append(inneri)
sie_obs = np.array(outer)

test_data = sie_obs[:,np.newaxis,:,:] #2002-2020
test_original = sie_original[:,np.newaxis,:,:]
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
probabilityl = dataset.variables['nn_probability_pred']

#bins
data = np.load("/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/model_output/bin_centers.npz")
bins = data["bins"]

#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------
#------------------------------------------------------
# BIN INDICES
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
#~~~~~~~~~~~~~~~~~~~~~FIGURE~~~~~~~~~~~~~~~~~~~~~~~~~~~
# prediction statistics, standardized residual
#------------------------------------------------------
#------------------------------------------------------ 
years = []
yearsm = []
for i in range(10):
    yearsi = np.arange(1978+i+1,2025+i+1)
    years.append(yearsi)
    
yearspred = np.array(years)
yearsinput = np.arange(1978,2025)

years = np.arange(1978,2025)

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
# ASYMPTOTIC CONVERGEnCE
#------------------------------------------------------ 
#------------------------------------------------------ 

#****EDIT START YEAR******


yearstart = np.arange(2000,2024)
for j in range(12,13):
    ps = 13 #september, predicted time frame
    
    #---------------------------------
    #*********************************
    #TOGGLE TO DESIRED INTEREST
    pp = 6 #starting time frame
    year_of_interest = 2012
    
    #*********************************
    #---------------------------------
    
    #figure properties
    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(96,96))
    axes = [ax1, ax2, ax3, ax4]
    
    #fig, ((ax1, ax2, ax3,ax4),(ax5, ax6, ax7,ax8)) = plt.subplots(2, 4, figsize=(160,96))
    #axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    
    kl = np.concatenate((np.arange(0,3)[:,np.newaxis],np.arange(3,6)[:,np.newaxis],np.arange(6,9)[:,np.newaxis]),axis=1)
    titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    title_letter = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)','(s)','(t)','(u)','(v)','(w)','(x)','(y)','(z)']
    label_large = 110
    label_small = 90
    
    #prediction
    prediction_array = np.array(predictionl)
    error_array = np.array(stdevl)
    
    
        
    
    
    #transfer bins for time frame
    bin_width = bins[21]-bins[20]
    probabilities = np.array(probabilityl)
    total_probability= np.sum(probabilities,axis=4)
    pdfs = probabilities*100
    
        

        
    #time based on year of interest
    if tf[pp] == 1:
        initialize = np.array(np.where(yearsinput == year_of_interest-1))[0,0]
        #use this for climatology lines on first row
        timek = np.arange(yearsinput[initialize]-1,yearsinput[initialize]+7)
        time_input = yearsinput[initialize]
    else:
        initialize = np.array(np.where(yearsinput == year_of_interest))[0,0]
        #use this for climatology lines on first row
        timek = np.arange(yearsinput[initialize]-2,yearsinput[initialize]+6) 
        time_input = yearsinput[initialize]
        
    for k in range(4):
            
        #mov_mean index
        #mov_mean = [0,1,2,3,4,5,6,7,8,9] #index of n+1 - year average
        mov_mean = [0,1,4,9] #index of n+1 - year average
        mov_meani = mov_mean[k]
        mov_mean_label = mov_meani+1
        time_pred = yearpred[pp,mov_meani,initialize]
        time_pred_5 = np.arange(time_pred,time_pred+5)
    
        #data: sie_obs from input ; tt = starting year
        data = sie_obs[pp,4,initialize]
        
        #prediction for september from pp = starting time frame ; tt = starting year
        predk = prediction_array[pp,:,mov_meani,initialize]
        errork = error_array[pp,:,mov_meani,initialize]
        
        
        #expected value, i.e. the prediction from
        #pp = starting time frame ; tt = starting year
        expected = prediction_array[pp,:,mov_meani,initialize]
        
        probabilityk = probabilities[pp,4,mov_meani,initialize,:]
        total_probability= np.sum(probabilityk,axis=0)
        pdfs = probabilityk*100
        
        #climatology statistics for september
        sie_train = np.reshape(sie_model[13,:,:,4],148*165,)
        clim_mean = np.nanmean(sie_train)
        clim_std = np.nanstd(sie_train)
        clim_mean_all = clim_mean
        clim_stdp1_all = clim_mean_all+clim_std
        clim_stdp2_all = clim_mean_all+ 2*clim_std
        clim_stdp3_all = clim_mean_all+ 3*clim_std
        clim_stdm1_all = clim_mean_all-clim_std
        clim_stdm2_all = clim_mean_all- 2*clim_std
        clim_stdm3_all = clim_mean_all- 3*clim_std
        axtick = np.array([clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all])
        axtick2 = np.array([clim_stdm3_all,clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all,clim_stdp3_all])
        
        #find bins for train data [i.e. climatological PDF]
        clim_bins = np.array(assign_time_series_to_bins(sie_train,bins))
        nb = clim_bins.shape[0]

        #find number in each bin
        clim_pdf = []
        for h in range(23):
            ph = np.divide(np.sum(clim_bins == h),nb)
            clim_pdf.append(ph) 
        climatological_probabilities = np.array(clim_pdf)*100

        #categorize pdf: high, low, moderate, etc.
        pdf = pdfs
        binsk = bins
        high = np.round(np.sum(pdf[binsk > clim_mean]),decimals=1)
        high_moderate = np.round(np.sum(pdf[(binsk > clim_mean) & (binsk < clim_stdp1_all)]),decimals=1)
        high_intense = np.round(np.sum(pdf[(binsk > clim_stdp1_all) & (binsk< clim_stdp2_all)]),decimals=1)
        high_extreme = np.round(np.sum(pdf[binsk > clim_stdp2_all]),decimals=1)
        low = np.round(np.sum(pdf[binsk < clim_mean]),decimals=1)
        low_moderate = np.round(np.sum(pdf[(binsk < clim_mean) & (binsk > clim_stdm1_all)]),decimals=1)
        low_intense = np.round(np.sum(pdf[(binsk < clim_stdm1_all) & (binsk > clim_stdm2_all)]),decimals=1)
        low_extreme = np.round(np.sum(pdf[binsk < clim_stdm2_all]),decimals=1)
              
        #difference from climatology
        clim_diff = np.divide((pdf-climatological_probabilities),climatological_probabilities)
        clim_blue = clim_diff[clim_diff < 0]
        clim_red = clim_diff[clim_diff > 0]
        bin_red = binsk[clim_diff > 0]
        bin_blue = binsk[clim_diff < 0]
        
        #ylim for third plots
        yf = np.ceil(clim_red)
        yo = -np.ceil(-clim_blue)
        
        #figure titles and axes labels
        if k > 0:
            xt1 = np.array(timek-mov_meani,dtype=str)
            xt2 = np.array(timek,dtype=str)
            xt3 = np.array(timek+mov_meani,dtype=str)
            xt1s = [label[-2:] for label in xt1]
            xt2s = [label[-2:] for label in xt2]
            xt3s = [label[-2:] for label in xt3]
            
            #labels for subplots with time range
            xstre = [f'{cxt1}-{cxt2}' for cxt1, cxt2 in zip(xt1s,xt2s)]
            xstr = [f'{cxt1}-{cxt2}' for cxt1, cxt2 in zip(xt2,xt3)]
            
            #title for two- and five-year forecast [prediction time through prediction time + 2 or 5]
            xstrtit = xstr[2]
        else:
            #title for one year forecast [prediction time]
            xtit = timek[2]
            
        ax = axes[k] 
        titlet = title_letter[k]
        ax.bar(bins[:]-bin_width/2,climatological_probabilities,width=bin_width,alpha=0.8,edgecolor='black',color='grey',label='climatological')
        ax.bar(bins[:]-bin_width/2,pdf,width=bin_width/1.5,alpha=0.7,edgecolor='black',color='red',label= 'prediction')
        ax.bar(data,20,width=0.01,color='blue',label='initial condition')
        #ax.set_xlim(-0.85,0.85)
        ax.set_ylim(0,20)
        #ax.set_title(f'Probabilistic forecast (%) for {mov_mean_label}-year average', fontsize=20)
        ax.tick_params(axis='both', labelsize=label_large)
        #ax.text(0.05,0.95,f'HIGH: {high} \nMOD: {high_moderate} \nINT:: {high_intense} \nEXT: {high_extreme} \nLOW: {low} \nMOD: {low_moderate} \nINT: {low_intense} \nEXT: {low_extreme}',fontsize=label_large,ha='left', va='top', transform=ax.transAxes)
        #ax.set_xlim(axtick2[0],axtick2[-1])
        #ax.set_xticks(axtick)
        ax.grid(True,axis='x')     
        ax.set_title(f'{titlet} After {mov_mean_label} yr', fontsize=label_large)
        if k > 1: 
            ax.set_xlabel('SIE', fontsize=label_large)
        handles, labels = ax2.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper right', ncol=1,fontsize=label_large)

