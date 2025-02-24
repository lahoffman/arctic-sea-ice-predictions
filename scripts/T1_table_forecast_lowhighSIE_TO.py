#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:13:58 2024

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
# I. LOAD TRANSFER OPERATOR
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
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------



#------------------------------------------------------
#------------------------------------------------------
# II. LOAD TEST DATA: NSIDC OBSERVATIONS
#------------------------------------------------------
#------------------------------------------------------

#[MMM, demeaned, detrended] residual from cmip6 historical (1979-2014) + residual cmip6 ssp585 (2015-2024)
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
sie_original = np.array(dataset.variables['sie_obs'])
sie_obs_te = np.array(dataset.variables['residual_mean_weighted']) #[10,17,45]
fit_obs = np.array(dataset.variables['fit_mean_weighted']) #[10,17,45]
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])


#normalize residual by mean and stdev of training data
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

#******************************************************************************************************************************************************
#ON/OFF for STANDARIZED OBS
sie_obs = np.array(outer)[:,:,1:] #standardized
#sie_obsi = np.array(outeri)[:,:,np.newaxis,:] #non-standardized
#******************************************************************************************************************************************************

'''
#rearrange shape of data
outer = []
for i in range(17):
    inner = []
    for j in range(47):
        te = sie_obsi[i,:,0,j]
        inner.append(te)
    outer.append(inner)
sie_obs = np.array(outer)   

#set residual to one unit standard deviation
outer = []
outerstd = []
outermean = []
for i in range(17):
    inner = []
    innerstd = []
    innermean = []
    for j in range(10):
        te = sie_obs_te[j,i,:] 
        test = np.nanstd(sie_obs_te[j,i,:]) 
        tem = np.nanmean(sie_obs_te[j,i,:])
        ted = np.divide((te-tem),test)       
        inner.append(ted)
        innerstd.append(test)
        innermean.append(tem)
    outer.append(inner)
    outerstd.append(innerstd)
    outermean.append(innermean)
sie_obs = np.array(outer)
residual_std = np.array(outerstd)
residual_mean = np.array(outermean)
'''

test_data = sie_obs[:,np.newaxis,:,:] #2002-2020
test_original = sie_original[:,np.newaxis,:,:]
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# III. LOAD TRAINING DATA: CMIP6 MODELS
#------------------------------------------------------
#------------------------------------------------------
loadpath_sie = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset_sie =  nc.Dataset(loadpath_sie,'r')
years = dataset_sie.variables['unique_years']
sie_model = sie_training
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
# PREDICTED POSSIBILITIES: BIN VALUES [MIDDLE]
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
#PREDICTIONS
#make predictions based on the residuals calculated from the observations
#------------------------------------------------------
#------------------------------------------------------
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
                      
            sie = np.reshape(test_data[i,:,j,:],[46,])
            
            sie_bins = assign_time_series_to_bins(sie,transferbins)
            nt = sie.shape[0]
            
            stdev = []
            prediction = []
            probability = []
            for k in range(nt):            
                bi = sie_bins[k]
                if ~np.isnan(bi):
                    prob_k = transferop[bi,:]
                else: 
                    prob_k = np.full([22,], np.nan)
                
                #prediction is expected value
                predictionk = np.sum(xi*prob_k) 
                stdevk = np.sqrt(np.sum(np.multiply(np.square(xi-predictionk),prob_k),axis=0))
                
                stdev.append(stdevk)
                prediction.append(predictionk)
                probability.append(prob_k)
            stdevj.append(stdev)
            predictionj.append(prediction)
            probabilityj.append(probability)
            binsj.append(sie_bins)
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



#------------------------------------------------------ 
#------------------------------------------------------ 
#~~~~~~~~~~~~~~~~~~~~~FIGURE~~~~~~~~~~~~~~~~~~~~~~~~~~~
# prediction statistics, standardized residual
#------------------------------------------------------
#------------------------------------------------------ 
years = []
for i in range(10):
    yearsi = np.arange(1979+i+1,2024+i+1)
    years.append(yearsi)
yearspred = np.array(years)

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
yearpred = np.array(yp)[:,:,:]
yearinput = np.array(yi)[:,:,:]
years_fit = np.arange(1979,2025)

#------------------------------------------------------ 
#------------------------------------------------------ 
# METRIC: FORECAST PROBABILITY LOW/HIGH SIE
#------------------------------------------------------ 
#------------------------------------------------------ 
ps = 13 #september, predicted time frame

#---------------------------------
#*********************************
#TOGGLE TO DESIRED INTEREST
pp = 11 #starting time frame
year_of_interest = 2024
#*********************************
#---------------------------------



label_large = 110
label_small = 90

#prediction
prediction_array = np.array(predictionl)
error_array = np.array(stdevl)

#time based on year of interest
yearsinterest = yearpred[pp,0,:]
if tf[pp] == 1:
    initialize = np.array(np.where(yearsinterest == year_of_interest-1))[0,0]
    #use this for climatology lines on first row
    timek = np.arange(yearsinterest[initialize]-1,yearsinterest[initialize]+7)
    time_input = yearsinterest[initialize]
else:
    initialize = np.array(np.where(yearsinterest == year_of_interest))[0,0]
    #use this for climatology lines on first row
    timek = np.arange(yearsinterest[initialize]-2,yearsinterest[initialize]+6) 
    time_input = yearsinterest[initialize]
    


#transfer bins for time frame
bins = transfer_bins[:,pp]
bin_width = bins[21]-bins[20]
probabilities = np.array(probabilityl)
total_probability= np.sum(probabilities,axis=4)
pdfs = probabilities*100


highall = []
lowall = []
for k in range(10):
    #mov_mean index
    mov_mean = [4] #index of n+1 - year average
    mov_meani = k
    mov_mean_label = mov_meani+1
    time_pred = yearpred[pp,mov_meani,initialize]
    time_pred_5 = np.arange(time_pred,time_pred+5)
    
    #data: sie_obs from input ; tt = starting year
    data = sie_obs[pp,mov_meani,initialize+1]
    
    #prediction for september from pp = starting time frame ; tt = starting year
    predk = prediction_array[:,pp,mov_meani,initialize+1]
    errork = error_array[:,pp,mov_meani,initialize+1]
    
    
    #expected value, i.e. the prediction from
    #pp = starting time frame ; tt = starting year
    bin_array = np.array(binsl)
    bink = np.array(bin_array[:,pp,mov_meani,initialize+1],dtype=int)
    expected = xi[bink]
    
    probabilityk = probabilities[mov_meani,pp,mov_meani,initialize+1,:]
    total_probability= np.sum(probabilityk,axis=0)
    pdfs = probabilityk*100
    
    #climatology statistics for september
    sie_train = np.reshape(sie_model[13,:,:,mov_meani],148*165,)
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
    clim_bins = np.array(assign_time_series_to_bins(sie_train,transfer_bins[:,j]))
    nb = clim_bins.shape[0]
    
    #find number in each bin
    clim_pdf = []
    for h in range(22):
        ph = np.divide(np.sum(clim_bins == h),nb)
        clim_pdf.append(ph) 
    climatological_probabilities = np.array(clim_pdf)*100
    
    #categorize pdf: high, low, moderate, etc.
    pdf = pdfs
    binsk = bins[1:]
    high = np.round(np.sum(pdf[binsk > clim_mean]),decimals=1)
    high_moderate = np.round(np.sum(pdf[(binsk > clim_mean) & (binsk < clim_stdp1_all)]),decimals=1)
    high_intense = np.round(np.sum(pdf[(binsk > clim_stdp1_all) & (binsk< clim_stdp2_all)]),decimals=1)
    high_extreme = np.round(np.sum(pdf[binsk > clim_stdp2_all]),decimals=1)
    low = np.round(np.sum(pdf[binsk < clim_mean]),decimals=1)
    low_moderate = np.round(np.sum(pdf[(binsk < clim_mean) & (binsk > clim_stdm1_all)]),decimals=1)
    low_intense = np.round(np.sum(pdf[(binsk < clim_stdm1_all) & (binsk > clim_stdm2_all)]),decimals=1)
    low_extreme = np.round(np.sum(pdf[binsk < clim_stdm2_all]),decimals=1)


    lowall.append(low)
    highall.append(high)
    

lowprob = np.array(lowall)
highprob = np.array(highall)
