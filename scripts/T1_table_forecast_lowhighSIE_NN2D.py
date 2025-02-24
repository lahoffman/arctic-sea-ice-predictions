#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:57:48 2024

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
#LOAD STANDARDIZATION VARIABLES: MIU & SIGMA, TRAINING
#------------------------------------------------------
#------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
transfer_operator = dataset.variables['transfer_operator']
transfer_bins = dataset.variables['transfer_bins']
sie_training = dataset.variables['train_standardized']
test_data = dataset.variables['test_standardized']
miu_train = dataset.variables['miu_tr']
sigma_train = dataset.variables['sigma_tr']
tb = np.array(transfer_bins)


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
sie_observed = np.array(dataset.variables['sie_obs'])[:,:,1:]
sie_obs_te = np.array(dataset.variables['residual_mean_weighted'])[:,:,1:] #[10,17,45]
fit_obs = np.array(dataset.variables['fit_mean_weighted'])[:,:,1:] #[10,17,45]

#standardize
#----------------------------------------
outer = []
outera = []
for i in range(12):
    inner = []
    innera = []
    for j in range(10):
        te = sie_obs_te[j,i+5,:]
        test = sigma_train[0,i+5,0,j]
        tem = miu_train[0,i+5,0,j]
        ted = np.divide(te-tem,test)       
        inner.append(ted)  
    outer.append(inner)
sie_obsi = np.array(outer)

#reshape test data 
#----------------------------------------
outer = []
outerog = []
for i in range(12):
    inner = []
    innerog = []
    for j in range(46):
        te = sie_obsi[i,:,j]
        tog = sie_observed[:,i,j]
        
        inner.append(te)
        innerog.append(tog)
    outer.append(inner)
    outerog.append(innerog)
sie_obs = np.array(outer)    
sie_original = np.array(outerog)

sie_monthly = sie_original
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
#FUZZY CLASSIFICATION BINS FOR OUTPUT
#----------------------------------------
#step 1: define bins
def define_fuzzyBin(data,numberOfBins,binSizeFactor):
    
    #build bins based on standar deviation, size factor and number of bins
    dataSep = data[:,8,:,:]
    sigma = np.round(np.nanstd(dataSep,axis=(0,1)),2)
    binSize = binSizeFactor*sigma[0,]/numberOfBins
    binsPositive = np.transpose(np.arange(0,numberOfBins))*binSize
    binsNegative = -binsPositive[1:,]
    bins = np.concatenate((binsNegative[::-1],binsPositive),axis=0)
    
    #define middle of bins
    binCenters = bins+binSize/2
    
    #set first and last bins to extent to infinity
    bins[0,] = np.NINF
    bins[-1,] = np.inf
    
    #bin labels
    #----------------------------------------
    label = []
    for i in range(bins.shape[0]-1):
        binlabel1 = bins[i]
        binlabel2 = bins[i+1]
        labeli = f'{np.round(binlabel1,2)} to {np.round(binlabel2,2)}'
        label.append(labeli)
    binLabels = label
    
    return bins, binCenters, binLabels, binSize


#a. define fuzzy bins for training data
#----------------------------------------
data = sie_training
numberOfBins=12
binSizeFactor=3
bins, binCenters, binLabels, binSize = define_fuzzyBin(data,numberOfBins,binSizeFactor)  
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
    transferbin = bins
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
#~~~~~~~~~~~~~~~~~~~~~FIGURE~~~~~~~~~~~~~~~~~~~~~~~~~~~
# prediction statistics, standardized residual
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
# METRIC: FORECAST PROBABILITY SIE LOW/HIGH
#------------------------------------------------------
#------------------------------------------------------

ps = 8 #september, predicted time frame

#---------------------------------
#*********************************
#TOGGLE TO DESIRED INTEREST
pp = 6 #starting time frame
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
bins = bins
bin_width = bins[21]-bins[20]
probabilities = np.array(probabilityl)
total_probability= np.sum(probabilities,axis=4)
pdfs = probabilities*100


lowall = []
highall = []
for k in range(10):
    #mov_mean index
    mov_mean = [4] #index of n+1 - year average
    mov_meani = k
    mov_mean_label = mov_meani+1
    time_pred = yearpred[pp,mov_meani,initialize]
    time_pred_5 = np.arange(time_pred,time_pred+5)
    
    #data: sie_obs from input ; tt = starting year
    data = sie_obs[pp,initialize,mov_meani]
    
    #prediction for september from pp = starting time frame ; tt = starting year
    predk = prediction_array[pp,:,mov_meani,initialize]
    errork = error_array[pp,:,mov_meani,initialize]
    
    
    #expected value, i.e. the prediction from
    #pp = starting time frame ; tt = starting year
    inputdata = sie_obs_nn[pp,:,:,:]
    bink =  np.array(assign_time_series_to_bins(inputdata[:,mov_meani,initialize],bins),dtype=int)
    expected = xi[bink]
    
    
    probabilityk = probabilities[pp,mov_meani,mov_meani,initialize,:]
    total_probability= np.sum(probabilityk,axis=0)
    pdfs = probabilityk*100
    
    #climatology statistics for september
    sie_train = np.reshape(sie_training[:,13,:,mov_meani],148*165,)
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
    
    lowall.append(low)
    highall.append(high)
    
lowprob = np.array(lowall)
highprob = np.array(highall)