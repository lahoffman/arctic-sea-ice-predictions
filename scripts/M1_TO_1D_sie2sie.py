#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:17:50 2024

@author: hoffmanl
"""

#------------------------------------------------------
#------------------------------------------------------
#change the following lines depending on the TO
#TO timeframe: 108-128 ; 171
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

#colorbars
import cmocean 

#import functions
#------------------
sys.path.append('/Users/hoffmanl/Documents/scripts/functions/')
from functions_general import ncdisp
from functions_general import movmean


#------------------------------------------------------------
#------------------------------------------------------------
#I. LOAD DATA
#yearly, DJF, september, march
#------------------------------------------------------
#load_path = '/Users/hoffmanl/Documents/data/mip46/sie_sit_siv_SM_RMMM_NORMZMOS_TSS_MM_ARCTIC.nc'
loadpath_sie = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset_sie =  nc.Dataset(loadpath_sie,'r')
years = dataset_sie.variables['unique_years']
siei = np.array(dataset_sie.variables['sie_ensemble_anomaly'])
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#II. TRAIN-VALIDATE-TEST SPLIT BY MODEL
#TVT Split: 70-15-15
#------------------------------------------------------------

ensemble_numbers = [10,40,25,33,50,10] #no members in each  model
ensemble_gt_10 = [1,2,3,4] #model index for ensemble members > 10
ensemble_cumsum = np.cumsum(ensemble_numbers)
ensemble_no_index = np.int64(np.concatenate((np.zeros(1,),ensemble_cumsum[:-1]),axis=0))
ts = ensemble_no_index[ensemble_gt_10]
te = ts+5

sie_training = siei

#training data
train_data = np.concatenate((sie_training[0:ts[0],:,:,:],sie_training[te[0]:ts[1],:,:,:],sie_training[te[1]:ts[2],:,:,:],sie_training[te[2]:ts[3],:,:,:],sie_training[te[3]:,:,:,:]),axis=0)

#testing data
test_data = np.concatenate((sie_training[ts[0]:te[0],:,:,:],sie_training[ts[1]:te[1],:,:,:],sie_training[ts[2]:te[2],:,:,:],sie_training[ts[3]:te[3],:,:,:]),axis=0)
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
#III. STANDARDIZE
#------------------------------------------------------------
sigma = np.nanstd(train_data,axis=(0,2))[np.newaxis,:,np.newaxis,:]
miu = np.nanmean(train_data,axis=(0,2))[np.newaxis,:,np.newaxis,:]

train_standardized = np.divide((train_data-miu),sigma)
test_standardized = np.divide((test_data-miu),sigma)

#******************************************************************************************************************************************************
#ON = NOT standardized; OFF = standardized
#train_standardized = train_data
#test_standardized = test_data
#******************************************************************************************************************************************************

sie = train_standardized
sigma_tr = sigma
miu_tr = miu
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#III. BINS
#------------------------------------------------------------
#bins for sie
tij = np.arange(1,11)
tmj = np.arange(0,10)
number_of_bins = 12
sigma = np.nanstd(sie,axis=(0,2))
binsize_factor = 3

#standardized cases
sigma_repeated = np.tile(sigma[:,0],(number_of_bins,1))
transfer_bins_positive = binsize_factor*sigma_repeated*np.transpose((np.tile(np.arange(0,number_of_bins),(17,1))))/number_of_bins
transfer_bins_negative = -transfer_bins_positive[1:,:]
transfer_bins = np.concatenate((transfer_bins_negative[::-1],transfer_bins_positive),axis=0)
transfer_bins[0,:] = np.NINF
transfer_bins[-1,:] = np.inf
number_bins = transfer_bins.shape[0]

#******************************************************************************************************************************************************
#ON/OFF FOR STANDARDIZED 
'''
#unstandardized case
sigma = np.nanstd(sie,axis=(0,1,2))
sigma_repeated = np.tile(sigma,(number_of_bins,1))
transfer_bins_positive = binsize_factor*sigma_repeated*np.transpose((np.tile(np.arange(0,number_of_bins),(10,1))))/number_of_bins
transfer_bins_negative = -transfer_bins_positive[1:,:]
transfer_bins = np.concatenate((transfer_bins_negative[::-1],transfer_bins_positive),axis=0)
transfer_bins[0,:] = np.NINF
transfer_bins[-1,:] = np.inf
number_bins = transfer_bins.shape[0]
'''
#******************************************************************************************************************************************************


#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#IV. MODEL FUNCTIONS
#------------------------------------------------------------
def find_numbers_within_range(array,xo,xf):
    numbers_within_range = []

    for index, number in enumerate(array):
        if xo < number < xf:
            numbers_within_range.append((number,index))
    return numbers_within_range
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#V. TRAIN TRANSFER OPERATOR
#------------------------------------------------------------

#loop through time frames
outeri = []
for i in range(17):
    
    #loop through moving mean
    outerij = []
    for j in range(10):
        
        #loop through delta t
        outerijh = []
        for h in range(10):
            
            #output is september, t+1
            data2 = sie[:,13,:,j]
            data_t2 = data2[:,tij[h]:]
            
            #input is other timeframes
            data1 = sie[:,i,:,j]
            
            #input is same year for: 
            #Xyearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
            data_t1ii = []
            if t0[i] == 0:
                data_t1 = data1[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_t1 = data1[:,1:]
                else:
                    data_t1 = data1[:,1:-tmj[h]]
                                    
            nt = data_t1.shape[1]
            nm = data_t1.shape[0]
            data_t1_reshape = np.reshape(data_t1,[nm*nt,])
            data_t2_reshape = np.reshape(data_t2,[nm*nt,])
            
            outerijhk = []
            for k in range(number_bins-1):
                bin_start = transfer_bins[k,j]
                bin_end = transfer_bins[k+1,j]
                
                #find [number and index] of data_t1_reshape in each bin
                bin_t1 = find_numbers_within_range(data_t1_reshape,bin_start,bin_end)
                bin_t1_array = np.array(bin_t1)
                N1 = bin_t1_array.shape[0]
                bin_t1_index = bin_t1_array[:,1].astype(int)
                
                
                #find the number of data_t2_reshape for each data_t1 within each bin
                bin_t2_at_index = data_t2_reshape[bin_t1_index]
                inner = []
                for m in range(number_bins-1):
                    bin2_start = transfer_bins[m,2]
                    bin2_end = transfer_bins[m+1,2]
                    bin_t2 = find_numbers_within_range(bin_t2_at_index,bin2_start,bin2_end)
                    bin_t2_array = np.array(bin_t2)
                    N2 = bin_t2_array.shape[0]
                    
                    transfer_var = N2/N1
                    inner.append(transfer_var)
                    
                
                outerijhk.append(inner)
            outerijh.append(outerijhk)
        outerij.append(outerijh)
    outeri.append(outerij)


transfer_operator = np.array(outeri)
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#VI. SAVE MODEL
#------------------------------------------------------------
#******************************************************************************************************************************************************
#ON/OFF FOR STANDARDIZED 
#filepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d2/M_transfer_operator_3sigma_sie2sie_mon2sep_normOFF_TVT_cmip6_185001-201412_vX.nc'
filepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
#******************************************************************************************************************************************************

with nc.Dataset(filepath,'w') as file:
    #create dimensions
    file.createDimension('nd',transfer_operator.shape[0]) #nd = 17, time frames
    file.createDimension('nT',transfer_operator.shape[1]) #T = 10, mov mean
    file.createDimension('ntau',transfer_operator.shape[2]) #tau = 10, lag time
    file.createDimension('n1',transfer_operator.shape[3]) #n1 = 22, no. bins
    file.createDimension('n2',transfer_operator.shape[4]) #n2 = 22, no. bins
    file.createDimension('nb',transfer_bins.shape[0]) #nb = 23, bin edges
    file.createDimension('ndi',transfer_bins.shape[1]) #ndi = nd
    file.createDimension('ni',miu_tr.shape[0]) #ni = 1 
    file.createDimension('nm',train_standardized.shape[0]) #m = 148, ensemble members in training
    file.createDimension('nt',train_standardized.shape[2]) #m = 165, timesteps
    file.createDimension('nmte',test_standardized.shape[0]) #m = 148, ensemble members in training
    
    

    
    #create variables
    transferop = file.createVariable('transfer_operator','f4',('nd','nT','ntau','n1','n2'))
    transferbins = file.createVariable('transfer_bins','f4',('nb','ndi'))
    miutrain = file.createVariable('miu_tr','f4',('ni','nd','ni','nT'))
    sigmatrain = file.createVariable('sigma_tr','f4',('ni','nd','ni','nT'))
    train = file.createVariable('train_standardized','f4',('nm','nd','nt','nT'))
    test = file.createVariable('test_standardized','f4',('nmte','nd','nt','nT'))

    #write data to variables 
    transferop[:] = transfer_operator
    transferbins[:] = transfer_bins
    miutrain[:] = miu_tr
    sigmatrain[:] = sigma_tr
    test[:] = test_standardized
    train[:] = train_standardized
    
    transferop.description = 'transfer operator for probabilities of moving from state n1 to n2 [nd x nT x ntau x n1 x n2]; nd = 4 [yearly mean, DJF mean, september, march]; nT = 1:10 year moving means ; ntau = 1-10 year prediction timesteps ; n1 x n2 = 22 phase bins'
#------------------------------------------------------------
#------------------------------------------------------------