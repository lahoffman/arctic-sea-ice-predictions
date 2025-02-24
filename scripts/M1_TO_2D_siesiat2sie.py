#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:18:52 2024

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

#colorbars
import cmocean 

#import functions
#------------------
sys.path.append('/Users/hoffmanl/Documents/scripts/functions/')
from functions_general import ncdisp
from functions_general import movmean


#------------------------------------------------------
#------------------------------------------------------
#I . load data
#------------------------------------------------------
#------------------------------------------------------

#a. SIE
#------------------------------------------------------
loadpath_sie = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset_sie =  nc.Dataset(loadpath_sie,'r')
sie_training = np.array(dataset_sie.variables['sie_ensemble_anomaly'])
tij = np.arange(1,11)
tmj = np.arange(0,10)


#b. SIAt
#------------------------------------------------------
loadpath_area = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siat_1p25m_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM_mon2sep_gn_185001-201412.nc'
dataset_siv =  nc.Dataset(loadpath_area,'r')
siat_training = np.array(dataset_siv.variables['area_ensemble_anomaly'])


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


#training data
sie_train_data = np.concatenate((sie_training[0:ts[0],:,:,:],sie_training[te[0]:ts[1],:,:,:],sie_training[te[1]:ts[2],:,:,:],sie_training[te[2]:ts[3],:,:,:],sie_training[te[3]:,:,:,:]),axis=0)
siat_train_data = np.concatenate((siat_training[0:ts[0],:,:,:],siat_training[te[0]:ts[1],:,:,:],siat_training[te[1]:ts[2],:,:,:],siat_training[te[2]:ts[3],:,:,:],siat_training[te[3]:,:,:,:]),axis=0)


#testing data
sie_test_data = np.concatenate((sie_training[ts[0]:te[0],:,:,:],sie_training[ts[1]:te[1],:,:,:],sie_training[ts[2]:te[2],:,:,:],sie_training[ts[3]:te[3],:,:,:]),axis=0)
siat_test_data = np.concatenate((siat_training[ts[0]:te[0],:,:,:],siat_training[ts[1]:te[1],:,:,:],siat_training[ts[2]:te[2],:,:,:],siat_training[ts[3]:te[3],:,:,:]),axis=0)

#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
#III. STANDARDIZE
#------------------------------------------------------------
sigma_sie = np.nanstd(sie_train_data,axis=(0,2))[np.newaxis,:,np.newaxis,:]
miu_sie = np.nanmean(sie_train_data,axis=(0,2))[np.newaxis,:,np.newaxis,:]

sie_train_standardized = np.divide((sie_train_data-miu_sie),sigma_sie)
sie_test_standardized = np.divide((sie_test_data-miu_sie),sigma_sie)

sigma_siat = np.nanstd(siat_train_data,axis=(0,2))[np.newaxis,:,np.newaxis,:]
miu_siat = np.nanmean(siat_train_data,axis=(0,2))[np.newaxis,:,np.newaxis,:]

siat_train_standardized = np.divide((siat_train_data-miu_siat),sigma_siat)
siat_test_standardized = np.divide((siat_test_data-miu_siat),sigma_siat)

#******************************************************************************************************************************************************
#ON = NOT standardized; OFF = standardized
#train_standardized = train_data
#test_standardized = test_data
#******************************************************************************************************************************************************

sie = sie_train_standardized[:,5:,:,:]
sie_test = sie_test_standardized[:,5:,:,:]
sie_sigma_tr = sigma_sie[:,5:,:,:]
sie_miu_tr = miu_sie[:,5:,:,:]

siat = siat_train_standardized[:,5:,:,:]
siat_test = siat_test_standardized[:,5:,:,:]
siat_sigma_tr = sigma_siat[:,5:,:,:]
siat_miu_tr = miu_siat[:,5:,:,:]
#------------------------------------------------------------
#------------------------------------------------------------


#bins for sie
tij = np.arange(1,11)
tmj = np.arange(0,10)
number_of_bins = 3
sigma = np.std(sie,axis=(0,2))
binsize_factor = 2
sigma_repeated = np.tile(sigma[:,0],(number_of_bins,1))
transfer_bins_positive = binsize_factor*sigma_repeated*np.transpose((np.tile(np.arange(0,number_of_bins),(12,1))))/number_of_bins
transfer_bins_negative = -transfer_bins_positive[1:,:]
transfer_bins = np.concatenate((transfer_bins_negative[::-1],transfer_bins_positive),axis=0)
transfer_bins[0,:] = np.NINF
transfer_bins[-1,:] = np.inf
number_bins = transfer_bins.shape[0]


#bins for area (siv > threshold)
sigma2 = np.std(siat,axis=(0,2))
sigma2_repeated = np.tile(sigma[:,0],(number_of_bins,1))
transfer_bins_positive2 = binsize_factor*sigma_repeated*np.transpose((np.tile(np.arange(0,number_of_bins),(12,1))))/number_of_bins
transfer_bins_negative2 = -transfer_bins_positive[1:,:]
transfer_bins2 = np.concatenate((transfer_bins_negative2[::-1],transfer_bins_positive2),axis=0)
transfer_bins2[0,:] = np.NINF
transfer_bins2[-1,:] = np.inf
number_bins2 = transfer_bins2.shape[0]

def find_numbers_within_range(array,xo,xf):
    numbers_within_range = []

    for index, number in enumerate(array):
        if xo < number < xf:
            numbers_within_range.append((number,index))
    return numbers_within_range
            

outeri = []
for i in range(12):
    outerij = []
    for j in range(10):
        outerijh = []
        for h in range(10):
            
            #output is september, t+1
            data2 = sie[:,8,:,j]
            data_t2 = data2[:,tij[h]:]
            
            #input is other timeframes
            data1a = siat[:,i,:,j]
            data1i = sie[:,i,:,j]
            
            #input is same year for: 
            #Xyearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
            data_t1ii = []
            data_t1ai = []
            if t0[i] == 0:
                data_t1i = data1i[:,:-tij[h]]
                data_t1a = data1a[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_t1i = data1i[:,1:]
                    data_t1a = data1a[:,1:]
                else:
                    data_t1i = data1i[:,1:-tmj[h]]
                    data_t1a = data1a[:,1:-tmj[h]]
                                    
            nt = data_t1i.shape[1]
            nm = data_t1i.shape[0]
            data_t1_reshape = np.reshape(data_t1i,[nm*nt,]) #SIE input
            data2_t1_reshape = np.reshape(data_t1a,[nm*nt,]) #AREA input
            data_t2_reshape = np.reshape(data_t2,[nm*nt,]) #SIE output
                
            
            outerijhk = []
            for k in range(number_bins-1):
                bin_start = transfer_bins[k,i]
                bin_end = transfer_bins[k+1,i]
                
                #find [number and index] of data_t1_reshape in each bin
                bin_t1 = find_numbers_within_range(data_t1_reshape,bin_start,bin_end)
                bin_t1_array = np.array(bin_t1)                
                bin_t1_index = bin_t1_array[:,1].astype(int)
                
                data2_t1_at_index = data2_t1_reshape[bin_t1_index]
                inner2 = []
                for l in range(number_bins2-1):
                    bin2_start = transfer_bins2[l,i]
                    bin2_end = transfer_bins2[l+1,i]
                    
                    #find [number and index] of data2_t1_reshape within each bin
                    bin2_t1 = find_numbers_within_range(data2_t1_at_index,bin2_start,bin2_end)
                    bin2_t1_array = np.array(bin2_t1)
                    sba = bin2_t1_array.shape[0]
                    inner = []
                    if sba > 0:
                        bin2_t1_index = bin2_t1_array[:,1].astype(int) 
                        #find the number of data_t2_reshape for each data2_t1 within each bin
                        indexog = bin_t1_index[bin2_t1_index]
                        data_t2_at_index = data_t2_reshape[indexog]
                        N1 = bin2_t1_array.shape[0]
                        
                        for m in range(number_bins-1):
                            bin3_start = transfer_bins[m,i]
                            bin3_end = transfer_bins[m+1,i]
                            bin_t2 = find_numbers_within_range(data_t2_at_index,bin3_start,bin3_end)
                            bin_t2_array = np.array(bin_t2)
                            N2 = bin_t2_array.shape[0]
                            
                            transfer_var = N2/N1
                            inner.append(transfer_var)
                        
                    else:
                        for m in range(number_bins-1):
                            transfer_var = 0
                            inner.append(transfer_var)
                        
                    inner2.append(inner)
                outerijhk.append(inner2)
            outerijh.append(outerijhk)
        outerij.append(outerijh)
    outeri.append(outerij)


transfer_operator = np.array(outeri)


filepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_2D_2sigma_3bins_siextentn_areathreshold_normTR_cmip6_185001-201412.nc'
with nc.Dataset(filepath,'w') as file:
    #create dimensions
    file.createDimension('nd',transfer_operator.shape[0])
    file.createDimension('nT',transfer_operator.shape[1])
    file.createDimension('ntau',transfer_operator.shape[2])
    file.createDimension('n1',transfer_operator.shape[3])
    file.createDimension('n2',transfer_operator.shape[4])
    file.createDimension('n3',transfer_operator.shape[5])
    file.createDimension('nb',transfer_bins.shape[0])
    file.createDimension('ni',sie_miu_tr.shape[0]) #ni = 1 
    file.createDimension('nm',sie.shape[0]) #m = 148, ensemble members in training
    file.createDimension('nt',sie.shape[2]) #m = 165, timesteps
    file.createDimension('nmte',sie_test.shape[0]) #m = 148, ensemble members in training
    
    
    
    #create variables
    transferop = file.createVariable('transfer_operator','f4',('nd','nT','ntau','n1','n2','n3'))
    transferbins = file.createVariable('transfer_bins','f4',('nb','nd'))
    transferbins_sit = file.createVariable('transfer_bins2','f4',('nb','nd'))
    siemiutrain = file.createVariable('sie_miu_tr','f4',('ni','nd','ni','nT'))
    siesigmatrain = file.createVariable('sie_sigma_tr','f4',('ni','nd','ni','nT'))
    sietrain = file.createVariable('sie','f4',('nm','nd','nt','nT'))
    sietest = file.createVariable('sie_test','f4',('nmte','nd','nt','nT'))
    siatmiutrain = file.createVariable('siat_miu_tr','f4',('ni','nd','ni','nT'))
    siatsigmatrain = file.createVariable('siat_sigma_tr','f4',('ni','nd','ni','nT'))
    siattrain = file.createVariable('siat','f4',('nm','nd','nt','nT'))
    siattest = file.createVariable('siat_test','f4',('nmte','nd','nt','nT'))


    #write data to variables 
    transferop[:] = transfer_operator
    transferbins[:] = transfer_bins
    transferbins_sit[:] = transfer_bins2
    siemiutrain[:] = sie_miu_tr
    siesigmatrain[:] = sie_sigma_tr
    sietest[:] = sie_test
    sietrain[:] = sie
    siatmiutrain[:] = siat_miu_tr
    siatsigmatrain[:] = siat_sigma_tr
    siattest[:] = siat_test
    siattrain[:] = siat
    
    transferop.description = 'transfer operator for probabilities of moving from state n1 to n2 [nd x nT x ntau x n1 x n2]; nd = 4 [yearly mean, DJF mean, september, march]; nT = 1:10 year moving means ; ntau = 1-10 year prediction timesteps ; n1 x n2 = 22 phase bins'

    