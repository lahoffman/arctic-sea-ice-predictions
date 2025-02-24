#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:29:20 2024

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


#------------------------------------------------------------
#------------------------------------------------------------
#LOAD TRANSFER OPERATOR
#trained on 1850-2014
#------------------------------------------------------
#------------------------------------------------------

#make config files for standardized and nonstandardized
#call config file when running in terminal (i.e. python config file)
#whatever varies between runs
#******************************************************************************************************************************************************
#ON/OFF FOR STANDARDIZED TO
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
#load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normOFF_TVT_cmip6_185001-201412_vX.nc'
#******************************************************************************************************************************************************

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


#------------------------------------------------------------
#------------------------------------------------------------
#TEST DATA: OBSERVATIONS, 1979-2014
#------------------------------------------------------
#------------------------------------------------------
#[MMM, demeaned, detrended] residual from cmip6 historical (1979-2014) + residual cmip6 ssp585 (2015-2024)
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_CMIP6_nonstd_197901202412_vX.nc'


dataset =  nc.Dataset(load_path,'r')
sie_observed = np.array(dataset.variables['sie_obs'])
sie_obs_te = np.array(dataset.variables['residual'])
#sie_obs_te = np.array(dataset.variables['residual'])[5,:,:,:]

fit_obs = np.array(dataset.variables['fit'])
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])
res_std = np.nanstd(sie_obs_te,axis=2)
res_mean = np.nanmean(sie_obs_te,axis=2)


#normalize residual by mean and stdev of training data
sienostdk = []
siestdk = []
for k in range(6):
    sienostdi = []
    siestdi = []
    for i in range(17):
        sienostdj = []
        siestdj = []
        for j in range(10):
            te = sie_obs_te[k,j,i,:] 
            test = sigma_train[:,i,:,j]
            tem = miu_train[:,i,:,j]
            ted = np.divide((te-tem),test)
            
            tei = sie_obs_te[k,j,i,:]       
            siestdj.append(ted) #standardized
            sienostdj.append(tei) #non-standardized
        siestdi.append(siestdj)
        sienostdi.append(sienostdj)
    siestdk.append(siestdi)
    sienostdk.append(sienostdi)

#******************************************************************************************************************************************************
#ON/OFF for STANDARIZED OBS
sie_obsi = np.array(siestdk) #standardized
#sie_obsi = np.array(outeri)[:,:,np.newaxis,:] #non-standardized
#******************************************************************************************************************************************************

#rearrange shape of data
outouter = []
for k in range(6):
    outer = []
    for i in range(17):
        inner = []
        for j in range(47):
            te = sie_obsi[k,i,:,0,j]
            inner.append(te)
        outer.append(inner)
    outouter.append(outer)
sie_obs = np.array(outouter)   
#------------------------------------------------------------
#------------------------------------------------------------ 

#------------------------------------------------------------
#------------------------------------------------------------
#PERFORMANCE METRICS
#------------------------------------------------------
#------------------------------------------------------              
#coefficient of determination
#------------------
def coefficient_of_determination(obs,xi,pi):
    nt = obs.shape[0]
    xi = np.tile(xi,(nt,1))
    xip = np.sum(np.multiply(xi,pi),axis=1)
    return 1 - (np.divide(np.nanmean(np.square(xip-obs),axis=0),np.nanmean(np.square(obs),axis=0)))


#reliabiltiy
#------------------
def reliability(obs,xi,pi):
    nt = obs.shape[0]
    nb = pi.shape[1]
    xi = np.tile(xi,(nt,1))
    xip = np.sum(np.multiply(xi,pi),axis=1)
    numerator = np.square(xip-obs)
    denominator = np.sum(np.multiply(np.square(xi-np.transpose(np.tile(xip,(nb,1)))),pi),axis=1)
    return np.sqrt(np.nanmean(np.divide(numerator,denominator),axis=0))

#monte carlo reliabiltiy uncertainty
#------------------
def monte_carlo_reliability(obs, xi, pi, n_samples=1000, ci=0.95):
    """
    Compute Monte Carlo confidence interval for reliability.
    
    Parameters:
    obs: array-like, shape (n_obs,)
        Observed values.
    xi: array-like, shape (nbins,)
        Predicted bin centers.
    pi: array-like, shape (n_obs, nbins)
        Predicted probability distributions.
    n_samples: int, default=1000
        Number of Monte Carlo resamples.
    ci: float, default=0.95
        Confidence level (e.g., 0.95 for 95% confidence interval).
    
    Returns:
    reliability_mean: float
        Mean reliability score.
    reliability_ci: tuple (lower_bound, upper_bound)
        Confidence interval bounds.
    """
    reliability_samples = []
    n_obs = len(obs)
    
    for _ in range(n_samples):
        resample_idx = np.random.choice(n_obs, n_obs, replace=True)  # Bootstrap resampling
        reliability_samples.append(reliability(obs[resample_idx], xi, pi[resample_idx]))
    
    reliability_samples = np.array(reliability_samples)
    reliability_mean = np.mean(reliability_samples)
    reliability_std = np.std(reliability_samples)
    
    z_score = 1.96 if ci == 0.95 else 1.645 if ci == 0.90 else None
    if z_score is None:
        raise ValueError("Unsupported confidence level. Use 0.95 or 0.90.")
    
    reliability_ci = (reliability_mean - z_score * reliability_std, 
                      reliability_mean + z_score * reliability_std)
    
    return reliability_mean, reliability_ci

def monte_carlo_r2_bootstrap(obs, xi, pi, n_samples=1000, ci=0.95):
    """
    Compute Monte Carlo confidence interval for R^2 using bootstrap resampling of observations.
    
    Parameters:
    obs: array-like, shape (n_obs,)
        Observed values.
    xi: array-like, shape (nbins,)
        Predicted bin centers.
    pi: array-like, shape (n_obs, nbins)
        Predicted probability distributions.
    n_samples: int, default=1000
        Number of Monte Carlo resamples.
    ci: float, default=0.95
        Confidence level (e.g., 0.95 for 95% confidence interval).
    
    Returns:
    r2_mean: float
        Mean R^2 score.
    r2_ci: tuple (lower_bound, upper_bound)
        Confidence interval bounds.
    """
    r2_samples = []
    
    for _ in range(n_samples):
        # Bootstrap resampling of observations
        bootstrap_indices = np.random.choice(len(obs), size=len(obs), replace=True)
        bootstrap_obs = obs[bootstrap_indices]
        bootstrap_pi = pi[bootstrap_indices,:]
        
        # Compute R^2 for this resampled set
        r2_metric = coefficient_of_determination(bootstrap_obs, xi, bootstrap_pi)
        r2_samples.append(r2_metric)
    
    r2_samples = np.array(r2_samples)
    r2_mean = np.mean(r2_samples)
    
    # Compute confidence interval using percentiles (non-parametric)
    lower_bound = np.percentile(r2_samples, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(r2_samples, (1 + ci) / 2 * 100)
    r2_ci = (lower_bound, upper_bound)
    
    return r2_mean, r2_ci
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#BIN INDICES
#------------------------------------------------------------
#------------------------------------------------------------
    
#find bin indices
def find_bin_index(value, bin_boundaries):
    # Iterate through bin boundaries
    for i in range(len(bin_boundaries) - 1):
        # Check if the value falls within the current bin boundaries
        if bin_boundaries[i] <= value < bin_boundaries[i + 1]:
            return i  # Return the index of the bin
    # If the value is outside all bins, return None or -1 (depending on preference)
    return np.nan

#assign time series to bins
def assign_time_series_to_bins(time_series, bin_boundaries):
    bin_indices = []
    # Iterate through the time series
    for value in time_series:
        # Find the bin index for the current time step
        bin_index = find_bin_index(value, bin_boundaries)
        bin_indices.append(bin_index)
    return bin_indices
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
# BIN VALUES: PREDICTED POSSIBILITIES
#------------------------------------------------------------
#------------------------------------------------------------
#predicted possibilities are the OUTPUT bin values; since transferbin is the bin edges, find the middle of the bins
predicted_poss = []
for k in range(17):
    bin_means = []
    transferbin = transfer_bins[:,2]
    for i in range(len(transferbin)-1):
        bin_mean = (transferbin[i]+transferbin[i+1])/2
        bin_means.append(bin_mean)
    predicted_possibilities = np.array(bin_means)
    predicted_possibilities[0,] = transferbin[1,]-(transferbin[2,]-transferbin[1,])/2
    predicted_possibilities[21,] = transferbin[21,]+(transferbin[2,]-transferbin[1,])/2
    predicted_poss.append(predicted_possibilities)
    
xi = np.array(predicted_possibilities)
#------------------------------------------------------------
#------------------------------------------------------------

ensemble_numbers = [10,40,25,33,50,10] #no members in each  model
ensemble_numbers_tr = [10,35,20,28,45,10]
ensemble_gt_10 = [1,2,4,5,6] #model index for ensemble members > 10
ensemble_cumsum = np.cumsum(ensemble_numbers_tr)
ensemble_no_index = np.int64(np.concatenate((np.zeros(1,),ensemble_cumsum[:]),axis=0))
ts = ensemble_no_index[ensemble_gt_10]
te = ts+5

percentskillfulTO5 = []
percentskillfulTO10 = []
oneyearjulyTO_cod = []
oneyearjulyTOrel = []
oneyearjulyTO_cod_CI = []
oneyearjulyTOrel_CI = []

#******************************************************************************************************************************************************
#------------------------------------------------------
#------------------------------------------------------    
#---------PERFECT MODEL OR OBSERVATIONS ? -------------
#------------------------------------------------------  
to_codz = []
to_relz = []
to_cod_CIz = []
to_rel_CIz = []
for z in range(6):
    #select specific CMIP6 model, change ensemble_no_index index
    test_data = sie_obs[z,:,:,:]
    ntest = test_data.shape[1]*test_data.shape[2]
    #------------------------------------------------------
    #------------------------------------------------------    
    #------------------------------------------------------
    #------------------------------------------------------    
    #******************************************************************************************************************************************************


    #------------------------------------------------------------
    #------------------------------------------------------------
    # TO: PERFORMANCE AND PREDICTIONS
    #------------------------------------------------------------
    #------------------------------------------------------------

    #evaluate for [monthly] ; [T = 10 moving means] ; [tau = 10 prediction time steps]
    tij = np.arange(1,11)
    tmj = np.arange(0,10)

    #i = time frames
    cod_CIi = []
    reliab_CIi = []
    outercoeff = []
    outerreal = []
    for i in range(17):
        
        #j = moving mean
        cod_CIj = []
        reliab_CIj = []
        outercoeff1 = []
        outerreal1 = []
        for j in range(10):
            
            #h = hindcast lag
            cod_CIk = []
            reliab_CIk = []
            outercoeff2 = []
            outerreal2 = []
            for h in range(10):
                transferop = transfer_operator[i,j,h,:,:]
                transferbin = transfer_bins[:,3]
                nb = transferbin.shape[0]
                
                #output is september
                data2 = test_data[13,:,j]
                data_rm2 = data2[j:] #only where not a NaN 
                data_t2 = data_rm2[tij[h]:] 
                
                #input is other timeframes
                data1 = test_data[i,:,j]
                data_rm1 = data1[j:]
                
                #input is same year for: 
                #Xyearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
                t0 = [0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)

                if t0[i] == 0:
                    data_t1 = data_rm1[:-tij[h]]
                elif t0[i] == 1:
                    if h == 0:
                        data_t1 = data_rm1[1:]
                    else:
                        data_t1 = data_rm1[1:-tmj[h]]
                                        
             
                ni = data_t1.shape[0]
                data_t1_reshape = np.reshape(data_t1,[ni,])
                data_t2_reshape = np.reshape(data_t2,[ni,])
                
                ot = data_t2_reshape
                
                bin_index = np.array(assign_time_series_to_bins(data_t1_reshape,transferbin))
                nbi = bin_index.shape[0]
                
                probability = []
                prediction = []
                stdev = []
                for l in range(nbi):            
                    bi = bin_index[l]
                    if ~np.isnan(bi):
                        prob_k = transferop[bi.astype(int),:]
                    else: 
                        prob_k = np.full([22,], np.nan)
                        
                    predictionk = np.sum(xi*prob_k) 
                    stdevk = np.sqrt(np.sum(np.multiply(np.square(xi-predictionk),prob_k),axis=0))
                        
                    probability.append(prob_k)
                    stdev.append(stdevk)
                    prediction.append(predictionk)
                pi = np.array(probability)
                pr = np.array(prediction)
                
                
                #metrics calculated from probabilities            
                coeff_det = coefficient_of_determination(ot, xi, pi)
                reliab = reliability(ot, xi, pi)
                reliab_mean, reliab_CI = monte_carlo_reliability(ot, xi, pi)
                cod_mean, cod_CI = monte_carlo_r2_bootstrap(ot,xi,pi)
            
                cod_CIk.append(cod_CI)
                reliab_CIk.append(reliab_CI)
                outercoeff2.append(coeff_det)
                outerreal2.append(reliab)
            cod_CIj.append(cod_CIk)
            reliab_CIj.append(reliab_CIk)
            outercoeff1.append(outercoeff2)
            outerreal1.append(outerreal2)
        cod_CIi.append(cod_CIj)
        reliab_CIi.append(reliab_CIj)
        outercoeff.append(outercoeff1)
        outerreal.append(outerreal1)
            
    to_cod = np.array(outercoeff)[5:,:,:]
    to_rel = np.array(outerreal)[5:,:,:]
    to_rel_CI = np.array(reliab_CIi)[5:,:,:,:]
    to_cod_CI = np.array(cod_CIi)[5:,:,:,:]

 
    #------------------------------------------------------------
    #------------------------------------------------------------



    #------------------------------------------------------------
    #------------------------------------------------------------
    #PERSISTENCE PERFORMANCE
    #------------------------------------------------------------
    #------------------------------------------------------------

    #coefficient determination, persistence
    def r_squared(obs,model):
        return 1 - (np.divide(np.nanmean(np.square(model-obs),axis=0),np.nanmean(np.square(obs),axis=0)))

    #correlation / acc, persistence
    def corr_ps(y_true_obs,y_pred_obs):
        return (np.nansum((y_true_obs-np.nanmean(y_true_obs))*(y_pred_obs-np.nanmean(y_pred_obs))))/((np.sqrt(np.nansum(np.square(y_true_obs-np.nanmean(y_true_obs)))))*(np.sqrt(np.nansum(np.square(y_pred_obs-np.nanmean(y_pred_obs))))))

    #rmse persistence
    def rmse_ps(y_true_obs,y_pred_obs):
        return np.sqrt(np.divide(np.nansum(np.square(y_pred_obs-y_true_obs)),nmt))


    tmj = np.arange(0,10)
    persistence_rsquaredi = []
    persistence_rmsei = []
    persistence_corri = []
    for i in range(17):
        
        persistence_rsquaredj = []
        persistence_rmsej = []
        persistence_corrj = []
        for j in range(10):
            
            persistence_rsquaredh = []
            persistence_rmseh = []
            persistence_corrh = []
            for h in range(1,11):
                nt2 = test_data.shape[1]
                        
                #output is september
                persistence_output = np.reshape(test_data[13,h:,j],(nt2-h))
                
                #input is same year for: 
                #Xyearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
                t0 = [0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)

                if t0[i] == 0:
                    ps_t1 = test_data[i,:-h,j]
                elif t0[i] == 1:
                    if h == 1:
                        ps_t1 = test_data[i,1:,j]
                    else:
                        ps_t1 = test_data[i,1:-tmj[h-1],j]
            
                persistence_input = np.reshape(ps_t1,(nt2-h))        
                persistence_rsquared = r_squared(persistence_input,persistence_output)
        
                persistence_rsquaredh.append(persistence_rsquared)
            persistence_rsquaredj.append(persistence_rsquaredh)
        persistence_rsquaredi.append(persistence_rsquaredj)


            
    persistence_cod = np.array(persistence_rsquaredi)[5:,:,:]
    #------------------------------------------------------------
    #------------------------------------------------------------


    #------------------------------------------------------------
    #------------------------------------------------------------
    #FIGURES: MODEL PERFORMANCE VS LAG TIME
    #------------------------------------------------------------
    #------------------------------------------------------------

    #plot time frame versus averaging time
    #xaxis is time frame (JAN-DEC)
    #each figure is a different hindcast lag
    #-----------------------------------------------
    #-----------------------------------------------
    mm1 = 0 #JAN = 5
    mm2 = 12 #DEC = 17

    xis = [] 
    for i in range(10):
        
        #plot coefficient of determination
        #------------------------------------------------------
        #------------------------------------------------------ 
        ff = np.arange(1,11)
        xtot = np.arange(1,121)
        xxi = xtot[(12*i):12*(i+1)]
        xx = np.flip(xxi)
        
        custom_ticks1 = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]  # Example custom tick positions
        custom_ticks2 = [0.0, 0.25, 0.5, 0.75, 1.0,1.25,1.5,1.75,2.0]
        
        
        # coefficient of determination: Create filled contour plots for each subplot
        contour_levels = np.linspace(-1, 1, 50)  # Adjust levels as needed
        data1 = np.transpose(to_cod[mm1:mm2,:,i])
        persistence = np.transpose(persistence_cod[mm1:mm2,:,i])
        maski = data1 > persistence
        data_rel = np.transpose(to_rel[mm1:mm2,:,i])
        data_corr = np.transpose(to_cod[mm1:mm2,:,i])
        data_rel_CI_low = np.round(np.transpose(to_rel_CI[mm1:mm2,:,i,0]),1)
        data_rel_CI_high = np.round(np.transpose(to_rel_CI[mm1:mm2,:,i,1]),1)
        data_cod_CI_low = np.round(np.transpose(to_cod_CI[mm1:mm2,:,i,0]),1)
        data_cod_CI_high = np.transpose(to_cod_CI[mm1:mm2,:,i,1])
        
        mask = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high)) & (data_cod_CI_low > 0) & (data_corr > persistence)
        X, Y = np.meshgrid(xx,ff)
        num_elements_less_than_minus_one = np.sum(data1 < -1)
        masked_Z = np.ma.masked_less(data1, -1)
        maskcod = (data_cod_CI_low > 0) & (data_corr > persistence)
        maskrel = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high))
        
        data11 = np.concatenate((data1[:,8:12],data1[:,0:8]),axis=1)
        maski11 = np.concatenate((maski[:,8:12],maski[:,0:8]),axis=1)
        mask11 = np.concatenate((mask[:,8:12],mask[:,0:8]),axis=1) 
        maskcod11 = np.concatenate((maskcod[:,8:12],maskcod[:,0:8]),axis=1)
        maskrel11 = np.concatenate((maskrel[:,8:12],maskrel[:,0:8]),axis=1)
        
        if i == 0:
            d1 = np.flip(data11,axis=1)
            mi1 = np.flip(maski11,axis=1)
            m1 = np.flip(mask11,axis=1)
            mc1 = np.flip(maskcod11,axis=1)
            mr1 = np.flip(maskrel11,axis=1)
        else:
            d1 = np.append(d1,np.flip(data11,axis=1),axis=1)
            mi1 = np.append(mi1,np.flip(maski11,axis=1),axis=1)
            m1 = np.append(m1,np.flip(mask11,axis=1),axis=1)
            mc1 = np.append(mc1,np.flip(maskcod11,axis=1),axis=1)
            mr1 = np.append(mr1,np.flip(maskrel11,axis=1),axis=1)
        
            
        #reliability
        contour_levels = np.linspace(0, 2, 30)  # Adjust levels as needed
        data2 = np.transpose(to_rel[mm1:mm2,:,i])
        num_elements_greater_than_two = np.sum(data2 > 2)
        masked_Z = np.ma.masked_greater(data2, 2)
        data_rel = np.transpose(to_rel[mm1:mm2,:,i])
        data_corr = np.transpose(to_cod[mm1:mm2,:,i])
        persistence = np.transpose(persistence_cod[mm1:mm2,:,i])
        mask = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high)) & (data_cod_CI_low > 0) & (data_corr > persistence)
        maskint = mask.astype(int)
        X, Y = np.meshgrid(xx,ff)
        maskcod = (data_cod_CI_low > 0) & (data_corr > persistence)
        maskrel = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high))
        
        data22 = np.concatenate((data2[:,8:12],data2[:,0:8]),axis=1)
        mask22 = np.concatenate((mask[:,8:12],mask[:,0:8]),axis=1) 
        maskcod22 = np.concatenate((maskcod[:,8:12],maskcod[:,0:8]),axis=1)
        maskrel22 = np.concatenate((maskrel[:,8:12],maskrel[:,0:8]),axis=1)

        if i == 0:
            d2 = np.flip(data22,axis=1)
            m2 = np.flip(mask22,axis=1)
            mc2 = np.flip(maskcod22,axis=1)
            mr2 = np.flip(maskrel22,axis=1)
        else:
            d2= np.append(d2,np.flip(data22,axis=1),axis=1)
            m2= np.append(m2,np.flip(mask22,axis=1),axis=1)
            mc2 = np.append(mc2,np.flip(maskcod22,axis=1),axis=1)
            mr2 = np.append(mr2,np.flip(maskrel22,axis=1),axis=1)

        xis.append(xx)


    ticks = ['S','O','N','D','J','F','M','A','M','J','J','A']
    tickflip = np.flip(ticks)
    ticki = np.tile(tickflip,10)
    aa = []
    for i in range(1,10):
        tickflip[-1,] = f'S\nt-{i}'
        aa.append(tickflip)
    ab = np.array(aa)
    xxis = np.reshape(np.array(xis),[120,])
    data11 = d1
    maski11 = mi1
    mask11 = m1
    maskcod11 = mc1
    ff = np.arange(1,11)
    xx = np.arange(120)
    label_small = 90
    label_big = 140
    label_mid = 90
    nmo = 60
    
    fig, axs = plt.subplots(2,1, figsize=(200,50))

    #coefficient of determination
    contour_levels = np.linspace(-1, 1, 50)  # Adjust levels as needed
    X, Y = np.meshgrid(xx,ff)
    num_elements_less_than_minus_one = np.sum(data11 < -1)
    masked_Z = np.ma.masked_less(data11, -1)
    cp1 = axs[0].contourf(xx, ff, data11, levels=contour_levels, cmap='RdGy_r')
    ch1 = axs[0].contourf(xx, ff, maski11, levels=[0, 0.5], colors='none',hatches=['', '|'],alpha=0.3) 
    axs[0].scatter(X[maskcod11], Y[maskcod11], s=100,color='black')
    axs[0].scatter(X[mask11], Y[mask11], s=300,color='maroon')

    if num_elements_less_than_minus_one > 0:
        axs[0].contourf(xx,ff, data11, levels=[data11.min(), -1], colors='dimgrey')

    cb1 = fig.colorbar(cp1, ax=axs[0],aspect=10)
    cb1.set_label('coefficient of determination', fontsize=label_mid) 
    cb1.set_ticks(custom_ticks1)
    cb1.ax.tick_params(labelsize=label_small)
    axs[0].set_xticks(xx)
    axs[0].set_xticklabels(ticki,fontsize=label_small)
    axs[0].set_yticklabels(np.arange(1,11),fontsize=label_small)


    data22 = d2
    mask22 = m2
    maskrel22 = mr2

    #reliability
    contour_levels = np.linspace(0, 2, 30)  # Adjust levels as needed
    num_elements_greater_than_two = np.sum(data22 > 2)
    cp2 = axs[1].contourf(xx, ff, data22, levels=contour_levels, cmap='BrBG_r',vmin=0, vmax=2)
    axs[1].contour(xx,ff,data22,levels = [0.5],colors='darkolivegreen')
    axs[1].contour(xx,ff,data22,levels = [1.5],colors='peru')
    axs[1].scatter(X[maskrel22], Y[maskrel22], s=100,color='black')
    axs[1].scatter(X[mask22], Y[mask22],s=300, color='peru')
    if num_elements_greater_than_two > 0:
        axs[1].contourf(xx,ff, data22, levels=[2,data22.max()], colors='saddlebrown')
        
    cb2 = fig.colorbar(cp2, ax=axs[1],aspect=10)
    cb2.set_label('reliability', fontsize=label_mid) 
    cb2.set_ticks(custom_ticks2)
    cb2.ax.tick_params(labelsize=label_small)
    axs[1].set_xticks(xx)
    axs[1].set_xticklabels(ticki,fontsize=label_small)

    axs[1].set_yticklabels(np.arange(1,11),fontsize=label_small)
    axs[1].set_ylabel('                                      averaging time [year]', fontsize=label_big)
    axs[1].set_xlabel('hindcast lag [previous month]', fontsize=label_big,labelpad=100)
    #------------------------------------------------------------
    #------------------------------------------------------------

    percentskillfulTOi5 = np.sum(mask11[:,0:60])/600
    percentskillfulTO5.append(percentskillfulTOi5)
    percentskillfulTOi10 = np.sum(mask11)/1200
    percentskillfulTO10.append(percentskillfulTOi10)
    
    oneyearjulyTO_codi = to_cod[6,0,0]
    oneyearjulyTO_cod.append(oneyearjulyTO_codi)
    oneyearjulyTO_cod_CIi = to_cod_CI[6,0,0,:]
    oneyearjulyTO_cod_CI.append(oneyearjulyTO_cod_CIi)
    
    oneyearjulyTOreli = to_rel[6,0,0]
    oneyearjulyTOrel.append(oneyearjulyTOreli)
    oneyearjulyTOrel_CIi = to_rel_CI[6,0,0,:]
    oneyearjulyTOrel_CI.append(oneyearjulyTOrel_CIi)
    
    to_codz.append(to_cod)
    to_relz.append(to_rel)
    to_cod_CIz.append(to_cod_CI)
    to_rel_CIz.append(to_rel_CI)
    
    
    
#------------------------------------------------------------
#------------------------------------------------------------
#SAVE PREDICTION (PROBABILSTIC AND DETERMINISTIC) AND PERFORMANCE TO NETCDF
#------------------------------------------------------------
#------------------------------------------------------------
percentskillfulTO5 = np.array(percentskillfulTO5)
percentskillfulTO10 = np.array(percentskillfulTO10)
oneyearjulyTO_cod = np.array(oneyearjulyTO_cod)
oneyearjulyTOrel = np.array(oneyearjulyTOrel)
oneyearjulyTO_cod_CI = np.array(oneyearjulyTO_cod_CI)
oneyearjulyTOrel_CI = np.array(oneyearjulyTOrel_CI)
    
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/TO_sie2sie_performance_perfect_CMIP6_CI.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('nm',percentskillfulTO5.shape[0]) #no. models
    file.createDimension('tf',to_rel.shape[1]) #time frames
    file.createDimension('T',to_rel.shape[2]) #mov mean, T
    file.createDimension('tau',to_rel.shape[3]) #lag time, tau
    file.createDimension('nt',to_rel.shape[4]) #timesteps, nt
    file.createDimension('hl',to_rel_CI.shape[5]) #CI, high & low
    
    #create variables
    #TO performance
    pctTO5 = file.createVariable('percentskillfulTO5','f4',('nm')) 
    pctTO10 = file.createVariable('percentskillfulTO10','f4',('nm')) 
    codTOj = file.createVariable('oneyearjulyTO_cod','f4',('nm'))
    relTOj = file.createVariable('oneyearjulyTO_rel','f4',('nm'))  
    codTOCIj = file.createVariable('oneyearjulyTO_cod','f4',('nm','hl'))
    relTOCIj = file.createVariable('oneyearjulyTO_rel','f4',('nm','hl')) 
    relTO = file.createVariable('to_rel','f4',('nm','tf','T','tau','nt'))
    codTO = file.createVariable('to_cod','f4',('nm','tf','T','tau','nt'))
    relTOCI = file.createVariable('to_rel_CI','f4',('nm','tf','T','tau','nt','hl'))
    codTOCI = file.createVariable('to_cod_CI','f4',('nm','tf','T','tau','nt','hl'))
    
    #write data to variables
    pctTO5[:] = percentskillfulTO5
    pctTO10[:] = percentskillfulTO10
    codTOj[:] = oneyearjulyTO_cod
    relTOj[:] = oneyearjulyTOrel
    codTOCIj[:] = oneyearjulyTO_cod_CI
    relTOCIj[:] = oneyearjulyTOrel_CI
    relTO[:] = to_rel
    codTO[:] = to_cod
    relTOCI[:] = to_rel_CI
    codTOCI[:] = to_cod_CI

#------------------------------------------------------------
#------------------------------------------------------------