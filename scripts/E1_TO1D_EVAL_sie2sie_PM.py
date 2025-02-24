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
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'


dataset =  nc.Dataset(load_path,'r')
sie_observed = np.array(dataset.variables['sie_obs'])
sie_obs_te = np.array(dataset.variables['residual_mean_weighted'])
#sie_obs_te = np.array(dataset.variables['residual'])[5,:,:,:]

fit_obs = np.array(dataset.variables['fit_mean_weighted'])
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])
res_std = np.nanstd(sie_obs_te,axis=2)
res_mean = np.nanmean(sie_obs_te,axis=2)

#no CanESM5
#sie_obs_te = np.array(dataset.variables['residual_mean_weighted_noCanESM'])
#fit_obs = np.array(dataset.variables['fit_mean_weighted_noCanESM'])

#normalize residual by mean and stdev of training data
outer = []
outeri = []
for i in range(17):
    inner = []
    inneri = []
    for j in range(10):
        te = sie_obs_te[j,i,:] 
        test = sigma_train[:,i,:,j]
        tem = miu_train[:,i,:,j]
        ted = np.divide((te-tem),test)
        
        tei = sie_obs_te[j,i,:]       
        inner.append(ted) #standardized
        inneri.append(tei) #non-standardized
    outer.append(inner)
    outeri.append(inneri)

#******************************************************************************************************************************************************
#ON/OFF for STANDARIZED OBS
sie_obsi = np.array(outer) #standardized
#sie_obsi = np.array(outeri)[:,:,np.newaxis,:] #non-standardized
#******************************************************************************************************************************************************

#rearrange shape of data
outer = []
for i in range(17):
    inner = []
    for j in range(47):
        te = sie_obsi[i,:,0,j]
        inner.append(te)
    outer.append(inner)
sie_obs = np.array(outer)   
#------------------------------------------------------------
#------------------------------------------------------------ 


#******************************************************************************************************************************************************
#------------------------------------------------------
#------------------------------------------------------    
#---------PERFECT MODEL OR OBSERVATIONS ? -------------
#------------------------------------------------------    

#use only some ensemble members from each CMIP6 model for 'perfect' case
#test_data = sie_obs[:,np.newaxis,:,:] #evaluate transfer operator on observations
#test_data = sie_training
test_data = sie_testing

#select specific CMIP6 model, change ensemble_no_index index
#test_data = sie_training[:,ensemble_no_index[1]:ensemble_no_index[2],:,:]

#use ALL training data for 'perfect' case
#test_data = sie_training

ntest = test_data.shape[1]*test_data.shape[2]
#------------------------------------------------------
#------------------------------------------------------    
#------------------------------------------------------
#------------------------------------------------------    
#******************************************************************************************************************************************************

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



#------------------------------------------------------------
#------------------------------------------------------------
# TO: PERFORMANCE AND PREDICTIONS
#------------------------------------------------------------
#------------------------------------------------------------

#evaluate for [monthly] ; [T = 10 moving means] ; [tau = 10 prediction time steps]
tij = np.arange(1,11)
tmj = np.arange(0,10)

#i = time frames
stdevi = []
predictioni = []
obsi = []
probi = []
acci = []
rmsei = []
outercoeff = []
outerreal = []
for i in range(17):
    
    #j = moving mean
    stdevj = []
    predictionj = []
    obsj = []
    probj = []
    accj = []
    rmsej = []
    outercoeff1 = []
    outerreal1 = []
    for j in range(10):
        
        #h = hindcast lag
        stdevh = []
        predictionh = []
        obsh = []
        probh = []
        acch = []
        rmseh = []
        outercoeff2 = []
        outerreal2 = []
        for h in range(10):
            transferop = transfer_operator[i,j,h,:,:]
            transferbin = transfer_bins[:,3]
            nb = transferbin.shape[0]
            
            #output is september
            data2 = test_data[13,:,:,j]
            data_rm2 = data2[:,j:] #only where not a NaN 
            data_t2 = data_rm2[:,tij[h]:] 
            
            #input is other timeframes
            data1 = test_data[i,:,:,j]
            data_rm1 = data1[:,j:]
            
            #input is same year for: 
            #Xyearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)

            if t0[i] == 0:
                data_t1 = data_rm1[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_t1 = data_rm1[:,1:]
                else:
                    data_t1 = data_rm1[:,1:-tmj[h]]
                                    
            nt = data_t1.shape[1]
            ni = data_t1.shape[0]
            data_t1_reshape = np.reshape(data_t1,[ni*nt,])
            data_t2_reshape = np.reshape(data_t2,[ni*nt,])
               
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
            
            #metrics calculated from predictions
            nmt = ni*nt
            acc_obs = np.divide(np.nansum(np.multiply((pr-np.nanmean(pr)),(ot-np.nanmean(ot)))),np.multiply(np.sqrt(np.nansum(np.square(pr-np.nanmean(pr)))),np.sqrt(np.nansum(np.square(ot-np.nanmean(ot))))))     
            rmse_obs = np.sqrt(np.divide(np.nansum(np.square(pr-ot)),nmt))
            
            
            nop = np.array(prediction).shape[0]
            if nop < ntest:
                nanfill = np.full(ntest-nop,np.nan)
                nanfillpr = np.full((ntest-nop,22),np.nan)
                otn = np.concatenate((nanfill,ot),0)
                predictionn = np.concatenate((nanfill,prediction),0)
                stdevn = np.concatenate((nanfill,stdev),0)
                probabilityn = np.concatenate((nanfillpr,probability),0)
            else:
                stdevn = stdev
                predictionn = prediction
                otn = ot
                probabilityn = probability
            
            
            stdevh.append(stdevn)
            predictionh.append(predictionn)
            obsh.append(otn)
            probh.append(probabilityn)
            acch.append(acc_obs)
            rmseh.append(rmse_obs)
            outercoeff2.append(coeff_det)
            outerreal2.append(reliab)
        stdevj.append(stdevh)
        predictionj.append(predictionh) 
        obsj.append(obsh)
        probj.append(probh)
        accj.append(acch)
        rmsej.append(rmseh)
        outercoeff1.append(outercoeff2)
        outerreal1.append(outerreal2)
    stdevi.append(stdevj)
    predictioni.append(predictionj)
    obsi.append(obsj)
    probi.append(probj)
    acci.append(accj)
    rmsei.append(rmsej)
    outercoeff.append(outercoeff1)
    outerreal.append(outerreal1)
        
coefficient_of_determ = np.array(outercoeff)
reliability = np.array(outerreal)
to_acc = np.array(acci)
to_rmse = np.array(rmsei)
to_cod = coefficient_of_determ
to_rel = reliability

to_input = np.array(obsi)
to_prediction_mean = np.array(predictioni)
to_prediction_err = np.array(stdevi)
to_probability = np.array(probi)
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
            nt1 = test_data.shape[1]
            nt2 = test_data.shape[2]
                       
            #output is september
            persistence_output = np.reshape(test_data[13,:,h:,j],(np.multiply(nt1,[nt2-h])))
            
            #input is same year for: 
            #Xyearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)

            if t0[i] == 0:
                ps_t1 = test_data[i,:,:-h,j]
            elif t0[i] == 1:
                if h == 1:
                    ps_t1 = test_data[i,:,1:,j]
                else:
                    ps_t1 = test_data[i,:,1:-tmj[h-1],j]
           
            persistence_input = np.reshape(ps_t1,(np.multiply(nt1,[nt2-h])))        
            persistence_rsquared = r_squared(persistence_input,persistence_output)
            persistence_rmse = rmse_ps(persistence_input,persistence_output)
            persistence_corr = corr_ps(persistence_input,persistence_output)
          
            persistence_rsquaredh.append(persistence_rsquared)
            persistence_corrh.append(persistence_corr)
            persistence_rmseh.append(persistence_rmse)
        persistence_rsquaredj.append(persistence_rsquaredh)
        persistence_corrj.append(persistence_corrh)
        persistence_rmsej.append(persistence_rmseh)
    persistence_rsquaredi.append(persistence_rsquaredj)
    persistence_corri.append(persistence_corrj)
    persistence_rmsei.append(persistence_rmsej)

        
persistence_cod = np.array(persistence_rsquaredi)
persistence_acc = np.array(persistence_corri)
persistence_rmse = np.array(persistence_rmsei)
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
mm1 = 5 #JAN = 5
mm2 = 17 #DEC = 17

# Create a figure and gridspec layout
#fig, axs = plt.subplots(2,1, figsize=(50, 10))

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
    data1 = np.transpose(coefficient_of_determ[mm1:mm2,:,i])
    persistence = np.transpose(persistence_cod[mm1:mm2,:,i])
    maski = data1 > persistence
    data_rel = np.transpose(reliability[mm1:mm2,:,i])
    data_corr = np.transpose(coefficient_of_determ[mm1:mm2,:,i])
    
    mask = ((0.8 < data_rel) & (data_rel < 1.2)) & (data_corr > 0.1) & (data_corr > persistence)
    X, Y = np.meshgrid(xx,ff)
    num_elements_less_than_minus_one = np.sum(data1 < -1)
    masked_Z = np.ma.masked_less(data1, -1)
    
    data11 = np.concatenate((data1[:,8:12],data1[:,0:8]),axis=1)
    maski11 = np.concatenate((maski[:,8:12],maski[:,0:8]),axis=1)
    mask11 = np.concatenate((mask[:,8:12],mask[:,0:8]),axis=1) 
    
    if i == 0:
        d1 = np.flip(data11,axis=1)
        mi1 = np.flip(maski11,axis=1)
        m1 = np.flip(mask11,axis=1)
    else:
        d1 = np.append(d1,np.flip(data11,axis=1),axis=1)
        mi1 = np.append(mi1,np.flip(maski11,axis=1),axis=1)
        m1 = np.append(m1,np.flip(mask11,axis=1),axis=1)
    
    
    
    
    #cp1 = axs[0].contourf(xx, ff, data11, levels=contour_levels, cmap='RdGy_r')
    #ch1 = axs[0].contourf(xx, ff, maski11, levels=[0, 0.5], colors='none',hatches=['', '|'],alpha=0.3) 
    #axs[0].contour(xx,ff,data11,levels=[0.4],colors='maroon')
    #axs[0].scatter(X[mask11], Y[mask11], color='maroon')
    #if num_elements_less_than_minus_one > 0:
    #    axs[0].contourf(xx,ff, data11, levels=[data11.min(), -1], colors='dimgrey')

          
    #reliability
    contour_levels = np.linspace(0, 2, 30)  # Adjust levels as needed
    data2 = np.transpose(reliability[mm1:mm2,:,i])
    num_elements_greater_than_two = np.sum(data2 > 2)
    masked_Z = np.ma.masked_greater(data2, 2)
    data_rel = np.transpose(reliability[mm1:mm2,:,i])
    data_corr = np.transpose(coefficient_of_determ[mm1:mm2,:,i])
    persistence = np.transpose(persistence_cod[mm1:mm2,:,i])
    mask = ((0.8 < data_rel) & (data_rel < 1.2)) & (data_corr > 0.1) & (data_corr > persistence)
    maskint = mask.astype(int)
    X, Y = np.meshgrid(xx,ff)
    
    data22 = np.concatenate((data2[:,8:12],data2[:,0:8]),axis=1)
    mask22 = np.concatenate((mask[:,8:12],mask[:,0:8]),axis=1) 
    
    if i == 0:
        d2 = np.flip(data22,axis=1)
        m2 = np.flip(mask22,axis=1)
    else:
        d2= np.append(d2,np.flip(data22,axis=1),axis=1)
        m2= np.append(m2,np.flip(mask22,axis=1),axis=1)
    
    #cp2 = axs[1].contourf(xx, ff, data22, levels=contour_levels, cmap='BrBG_r',vmin=0, vmax=2)
    #axs[1].contour(xx,ff,data22,levels = [0.5],colors='darkolivegreen')
    #axs[1].contour(xx,ff,data22,levels = [1.5],colors='peru')
    #axs[1].scatter(X[mask22], Y[mask22], color='peru')
    #if num_elements_greater_than_two > 0:
    #    axs[1].contourf(xx,ff, data22, levels=[2,data22.max()], colors='saddlebrown')
        
    xis.append(xx)





#ticks = ['SEP','OCT','NOV','DEC','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG']
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
ff = np.arange(1,11)
xx = np.arange(120)
label_small = 90
label_big = 140
label_mid = 90
 
fig, axs = plt.subplots(2,1, figsize=(200,50))

#coefficient of determination
contour_levels = np.linspace(-1, 1, 50)  # Adjust levels as needed
X, Y = np.meshgrid(xx,ff)
num_elements_less_than_minus_one = np.sum(data11 < -1)
masked_Z = np.ma.masked_less(data11, -1)
cp1 = axs[0].contourf(xx, ff, data11, levels=contour_levels, cmap='RdGy_r')
ch1 = axs[0].contourf(xx, ff, maski11, levels=[0, 0.5], colors='none',hatches=['', '|'],alpha=0.3) 
#axs[0].contour(xx,ff,data11,levels=[0.4],colors='maroon')
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

#reliability
contour_levels = np.linspace(0, 2, 30)  # Adjust levels as needed
num_elements_greater_than_two = np.sum(data22 > 2)
cp2 = axs[1].contourf(xx, ff, data22, levels=contour_levels, cmap='BrBG_r',vmin=0, vmax=2)
axs[1].contour(xx,ff,data22,levels = [0.5],colors='darkolivegreen')
axs[1].contour(xx,ff,data22,levels = [1.5],colors='peru')
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

percentskillfulTO_5yr = np.sum(mask11[:,0:60])/600
percentskillfulTO_10yr = np.sum(mask11)/1200
oneyearjulyTO_cod = coefficient_of_determ[11,0,0]
oneyearjulyTOrel = reliability[11,0,0]

#------------------------------------------------------------
#------------------------------------------------------------
#SAVE PREDICTION (PROBABILSTIC AND DETERMINISTIC) AND PERFORMANCE TO NETCDF
#------------------------------------------------------------
#------------------------------------------------------------

    
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/TO_sie2sie_performance_perfect_CMIP6_ALL.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('nm',percentskillfulTO_5yr.shape[0]) #no. time frames

    #create variables
    #TO performance
    pctTO5 = file.createVariable('percentskillfulTO_5yr','f4',('nm')) 
    pctTO10 = file.createVariable('percentskillfulTO_10yr','f4',('nm')) 
    codTO = file.createVariable('oneyearjulyTO_cod','f4',('nm'))
    relTO = file.createVariable('oneyearjulyTO_rel','f4',('nm'))  

    #write data to variables
    pctTO5[:] = percentskillfulTO_5yr
    pctTO10[:] = percentskillfulTO_10yr
    codTO[:] = oneyearjulyTO_cod
    relTO[:] = oneyearjulyTOrel

#------------------------------------------------------------
#------------------------------------------------------------

'''
#------------------------------------------------------------
#------------------------------------------------------------
#SAVE PREDICTION (PROBABILSTIC AND DETERMINISTIC) AND PERFORMANCE TO NETCDF
#------------------------------------------------------------
#------------------------------------------------------------
#savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/TO_sie2sie_performance_perfect_PM_testdata.nc'
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/TO_sie2sie_performance_obs.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',coefficient_of_determ.shape[0]) #no. time frames
    file.createDimension('ntau',coefficient_of_determ.shape[1]) #hindcast lag
    file.createDimension('nT',coefficient_of_determ.shape[2]) #no. moving means
    file.createDimension('nt',to_prediction_mean.shape[3]) #no. timesteps
    file.createDimension('nb',to_probability.shape[4]) #no. bins
  

    #create variables
    #TO performance
    acc = file.createVariable('to_acc','f4',('tf','nT','ntau')) 
    rmse = file.createVariable('to_rmse','f4',('tf','nT','ntau'))
    coeffdet = file.createVariable('to_cod','f4',('tf','nT','ntau'))  
    rel = file.createVariable('to_rel','f4',('tf','nT','ntau'))
    
    #persistence performance
    accps = file.createVariable('persistence_acc','f4',('tf','nT','ntau')) 
    codps = file.createVariable('persistence_cod','f4',('tf','nT','ntau')) 
    rmseps = file.createVariable('persistence_rmse','f4',('tf','nT','ntau')) 
    
    #model input and output
    inpt = file.createVariable('to_input','f4',('tf','ntau','nT','nt'))
    pred = file.createVariable('to_prediction_mean','f4',('tf','ntau','nT','nt'))
    prederr = file.createVariable('to_prediction_err','f4',('tf','ntau','nT','nt'))
    prob = file.createVariable('to_probability','f4',('tf','ntau','nT','nt','nb'))

    #write data to variables
    acc[:] = to_acc
    rmse[:] = to_rmse
    coeffdet[:] = to_cod
    rel[:] = to_rel
    
    accps[:] = persistence_acc
    codps[:] = persistence_cod
    rmseps[:] = persistence_rmse
    
    inpt[:] = to_input
    pred[:] = to_prediction_mean
    prederr[:] = to_prediction_err
    prob[:] = to_probability
#------------------------------------------------------------
#------------------------------------------------------------
'''