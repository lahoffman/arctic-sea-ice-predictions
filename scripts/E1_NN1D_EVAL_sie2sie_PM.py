#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:24:33 2024

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
# I. TRAINING DATA: CMIP6 data 
#------------------------------------------------------------
#------------------------------------------------------------

# a. SIEXTENTN
#------------------------------------------------------------
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset =  nc.Dataset(loadpath,'r')
years= dataset.variables['unique_years']
sie_tr = np.array(dataset.variables['sie_ensemble_anomaly'])

#------------------------------------------------------
#------------------------------------------------------
# LOAD NN PERFORMANCE
#------------------------------------------------------
#------------------------------------------------------
loadpath_nn= '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_1D_sie2sie_performance.nc' #2D NN
#loadpath_nn = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d1/NN_sie2sie_performance.nc' #1D NN
dataset =  nc.Dataset(loadpath_nn,'r')
pred_nn = dataset.variables['model_prediction']
true_nn = dataset.variables['model_input']
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------------
# V. DEFINE PREDICTION TIME
#------------------------------------------------------------
#------------------------------------------------------------
#time frames: yearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
current_date = datetime.now()
current_year = current_date.year
current_month = current_date.month
end_year = current_year
start_year = 1979

years = []
yearsm = []
for i in range(10):
    yearsim = np.arange(start_year+i,end_year+i)
    yearsi = np.arange(start_year+i+1,end_year+i+1)
    
    yearsm.append(yearsim)
    years.append(yearsi)
    
yearspred = np.array(years)
yearspredm = np.array(yearsm)

yp = []
for i in range(12):
    if t0[i] == 0:
        yearpred = yearspred

    else:
        yearpred = yearspredm
    yp.append(yearpred)
yearpred = np.array(yp)
#------------------------------------------------------------
#------------------------------------------------------------

#FUZZY CLASSIFICATION BINS FOR OUTPUT
#----------------------------------------
#step 1: define bins
def define_fuzzyBin(data,numberOfBins,binSizeFactor):
    
    #build bins based on standar deviation, size factor and number of bins
    dataSep = data[:,13,:,:]
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

#step2: convert data to fuzzy bins
#----------------------------------------
def convert_fuzzyBins(data,binCenters,sigmaG=0.5):
    probabilities = np.exp(-0.5*((data[:,None]-binCenters)/sigmaG)**2)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    
    return probabilities


#a. define fuzzy bins for training data
#----------------------------------------
data = sie_tr
datafile = np.load("/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_1D_sie2sie/model_output/bin_centers.npz")
binCenters = datafile["bins"]

#------------------------------------------------------------
#------------------------------------------------------------
# VI. PROBABILISTIC PERFORMANCE METRICS
#------------------------------------------------------------
#------------------------------------------------------------
#coefficient of determination
#------------------------------
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

nmt = pred_nn.shape[3]
ntest = pred_nn.shape[3]

#save model output, observations
#------------------------------
ypp = []
yppe = []
yprob = []
ytt =[]
skillk = []
corrk = []
acck = []
rmsek = []
relk = []
codk = []

#time frames, tf
for k in range(12):
    ypouter = []
    ypeouter = []
    ytouter = []
    yprobouter = []
    skilli = []
    corri = []
    acci = []
    rmsei = []
    reli = []
    codi = []
        
    #moving mean, T
    for i in range(10):
        ypinner = []
        ypeinner = []
        ytinner = []
        yprobinner = []
        skillj = []
        corrj = []
        accj = []
        rmsej = []
        relj = []
        codj = []
        #lag time, tau
        for j in range(10):
            ypobs = pred_nn[k,i,j,:,:]
            ytobs = true_nn[k,i,j,:,:]
            ytime = yearpred[k,i,:]
            
            #expected value / mean prediction
            ypobsexpected = np.dot(ypobs,binCenters) #expeted value from fuzzy classification, prediction
            ytobsexpected = np.dot(ytobs,binCenters) #'expected value from fuzzy classification', observed (same as sie_test for september)
            
            #prediction error (stdev)
            ypobstile = np.transpose(np.tile(ypobsexpected,(23,1)))
            ntt = ypobsexpected.shape[0]
            xi = np.tile(binCenters,(ntt,1))
            ypobserr = np.sqrt(np.sum(np.multiply(np.square(xi-ypobstile),ypobs),axis=1))
            
            #prediction performance
            rel = reliability(ytobsexpected,binCenters,ypobs)
            cod = coefficient_of_determination(ytobsexpected,binCenters,ypobs)
            
            #pad with nans for size consistency in np.append
            #year range depends on time frame
            #time = prediction time
            #is there data or NaN in input
            nanpadding1D = np.full((ntest-ntt,),np.nan)
            nanpadding2D = np.full((ntest-ntt,23),np.nan)
            ypobs_padded = np.concatenate((nanpadding1D,ypobsexpected),axis=0)
            ytobs_padded = np.concatenate((nanpadding1D,ytobsexpected),axis=0)
            ypobserr_padded = np.concatenate((nanpadding1D,ypobserr),axis=0)
            
    
            #performance
            y_pred_obs = ypobs_padded
            y_true_obs = ytobs_padded
            
            skill_obs = 1 - (np.sqrt(np.nanmean(np.square(y_pred_obs - y_true_obs))))/((np.nanstd(y_true_obs)))
            correlation_obs = (np.nansum((y_true_obs-np.nanmean(y_true_obs))*(y_pred_obs-np.nanmean(y_pred_obs))))/((np.sqrt(np.nansum(np.square(y_true_obs-np.nanmean(y_true_obs)))))*(np.sqrt(np.nansum(np.square(y_pred_obs-np.nanmean(y_pred_obs))))))
            #coefficient_of_determination_obs = 1 - np.divide(np.nanmean(np.square(y_pred_obs - y_true_obs)),np.nanmean(np.square(y_true_obs)))
            acc_obs = np.divide(np.nansum(np.multiply((y_pred_obs-np.nanmean(y_pred_obs)),(y_true_obs-np.nanmean(y_true_obs)))),np.multiply(np.sqrt(np.nansum(np.square(y_pred_obs-np.nanmean(y_pred_obs)))),np.sqrt(np.nansum(np.square(y_true_obs-np.nanmean(y_true_obs))))))     
            rmse_obs = np.sqrt(np.divide(np.nansum(np.square(y_pred_obs-y_true_obs)),nmt))
                      
            ypinner.append(ypobs_padded)
            ypeinner.append(ypobserr_padded)
            ytinner.append(ytobs_padded)
            skillj.append(skill_obs)
            corrj.append(correlation_obs)
            accj.append(acc_obs)
            rmsej.append(rmse_obs) 
            codj.append(cod)
            relj.append(rel)
        ypouter.append(ypinner)
        ypeouter.append(ypeinner)
        ytouter.append(ytinner)
        skilli.append(skillj)
        corri.append(corrj)
        acci.append(accj)
        rmsei.append(rmsej)
        codi.append(codj)
        reli.append(relj)
    ypp.append(ypouter)
    yppe.append(ypeouter)
    ytt.append(ytouter)
    skillk.append(skilli)
    corrk.append(corri)
    acck.append(acci)
    rmsek.append(rmsei)
    codk.append(codi)
    relk.append(reli)

Acorrelation_obs = np.array(corrk)[:,0,0]
Askillnp_obs= np.array(skillk)[:,0,0]
Aroot_mse_obs = np.array(rmsek)[:,0,0]
Aacc_obs = np.array(acck)[:,0,0]
Acod = np.array(codk)[:,0,0]
Arel = np.array(relk)[:,0,0]

nn_acc = np.array(acck)
nn_rmse = np.array(rmsek)
nn_cod = np.array(codk)
nn_rel = np.array(relk)
            
nn_prediction_mean_obs = np.array(ypp)
nn_prediction_err_obs = np.array(yppe)
nn_input_obs = np.array(ytt)
#nn_probability_obs = np.array(yprob)



#------------------------------------------------------------
#------------------------------------------------------------
#TRAINING DATA: CMIP6 data that was used to build transfer operator
#------------------------------------------------------------
#------------------------------------------------------------

loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset =  nc.Dataset(loadpath,'r')
years= dataset.variables['unique_years']
sie_tr = np.array(dataset.variables['sie_ensemble_anomaly'])

#rearrange array so same shape as testing data
siet = []
for i in range(17):
    siei = sie_tr[:,i,:,:]
    siet.append(siei)
sie_training = np.array(siet)


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
siete = np.concatenate((sie_training[:,ts[0]:te[0],:,:],sie_training[:,ts[1]:te[1],:,:],sie_training[:,ts[2]:te[2],:,:],sie_training[:,ts[3]:te[3],:,:]),axis=1)


# STANDARDIZE
#------------------------------------------------------------
#------------------------------------------------------------
sigma_sie = np.nanstd(sie,axis=(1,2))[:,np.newaxis,np.newaxis,:]
miu_sie = np.nanmean(sie,axis=(1,2))[:,np.newaxis,np.newaxis,:]

train_standardized_sie = np.divide((sie-miu_sie),sigma_sie)
test_standardized_sie = np.divide((siete-miu_sie),sigma_sie)

sie_training= train_standardized_sie[5:,:,:,:]
siete = test_standardized_sie[5:,:,:,:]
#------------------------------------------------------------
#------------------------------------------------------------



#------------------------------------------------------
#------------------------------------------------------    
#---------PERFECT MODEL OR OBSERVATIONS ? -------------
#------------------------------------------------------    

#test_data = sie_obs[:,np.newaxis,:,:] #evaluate transfer operator on observations
test_data = siete #evaluate transfer operator for perfect model setting

ntest = test_data.shape[1]*test_data.shape[2]
#------------------------------------------------------
#------------------------------------------------------    
#------------------------------------------------------
#------------------------------------------------------ 

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
for i in range(12):
    
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
            persistence_output = np.reshape(test_data[8,:,h:,j],(np.multiply(nt1,[nt2-h])))
            
            #input is same year for: 
            #Xyearly, JFM, AMJ, X[JAS], X[OND], J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)

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

coefficient_of_determ = nn_cod
reliability = nn_rel

#plot time frame versus averaging time
#xaxis is time frame (JAN-DEC)
#each figure is a different hindcast lag
#-----------------------------------------------
#-----------------------------------------------
mm1 = 0 #JAN = 5
mm2 = 12 #DEC = 17

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



'''
#------------------------------------------------------------
#------------------------------------------------------------
#SAVE PREDICTION (PROBABILSTIC AND DETERMINISTIC) AND PERFORMANCE TO NETCDF
#------------------------------------------------------------
#------------------------------------------------------------
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_1D_sie2sie_performance_perfectmodel.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',nn_acc.shape[0]) #no. time frames
    file.createDimension('ntau',nn_acc.shape[1]) #hindcast lag
    file.createDimension('nT',nn_acc.shape[2]) #no. moving means
    file.createDimension('nt',nn_prediction_mean_obs.shape[3]) #no. timesteps, obs
    file.createDimension('ntm',true_nn.shape[3]) #no. timesteps, model
    file.createDimension('nb',true_nn.shape[4]) #no. bins
  

    #create variables
    #NN performance
    acc = file.createVariable('nn_acc','f4',('tf','nT','ntau')) 
    rmse = file.createVariable('nn_rmse','f4',('tf','nT','ntau'))
    coeffdet = file.createVariable('nn_cod','f4',('tf','nT','ntau'))  
    rel = file.createVariable('nn_rel','f4',('tf','nT','ntau'))
    
    
    #write data to variables
    acc[:] = nn_acc
    rmse[:] = nn_rmse
    coeffdet[:] = nn_cod
    rel[:] = nn_rel
    
    
#------------------------------------------------------------
#------------------------------------------------------------
'''