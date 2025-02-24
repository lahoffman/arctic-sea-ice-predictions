#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:52:07 2024

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
import seaborn as sb
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

#machine learning
#------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
import keras.utils
import sklearn
from sklearn.model_selection import train_test_split
from scipy import stats, odr
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#plotting
#------------------
#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable

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
# LOAD PS PERFORMANCE
#------------------------------------------------------
#------------------------------------------------------
loadpath_to = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/TO_sie2sie_performance_obs.nc'
dataset =  nc.Dataset(loadpath_to,'r')
cod_ps = dataset.variables['persistence_cod'][5:,:,:]
#cod_to = dataset.variables['to_acc'][5:,:,:]
#cod_ps = dataset.variables['persistence_acc'][5:,:,:]
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
# I. TRAINING DATA: CMIP6 data 
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
#LOAD TRAINING & TESTING DATA, CMIIP6
#------------------------------------------------------
#------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
train_data = dataset.variables['train_standardized']
test_data = dataset.variables['test_standardized']
miu_train = dataset.variables['miu_tr']
sigma_train = dataset.variables['sigma_tr']

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
# LOAD 2DNN PREDICTIONS
#------------------------------------------------------
#------------------------------------------------------
loadpath_nn= '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_2D_siearea2sie_bins_performance.nc' #2D NN
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
    dataSep = data[13,:,:,0]
    sigma = np.round(np.nanstd(dataSep,axis=(0,1)),2)
    binSize = binSizeFactor*sigma/numberOfBins
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
data = sie_training
numberOfBins=12
binSizeFactor=3
bins, binCenters, binLabels, binSize = define_fuzzyBin(data,numberOfBins,binSizeFactor)   


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
        bootstrap_pi = pi[bootstrap_indices]
        
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

nmt = pred_nn.shape[3]
ntest = pred_nn.shape[3]

#save model output, observations
#------------------------------
relk = []
codk = []
relCIk = []
codCIk = []
#time frames, tf
for k in range(12):
    
    reli = []
    codi = []
    relCIi = []
    codCIi = []
    #moving mean, T
    for i in range(10):
        
        relj = []
        codj = []
        relCIj = []
        codCIj = []
        #lag time, tau
        for j in range(10):
            ypobs = pred_nn[k,i,j,:,:]
            ytobs = true_nn[k,i,j,:]
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
            reliability_mean, reliability_ci = monte_carlo_reliability(ytobsexpected,binCenters,ypobs)
            cod_mean, cod_CI = monte_carlo_r2_bootstrap(ytobsexpected,binCenters,ypobs)
            
            codj.append(cod)
            relj.append(rel)
            relCIj.append(reliability_ci)
            codCIj.append(cod_CI)
        codi.append(codj)
        reli.append(relj)
        relCIi.append(relCIj)
        codCIi.append(codCIj)
    codk.append(codi)
    relk.append(reli)
    relCIk.append(relCIi)
    codCIk.append(codCIi)

nn2d_cod = np.array(codk)
nn2d_rel = np.array(relk)
nn2d_rel_CI = np.array(relCIk)
nn2d_cod_CI = np.array(codCIk)
            

#------------------------------------------------------
#------------------------------------------------------
# LOAD 1DNN PREDICTIONS
#------------------------------------------------------
#------------------------------------------------------
loadpath_nn= '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_1D_sie2sie_performance.nc' #2D NN
dataset =  nc.Dataset(loadpath_nn,'r')
pred_nn = dataset.variables['model_prediction']
true_nn = dataset.variables['model_input']
#------------------------------------------------------
#------------------------------------------------------


nmt = pred_nn.shape[3]
ntest = pred_nn.shape[3]

#save model output, observations
#------------------------------
relk = []
codk = []
relCIk = []
codCIk = []
#time frames, tf
for k in range(12):
    
    reli = []
    codi = []
    relCIi = []
    codCIi = []
    #moving mean, T
    for i in range(10):
        
        relj = []
        codj = []
        relCIj = []
        codCIj = []
        #lag time, tau
        for j in range(10):
            ypobs = pred_nn[k,i,j,:,:]
            ytobs = true_nn[k,i,j,:]
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
            reliability_mean, reliability_ci = monte_carlo_reliability(ytobsexpected,binCenters,ypobs)
            cod_mean, cod_CI = monte_carlo_r2_bootstrap(ytobsexpected,binCenters,ypobs)
            
            codj.append(cod)
            relj.append(rel)
            relCIj.append(reliability_ci)
            codCIj.append(cod_CI)
        codi.append(codj)
        reli.append(relj)
        relCIi.append(relCIj)
        codCIi.append(codCIj)
    codk.append(codi)
    relk.append(reli)
    relCIk.append(relCIi)
    codCIk.append(codCIi)

nn1d_cod = np.array(codk)
nn1d_rel = np.array(relk)
nn1d_rel_CI = np.array(relCIk)
nn1d_cod_CI = np.array(codCIk)


#------------------------------------------------------
#------------------------------------------------------
# PLOT: TO vs NN PERFORMANCE, 5-YEAR
#------------------------------------------------------
#------------------------------------------------------

#plot time frame versus averaging time
#xaxis is time frame (JAN-DEC)
#each figure is a different hindcast lag
#-----------------------------------------------
#-----------------------------------------------
mm1 = 0 #JAN = 5
mm2 = 12 #DEC = 17

# Create a figure and gridspec layout
#fig, axs = plt.subplots(2,1, figsize=(50, 10))
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(200,150))

for j in range(3):
    
    if j == 0:
        cod = np.array(nn1d_cod)
        rel = np.array(nn1d_rel)
        relCI = np.array(nn1d_rel_CI)
        codCI = np.array(nn1d_cod_CI)
        
    elif j == 1:
        cod = np.array(nn2d_cod)
        rel= np.array(nn2d_rel)
        relCI = np.array(nn2d_rel_CI)
        codCI = np.array(nn2d_cod_CI)
        
    else:
        cod = np.array(nn2d_cod)-np.array(nn1d_cod)
        rel = np.sqrt(np.square(np.array(nn1d_rel)-1))-np.sqrt(np.square(np.array(nn2d_rel)-1))
        
    
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
        custom_ticks3 = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        
        # coefficient of determination: Create filled contour plots for each subplot
        contour_levels = np.linspace(-1, 1, 50)  # Adjust levels as needed
        data1 = np.transpose(cod[mm1:mm2,:,i])
        persistence = np.transpose(cod_ps[mm1:mm2,:,i])
        maski = data1 > persistence
        data_rel = np.transpose(rel[mm1:mm2,:,i])
        data_corr = np.transpose(cod[mm1:mm2,:,i])
        data_rel_CI_low = np.transpose(relCI[mm1:mm2,:,i,0])
        data_rel_CI_high = np.transpose(relCI[mm1:mm2,:,i,1])
        data_cod_CI_low = np.transpose(codCI[mm1:mm2,:,i,0])
        data_cod_CI_high = np.transpose(codCI[mm1:mm2,:,i,1])
        
        #mask = ((0.8 < data_rel) & (data_rel < 1.2)) & (data_corr > 0.1) & (data_corr > persistence)
        mask = ((np.round(data_rel_CI_low,2) <= 1) & (1 <= np.round(data_rel_CI_high))) & (data_cod_CI_low > 0) & (data_corr > persistence)
        X, Y = np.meshgrid(xx,ff)
        num_elements_less_than_minus_one = np.sum(data1 < -1)
        masked_Z = np.ma.masked_less(data1, -1)
        maskcod = (data_cod_CI_low > 0) & (data_corr > persistence)
        maskrel = ((np.round(data_rel_CI_low,2) <= 1) & (1 <= np.round(data_rel_CI_high)))
        
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
        data2 = np.transpose(rel[mm1:mm2,:,i])
        num_elements_greater_than_two = np.sum(data2 > 2)
        masked_Z = np.ma.masked_greater(data2, 2)
        data_rel = np.transpose(rel[mm1:mm2,:,i])
        data_corr = np.transpose(cod[mm1:mm2,:,i])
        persistence = np.transpose(cod_ps[mm1:mm2,:,i])
        mask = ((np.round(data_rel_CI_low,2) <= 1) & (1 <= np.round(data_rel_CI_high))) & (data_cod_CI_low > 0) & (data_corr > persistence)
        maskint = mask.astype(int)
        X, Y = np.meshgrid(xx,ff)
        maskcod = (data_cod_CI_low > 0) & (data_corr > persistence)
        maskrel = ((np.round(data_rel_CI_low,2) <= 1) & (1 <= np.round(data_rel_CI_high)))
        
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
    maskcod11 = mc1
    ff = np.arange(1,11)
    xx = np.arange(120)
    label_small = 90
    label_big = 140
    label_mid = 90
    nmo = 60
     
    
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    kl = np.concatenate((np.arange(0,2)[:,np.newaxis],np.arange(2,4)[:,np.newaxis],np.arange(4,6)[:,np.newaxis]),axis=1)
    titles = ['(a)','(b)','(c)','(d)','(e)','(f)']
    if j < 2:
        #coefficient of determination
        cmap = mpl.cm.get_cmap('RdGy_r')
        cendcod = cmap(0)
        cendcodf = cmap(2.0)
        k = 0
        ax = axes[kl[k,j]]
        contour_levels = np.linspace(-1, 1, 50)  # Adjust levels as needed
        X, Y = np.meshgrid(xx,ff)
        num_elements_less_than_minus_one = np.sum(data11 < -1)
        masked_Z = np.ma.masked_less(data11, -1)
        cp1 = ax.contourf(xx[0:nmo], ff, data11[:,0:nmo], levels=contour_levels, cmap='RdGy_r')
        ch1 = ax.contourf(xx[0:nmo], ff, maski11[:,0:nmo], levels=[0, 0.5], colors='none',hatches=['x', 'x'],alpha=0.3) 
        #axs[0].contour(xx,ff,data11,levels=[0.4],colors='maroon')
        Xshort = X[:,0:nmo]
        Yshort = Y[:,0:nmo]
        ax.scatter(Xshort[maskcod11[:,0:nmo]], Yshort[maskcod11[:,0:nmo]], s=100,color='black')
        if j == 0:
            ax.scatter(Xshort[mask11[:,0:nmo]], Yshort[mask11[:,0:nmo]], s=1000,marker='o',color='maroon',edgecolors= "black")
        elif j == 1:
            ax.scatter(Xshort[mask11[:,0:nmo]], Yshort[mask11[:,0:nmo]], s=1000,marker='o',color='seagreen',edgecolors= "black")
        
        
        if num_elements_less_than_minus_one > 0:
            ax.contourf(xx[0:nmo],ff, data11[:,0:nmo], levels=[data11.min(), -1], colors=[cendcod])
        
        cb1 = fig.colorbar(cp1, ax=ax,aspect=10)
        cb1.set_label('coefficient of determination', fontsize=label_big) 
        cb1.set_ticks(custom_ticks1)
        cb1.ax.tick_params(labelsize=label_small)
        ax.set_xticks(xx[0:nmo])
        ax.set_xticklabels(ticki[0:nmo],fontsize=label_small)
        ax.set_yticklabels(np.arange(1,11),fontsize=label_small)
        ax.set_title(titles[kl[k,j]])
        
        data22 = d2
        mask22 = m2
        maskrel22 = mr2
        
        #reliability
        cmap = mpl.cm.get_cmap('BrBG_r')
        cendrel = cmap(1.0)
        k = 1
        ax = axes[kl[k,j]]
        contour_levels = np.linspace(0, 2, 30)  # Adjust levels as needed
        num_elements_greater_than_two = np.sum(data22 > 2)
        cp2 = ax.contourf(xx[0:nmo], ff, data22[:,0:nmo], levels=contour_levels, cmap='BrBG_r',vmin=0, vmax=2)
        ax.contour(xx[0:nmo],ff,data22[:,0:nmo],levels = [0.5],colors='darkolivegreen')
        ax.contour(xx[0:nmo],ff,data22[:,0:nmo],levels = [1.5],colors='peru')
        ax.scatter(Xshort[maskrel22[:,0:nmo]], Yshort[maskrel22[:,0:nmo]], s=100,color='black')
        if j == 0:
            ax.scatter(Xshort[mask22[:,0:nmo]], Yshort[mask22[:,0:nmo]],s=1000,marker='o',color='maroon',edgecolors= "black")
        elif j == 1:
            ax.scatter(Xshort[mask22[:,0:nmo]], Yshort[mask22[:,0:nmo]],s=1000, marker='o', color='seagreen',edgecolors= "black")
        
        if num_elements_greater_than_two > 0:
            ax.contourf(xx[0:nmo],ff, data22[:,0:nmo], levels=[2,data22.max()], colors=[cendrel])
        

        cb2 = fig.colorbar(cp2, ax= ax,aspect=10)
        cb2.set_label('reliability', fontsize=label_big) 
        cb2.set_ticks(custom_ticks2)
        cb2.ax.tick_params(labelsize=label_small)
        ax.set_xticks(xx[0:nmo])
        ax.set_xticklabels(ticki[0:nmo],fontsize=label_small)
        ax.set_yticklabels(np.arange(1,11),fontsize=label_small)
        ax.set_title(titles[kl[k,j]])
        
        if j == 0:
            maskcod0 = mask11
            maskrel0 = mask22
        if j == 1:
            maskcod1 = mask11
            maskrel1 = mask22
        
    
    elif j == 2:
        #coefficient of determination
        k = 0
        ax = axes[kl[k,j]]
        maskcod = maskcod0 & maskcod1
        maskcodall = maskcod0 | maskcod1
        maskonly0 = np.sum(maskcod0 & ~maskcod1)
        maskonly1 = np.sum(maskcod1 & ~maskcod0)
        maskboth = np.sum(maskcod)
        maskall = np.sum(maskcodall)
        total0 = maskonly0/maskall
        total1 = maskonly1/maskall
        totalboth = maskboth/maskall
        
        contour_levels = np.linspace(-2.0,2.0, 50)  # Adjust levels as needed
        X, Y = np.meshgrid(xx,ff)
        Xshort,Yshort = np.meshgrid(xx[0:nmo],ff)
        num_elements_greater_than_two = np.sum(data11 > 2)
        masked_Z = np.ma.masked_less(data11, -1)
        cp1 = ax.contourf(xx[0:nmo], ff, data11[:,0:nmo], levels=contour_levels, cmap='seismic')
        ax.scatter(Xshort[maskcod0[:,0:nmo]], Yshort[maskcod0[:,0:nmo]], marker='o', s=1500,color='maroon',edgecolors= "black")
        ax.scatter(Xshort[maskcod1[:,0:nmo]], Yshort[maskcod1[:,0:nmo]], marker='o', s=1500,color='seagreen',edgecolors= "black")
        ax.scatter(Xshort[maskcod[:,0:nmo]], Yshort[maskcod[:,0:nmo]], marker = 'o', s=1500,color='black',edgecolors= "black")
        if num_elements_greater_than_two > 0:
            ax.contourf(xx[0:nmo],ff, data11[:,0:nmo], levels=[2,data11.max()], colors=[cendcodf])
        
        cb1 = fig.colorbar(cp1, ax=ax,aspect=10)
        cb1.set_label(r'$cod_{nn2d} - cod_{nn1d}$', fontsize=label_big) 
        cb1.set_ticks(custom_ticks3)
        cb1.ax.tick_params(labelsize=label_small)
        ax.set_xticks(xx[0:nmo])
        ax.set_xticklabels(ticki[0:nmo],fontsize=label_small)
        ax.set_yticklabels(np.arange(1,11),fontsize=label_small)
        ax.set_title(titles[kl[k,j]])
        
        data22 = d2
        mask22 = m2
        
        #reliability
        k = 1
        ax = axes[kl[k,j]]
        maskrel = maskrel0 & maskrel1
        maskrelall = maskrel0 | maskrel1
        contour_levels = np.linspace(-2,2, 50)  # Adjust levels as needed
        num_elements_greater_than_two = np.sum(data22 > 2)
        cp2 = ax.contourf(xx[0:nmo], ff, data22[:,0:nmo], levels=contour_levels, cmap='seismic',vmin=-2, vmax=2)
        ax.scatter(Xshort[maskrel0[:,0:nmo]], Yshort[maskrel0[:,0:nmo]],s=1500,marker='o',color='maroon',edgecolors= "black")
        ax.scatter(Xshort[maskrel1[:,0:nmo]], Yshort[maskrel1[:,0:nmo]],s=1500,marker='o',color='seagreen',edgecolors= "black")
        ax.scatter(Xshort[maskrel[:,0:nmo]], Yshort[maskrel[:,0:nmo]],s=1500,marker='o', color='black',edgecolors= "black")
             
        cb2 = fig.colorbar(cp2, ax= ax,aspect=10)
        cb2.set_label(r'$\sqrt{(rel_{nn1d}-1)^2} - \sqrt{(rel_{nn2d}-1)^2}$', fontsize=label_big) 
        cb2.set_ticks(custom_ticks3)
        cb2.ax.tick_params(labelsize=label_small)
        ax.set_xticks(xx[0:nmo])
        ax.set_xticklabels(ticki[0:nmo],fontsize=label_small)
        ax.set_yticklabels(np.arange(1,11),fontsize=label_small)
        ax.set_title(titles[kl[k,j]])
    
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.87, top=0.87, bottom=0.1, hspace=0.2, wspace=0.01)
    #ax1.set_ylabel('transfer operator (TO)', fontsize=label_big)
    #ax5.set_ylabel(r'$\Delta (NN, TO)$', fontsize=label_big)
    ax3.set_ylabel('averaging time [year]', fontsize=label_big)
    fig.text(0.38, 0.06,'hindcast lag [previous month]', fontsize=label_big)
    fig.text(0.23, 0.335,r'(e) $\Delta (NN2D, NN1D)$', fontsize=label_big)
    fig.text(0.63, 0.335,r'(f) $\Delta (NN2D, NN1D)$', fontsize=label_big)
    fig.text(0.2, 0.605,'(c) neural network, 2D (NN2D)', fontsize=label_big)
    fig.text(0.6, 0.605,'(d) neural network, 2D (NN2D)', fontsize=label_big)
    fig.text(0.2, 0.875,'(a) neural network, 1D (NN1D)', fontsize=label_big)
    fig.text(0.6, 0.875,'(b) neural network, 1D (NN1D)', fontsize=label_big)

#------------------------------------------------------
#------------------------------------------------------

percentskillfulNN = np.sum(maskcod1[:,0:60])/600
percentskillfulTO = np.sum(maskcod0[:,0:60])/600

maskcod1reshape = np.reshape(maskcod1[:,0:60],(10,5,12))
maskcod0reshape = np.reshape(maskcod0[:,0:60],(10,5,12))
monthlymask1 = np.sum(maskcod1reshape,axis=(0,1))
monthlymask0 = np.sum(maskcod0reshape,axis=(0,1))



'''
#------------------------------------------------------------
#------------------------------------------------------------
# SAVE: PERFORMANCE CI
#------------------------------------------------------------
#------------------------------------------------------------

savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/performance_CI_NN1D_NN2D_PM.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',nn1d_cod_CI.shape[0]) #no. time frames, tf = 12
    file.createDimension('nT',nn1d_cod_CI.shape[1]) #T = 10
    file.createDimension('ntau',nn1d_cod_CI.shape[2]) #tau = 10
    file.createDimension('nl',nn1d_cod_CI.shape[3]) #upper,lower [2]

    #create variables
    #TO performance
    nn1dcod = file.createVariable('nn1d_cod','f4',('tf','nT','ntau')) 
    nn1drel = file.createVariable('nn1d_rel','f4',('tf','nT','ntau')) 
    nn1dcodci = file.createVariable('nn1d_cod_CI','f4',('tf','nT','ntau','nl')) 
    nn1drelci = file.createVariable('nn1d_rel_CI','f4',('tf','nT','ntau','nl')) 
    
    nn2dcod = file.createVariable('nn2d_cod','f4',('tf','nT','ntau')) 
    nn2drel = file.createVariable('nn2d_rel','f4',('tf','nT','ntau')) 
    nn2dcodci = file.createVariable('nn2d_cod_CI','f4',('tf','nT','ntau','nl')) 
    nn2drelci = file.createVariable('nn2d_rel_CI','f4',('tf','nT','ntau','nl')) 
    
    pscod = file.createVariable('cod_ps','f4',('tf','nT','ntau')) 

    #write data to variables
    nn1dcod[:] = nn1d_cod
    nn1drel[:] = nn1d_rel
    nn1dcodci[:] = nn1d_cod_CI
    nn1drelci[:] = nn1d_rel_CI

    nn2dcod[:] = nn2d_cod
    nn2drel[:] = nn2d_rel
    nn2dcodci[:] = nn2d_cod_CI
    nn2drelci[:] = nn2d_rel_CI
    
    pscod[:] = cod_ps

#------------------------------------------------------------
#------------------------------------------------------------
'''