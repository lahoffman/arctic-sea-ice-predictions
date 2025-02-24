#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:26:59 2024

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
from scipy.io import loadmat
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
#------------------------------------------------------
#------------------------------------------------------

#------------------------------------------------------ 
#------------------------------------------------------ 
# LOAD: PREDICTION & NSIDC, 1989-2024
#------------------------------------------------------ 
#------------------------------------------------------ 

#NN predictions
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/prediction_full_detrend_2DNN.nc'
dataset =  nc.Dataset(loadpath,'r')
sie_prediction_residual_nn = np.array(dataset.variables['sie_prediction_residual'])
sie_prediction_nn = np.array(dataset.variables['sie_prediction'])
sie_sep_residual_nsidc_nn = np.array(dataset.variables['sie_sep_residual_nsidc'])
sie_sep_nsidc_nn = np.array(dataset.variables['sie_sep_nsidc'])

#TO predictions
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/prediction_full_detrend_1DTO.nc'
dataset =  nc.Dataset(loadpath,'r')
sie_prediction_residual_to = np.array(dataset.variables['sie_prediction_residual'])[:,1:]
sie_prediction_to = np.array(dataset.variables['sie_prediction'])[:,1:]
sie_sep_residual_nsidc_to = np.array(dataset.variables['sie_sep_residual_nsidc'])[1:]
sie_sep_nsidc_to = np.array(dataset.variables['sie_sep_nsidc'])[1:]
sie_persistence_full_pred = np.array(dataset.variables['sie_persistence_full_pred'])[:,1:]

#------------------------------------------------------ 


#------------------------------------------------------ 
#------------------------------------------------------ 
# EVALUATION METRICS
#------------------------------------------------------ 
#------------------------------------------------------ 

def rmse(y_true_obs,y_pred_obs):
    nmt = y_true_obs.shape[0]
    return np.sqrt(np.divide(np.nansum(np.square(y_pred_obs-y_true_obs)),nmt))

def acc(y_true,y_pred):
    return (np.nansum((y_true-np.nanmean(y_true))*(y_pred-np.nanmean(y_pred))))/((np.sqrt(np.nansum(np.square(y_true-np.nanmean(y_true)))))*(np.sqrt(np.nansum(np.square(y_pred-np.nanmean(y_pred))))))

def rmsedt(y_true_obs,y_pred_obs,obs_lin,pred_lin):
    nmt = y_true_obs.shape[0]
    return np.sqrt(np.divide(np.nansum(np.square((y_pred_obs-pred_lin)-(y_true_obs-obs_lin))),nmt))

def accdt(y_true,y_pred,obs_lin,pred_lin):
    return (np.nansum((y_true-obs_lin)*(y_pred-pred_lin)))/((np.sqrt(np.nansum(np.square(y_true-obs_lin))))*(np.sqrt(np.nansum(np.square(y_pred-pred_lin)))))

#------------------------------------------------------ 


#------------------------------------------------------ 
#------------------------------------------------------ 
# LINEAR TREND, OBS & PRED
#------------------------------------------------------ 
#------------------------------------------------------ 
x = np.arange(1980,2025)
y = sie_sep_nsidc_to
slope, intercept = np.polyfit(x, y, 1)
trend_line_obs = slope * x + intercept

trend_pred_nn = []
trend_pred_to = []
trend_pred_ps = []
for i in range(12):
    y = sie_prediction_nn[i,:]
    slope, intercept = np.polyfit(x, y, 1)
    trend_line_pred_nn = slope * x + intercept
    trend_pred_nn.append(trend_line_pred_nn)

    y = sie_prediction_to[i,:]
    slope, intercept = np.polyfit(x, y, 1)
    trend_line_pred_to = slope * x + intercept
    trend_pred_to.append(trend_line_pred_to)
    
    y = sie_persistence_full_pred[i,:]
    slope, intercept = np.polyfit(x, y, 1)
    trend_line_pred_ps = slope * x + intercept
    trend_pred_ps.append(trend_line_pred_ps)

trend_lin_to = np.array(trend_pred_to)
trend_lin_nn = np.array(trend_pred_nn)
trend_lin_ps = np.array(trend_pred_ps)

acctoj = []
acctodj = []
accnnj = []
accnndj = []
accpsj = []
accpsdj = []
rmsetoj = []
rmsetodj = []
rmsennj = []
rmsenndj = []
rmsepsj = []
rmsepsdj = []
for i in range(4,8):
    acctoi = acc(sie_sep_nsidc_to,sie_prediction_to[i,:])
    acctodi = accdt(sie_sep_nsidc_to,sie_prediction_to[i,:],trend_line_obs,trend_lin_to[i,:])
    accnni = acc(sie_sep_nsidc_nn,sie_prediction_nn[i,:])
    accnndi = accdt(sie_sep_nsidc_nn,sie_prediction_nn[i,:],trend_line_obs,trend_lin_nn[i,:])
    accpsi = acc(sie_sep_nsidc_nn,sie_persistence_full_pred[i,:])
    accpsdi = accdt(sie_sep_nsidc_nn,sie_persistence_full_pred[i,:],trend_line_obs,trend_lin_ps[i,:])
    
    acctoj.append(acctoi)
    acctodj.append(acctodi)
    accnnj.append(accnni)
    accnndj.append(accnndi)
    accpsj.append(accpsi)
    accpsdj.append(accpsdi)
    
    rmsetoi = rmse(sie_sep_nsidc_to,sie_prediction_to[i,:])
    rmsetodi = rmsedt(sie_sep_nsidc_to,sie_prediction_to[i,:],trend_line_obs,trend_lin_to[i,:])
    rmsenni = rmse(sie_sep_nsidc_nn,sie_prediction_nn[i,:])
    rmsenndi = rmsedt(sie_sep_nsidc_nn,sie_prediction_nn[i,:],trend_line_obs,trend_lin_nn[i,:])
    rmsepsi = rmse(sie_sep_nsidc_nn,sie_persistence_full_pred[i,:])
    rmsepsdi = rmsedt(sie_sep_nsidc_nn,sie_persistence_full_pred[i,:],trend_line_obs,trend_lin_ps[i,:])
    
    rmsetoj.append(rmsetoi)
    rmsetodj.append(rmsetodi)
    rmsennj.append(rmsenni)
    rmsenndj.append(rmsenndi)
    rmsepsj.append(rmsepsi)
    rmsepsdj.append(rmsepsdi)


accto = np.array(acctoj)
accdetrendto = np.array(acctodj)
accnn = np.array(accnnj)
accdetrendnn = np.array(accnndj)
accps = np.array(accpsj)
accdetrendps = np.array(accpsdj)

rmseto = np.array(rmsetoj)
rmsedetrendto = np.array(rmsetodj)
rmsenn = np.array(rmsennj)
rmsedetrendnn = np.array(rmsenndj)
rmseps = np.array(rmsepsj)
rmsedetrendps = np.array(rmsepsdj)

#------------------------------------------------------
#------------------------------------------------------
# LOAD BUSHUK ET AL DATA
#------------------------------------------------------
#------------------------------------------------------

#filepath
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/Bushuk_processed_data/'

#2001-2020 statistical models
filep = 'SIO_skill_statistical_1993_2021_nsidc.mat'
file_path = loadpath+filep
data = loadmat(file_path)
print(data.keys())  # Lists variable names
model_list = data["model_list"]  
ACC_panArctic_MM = data['ACC_panArctic_MM']
ACC_detrend_panArctic_MM  = data['ACC_detrend_panArctic_MM']
RMSE_panArctic_MM = data['RMSE_panArctic_MM']
RMSE_detrend_panArctic_MM  = data['RMSE_detrend_panArctic_MM']


#multi-model, median, 2001-2020
filep = 'median_raw_stat/SIO_skill_1993_2021_nsidc.mat'
file_path = loadpath+filep
data = loadmat(file_path)
print(data.keys())  # Lists variable names
ACC_panArctic = data["ACC_panArctic"]
ACC_detrend_panArctic = data["ACC_detrend_panArctic"]
RMSE_panArctic = data["RMSE_panArctic"]
RMSE_detrend_panArctic = data["RMSE_detrend_panArctic"]


#------------------------------------------------------
#------------------------------------------------------ 
# PLOT
#------------------------------------------------------
#------------------------------------------------------ 

fig = plt.figure(figsize=(10,10))
label_large = 18
label_small = 14
tit = ['(a) PanArtic ACC','(b) PanArctic ACC, detrended','(c) PanArctic RMSE','(d) PanArctic RMSE, detrended']
xl = ['Jun 1','Jul 1','Aug 1','Sep 1']



plt.subplot(2,2,1)
for i in range(17):
    if i == 0:
        plt.plot(np.arange(4),ACC_panArctic_MM[:,i],color='lightgray', linewidth=2.0, markersize=0,label='statistical models, Bushuk et al. (2024)')
    else:
        plt.plot(np.arange(4),ACC_panArctic_MM[:,i],color='lightgray', linewidth=2.0, markersize=0,label='')
plt.plot(np.arange(4),ACC_panArctic,color='dimgray', linewidth=3.0, marker='o', markersize=5,label='multi-model median, Bushuk et al. (2024)')
plt.plot(np.arange(4),accto,color='r', linewidth=2.0, marker='o', markersize=5,label='transfer operator')
plt.plot(np.arange(4),accnn,color='g', linewidth=2.0, marker='o', markersize=5,label='neural network')
plt.plot(np.arange(4),accps,color='k', linestyle='--', linewidth=2.0, marker='o', markersize=5,label='persistence')
plt.title(tit[0],fontsize=label_large)
plt.ylabel('Correlation',fontsize=label_large)
plt.ylim(0,1)
plt.xticks(np.arange(4),labels =xl,fontsize=label_small)
plt.legend(loc='lower right')

plt.subplot(2,2,2)
for i in range(17):
    plt.plot(np.arange(4),ACC_detrend_panArctic_MM[:,i],color='lightgray', linewidth=2.0, markersize=0,label='')
plt.plot(np.arange(4),ACC_detrend_panArctic,color='dimgray', linewidth=3.0, marker='o', markersize=5,label='multi-model median, Bushuk et al. (2024)')
plt.plot(np.arange(4),accdetrendto,color='r', linewidth=2.0, marker='o', markersize=5,label='transfer operator')
plt.plot(np.arange(4),accdetrendnn,color='g', linewidth=2.0, marker='o', markersize=5,label='neural network')
plt.plot(np.arange(4),accdetrendps,color='k',linestyle='--', linewidth=2.0, marker='o', markersize=5,label='persistence')
plt.title(tit[1],fontsize=label_large)
plt.ylim(0,1)
plt.xticks(np.arange(4),labels =xl,fontsize=label_small)

plt.subplot(2,2,3)
for i in range(17):
    plt.plot(np.arange(4),RMSE_panArctic_MM[:,i],color='lightgray', linewidth=2.0, markersize=0,label='')
plt.plot(np.arange(4),RMSE_panArctic,color='dimgray', linewidth=3.0, marker='o', markersize=5,label='multi-model median, Bushuk et al. (2024)')
plt.plot(np.arange(4),rmseto,color='r', linewidth=2.0, marker='o', markersize=5,label='transfer operator')
plt.plot(np.arange(4),rmsenn,color='g', linewidth=2.0, marker='o', markersize=5,label='neural network')
plt.plot(np.arange(4),rmseps,color='k', linestyle='--', linewidth=2.0, marker='o', markersize=5,label='persistence')
plt.title(tit[2],fontsize=label_large)
plt.ylabel('RMSE (Mkm$^2$)',fontsize=label_large)
plt.ylim(0,2)
plt.xticks(np.arange(4),labels =xl,fontsize=label_small)

plt.subplot(2,2,4)
for i in range(17):
    plt.plot(np.arange(4),RMSE_detrend_panArctic_MM[:,i],color='lightgray', linewidth=2.0, markersize=0,label='Bushuk et al. (2024)')
plt.plot(np.arange(4),RMSE_detrend_panArctic,color='dimgray', linewidth=3.0, marker='o', markersize=5,label='multi-model median, Bushuk et al. (2024)')
plt.plot(np.arange(4),rmsedetrendto,color='r', linewidth=2.0, marker='o', markersize=5,label='transfer operator')
plt.plot(np.arange(4),rmsedetrendnn,color='g', linewidth=2.0, marker='o', markersize=5,label='neural network')
plt.plot(np.arange(4),rmsedetrendps,color='k',linestyle='--', linewidth=2.0, marker='o', markersize=5,label='persistence')
plt.title(tit[3],fontsize=label_large)
plt.ylim(0,1)   
plt.xticks(np.arange(4),labels =xl,fontsize=label_small)