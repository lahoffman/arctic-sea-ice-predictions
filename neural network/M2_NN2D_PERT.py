"""
Created on Tue Jan 07 2025

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

#machine learning
#------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
import keras.utils
from keras.layers import Dense, Activation
import sklearn
from sklearn.model_selection import train_test_split
from scipy import stats, odr
#from keras import regularizers
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

time_frames = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
start_mo = 11
end_mo = 12

# a. SIEXTENTN
#------------------------------------------------------------
#load_path = '/Users/hoffmanl/Documents/data/mip46/sie_sit_siv_SM_RMMM_NORMZMOS_TSS_MM_ARCTIC.nc'
loadpath_sie = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset_sie =  nc.Dataset(loadpath_sie,'r')
years = dataset_sie.variables['unique_years']
sie_tr = np.array(dataset_sie.variables['sie_ensemble_anomaly'])
tij = np.arange(1,11)
tmj = np.arange(0,10)

siet = []
for i in range(17):
    siei = sie_tr[:,i,:,:]
    siet.append(siei)
sie_training = np.array(siet)


# b. AREA(SIVOL > THRESHOLD)
#------------------------------------------------------------
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siat_1p25m_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM_mon2sep_gn_185001-201412.nc'
dataset =  nc.Dataset(loadpath,'r')
area_tr = np.array(dataset.variables['area_ensemble_anomaly'])

areat = []
for i in range(17):
    areai = area_tr[:,i,:,:]
    areat.append(areai)
area_training = np.array(areat)
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
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
area = np.concatenate((area_training[:,0:ts[0],:,:],area_training[:,te[0]:ts[1],:,:],area_training[:,te[1]:ts[2],:,:],area_training[:,te[2]:ts[3],:,:],area_training[:,te[3]:,:,:]),axis=1)

siete = np.concatenate((sie_training[:,ts[0]:te[0],:,:],sie_training[:,ts[1]:te[1],:,:],sie_training[:,ts[2]:te[2],:,:],sie_training[:,ts[3]:te[3],:,:]),axis=1)
areate = np.concatenate((area_training[:,ts[0]:te[0],:,:],area_training[:,ts[1]:te[1],:,:],area_training[:,ts[2]:te[2],:,:],area_training[:,ts[3]:te[3],:,:]),axis=1)
#------------------------------------------------------------
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
# STANDARDIZE
#------------------------------------------------------------
#------------------------------------------------------------
sigma_sie = np.nanstd(sie,axis=(1,2))[:,np.newaxis,np.newaxis,:]
miu_sie = np.nanmean(sie,axis=(1,2))[:,np.newaxis,np.newaxis,:]

train_standardized_sie = np.divide((sie-miu_sie),sigma_sie)
test_standardized_sie = np.divide((siete-miu_sie),sigma_sie)

sie = train_standardized_sie[5:,:,:,:]
siete = test_standardized_sie[5:,:,:,:]

sigma_siat = np.nanstd(area,axis=(1,2))[:,np.newaxis,np.newaxis,:]
miu_siat = np.nanmean(area,axis=(1,2))[:,np.newaxis,np.newaxis,:]

train_standardized_siat = np.divide((area-miu_siat),sigma_siat)
test_standardized_siat = np.divide((areate-miu_siat),sigma_siat)

area = train_standardized_siat[5:,:,:,:]
areate = test_standardized_siat[5:,:,:,:]
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
# III. NEURAL NETWORK PARAMETERS
#------------------------------------------------------------
#------------------------------------------------------------

#define loss functions
#----------------------------------------
#define NRMSE function
def norm_root_mean_squared_error(y_true,y_pred):
    return  (K.sqrt(K.mean(K.square(y_pred - y_true))))/((K.std(y_true)))

#define pearson correlation 
def corr(y_true, y_pred):
    return (K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred))))/((K.sqrt(K.sum(K.square(y_true-K.mean(y_true)))))*(K.sqrt(K.sum(K.square(y_pred-K.mean(y_pred))))))

#define RMSD for perturbation
def rmsd_pert(y_pert,y_control):
    n = y_pert.shape[0]
    return K.sqrt((1/n)*K.sum(K.square(y_pert-y_control)))

def monte_carlo_rmsd_bootstrap(pert, control, n_samples=1000, ci=0.95):
    rmsd_samples = []
    
    for _ in range(n_samples):
        # Bootstrap resampling of observations
        bootstrap_indices = np.random.choice(len(pert), size=len(pert), replace=True)
        bootstrap_obs = pert[bootstrap_indices]
        bootstrap_pi = control[bootstrap_indices]
        
        # Compute R^2 for this resampled set
        rmsd_metric = rmsd_pert(bootstrap_obs, bootstrap_pi)
        rmsd_samples.append(rmsd_metric)
    
    rmsdsamples = np.array(rmsd_samples)
    rmsd_mean = np.mean(rmsd_samples)
    
    # Compute confidence interval using percentiles (non-parametric)
    lower_bound = np.percentile(rmsd_samples, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(rmsd_samples, (1 + ci) / 2 * 100)
    rmsd_ci = (lower_bound, upper_bound)
    
    return rmsd_mean, rmsd_ci

#define model hyper parameters
#----------------------------------------
# LOSS FUNCTION
LOSS = 'kullback_leibler_divergence'
METRIC = tf.keras.metrics.RootMeanSquaredError() # Metric for assessing model skill

# MODEL TRAINING
N_UNITS = 10 # number of nodes in layer
NUM_EPOCHS = 10 # Max number of times all of the data will be seen iteratively in training
BATCH_SIZE = 200 # Number of samples per epoch
ACTIVATION_FUNCTION = 'relu' #activation function [others are 'sigmoid','tanh','linear']
LEARNING_RATE = .01 # Learning rate (think step size)
DROP = 0.2 # dropout rate
OPTIMIZER = 'adam' #gradient descent algorithm
RL2 = 0.01

#FUZZY CLASSIFICATION BINS FOR OUTPUT
#----------------------------------------
#step 1: define bins
def define_fuzzyBin(data,numberOfBins,binSizeFactor):
    
    #build bins based on standard deviation, size factor and number of bins
    dataSep = data[8,:,:,:]
    sigma = np.round(np.nanstd(dataSep,axis=(1,2)),2)
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
 
#------------------------------------------------------------
#------------------------------------------------------------



#------------------------------------------------------------
#------------------------------------------------------------
# IV. BUILD NEURAL NETWORK
#------------------------------------------------------------
#------------------------------------------------------------

#a. define fuzzy bins for training data
#----------------------------------------
data = sie
numberOfBins=12
binSizeFactor=3
bins, binCenters, binLabels, binSize = define_fuzzyBin(data,numberOfBins,binSizeFactor)   

#b. train NN
#there will be a separate NN for each:
#time frame (tf), moving mean(T), and lag time (tau)
#----------------------------------------
y_pred_ics_i = []
x_ics_i = []  
pert_rmsd_SIEijk = []
pert_rmsd_SIAtijk = []
pert_rmsd_SIEijk_mean = []
pert_rmsd_SIEijk_CI = []
pert_rmsd_SIAtijk_mean = []
pert_rmsd_SIAtijk_CI = []
#time frames, tf
for i in range(12):
    y_pred_ics_ij = []
    x_ics_ij = []  
    pert_rmsd_SIEij = []
    pert_rmsd_SIAtij = []
    pert_rmsd_SIEij_mean = []
    pert_rmsd_SIEij_CI = []
    pert_rmsd_SIAtij_mean = []
    pert_rmsd_SIAtij_CI = []
    #moving mean, T
    for j in range(1):
        y_pred_ics_ijh = []  
        x_ics_ijh = []  
        pert_rmsd_SIEi = []
        pert_rmsd_SIAti = []
        pert_rmsd_SIEi_mean = []
        pert_rmsd_SIEi_CI = []
        pert_rmsd_SIAti_mean = []
        pert_rmsd_SIAti_CI = []
        #lag time, tau
        for h in range(1):
            
            #TRAINING DATA: CMIP6 MODELS 
            #----------------------
            #data at t and t+1
            #output is september, t+1
            data2 = sie[8,:,:,j]
            data_t2 = data2[:,tij[h]:]
            
            #input is other timeframes
            data11 = sie[i,:,:,j]
            data12 = area[i,:,:,j]
            

            #input is same year for: 
            #J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
            data_t11ii = []
            data_t12ii = []
            if t0[i] == 0:
                data_t11 = data11[:,:-tij[h]]
                data_t12 = data12[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_t11 = data11[:,1:]
                    data_t12 = data12[:,1:]
                else:
                    data_t11 = data11[:,1:-tmj[h]]
                    data_t12 = data12[:,1:-tmj[h]]

            
            nt = data_t11.shape[1]
            nm = data_t11.shape[0]
            data_t11_reshape = np.reshape(data_t11,[nm*nt,1])
            data_t12_reshape = np.reshape(data_t12,[nm*nt,1])
            data_t2_reshapei = np.reshape(data_t2,[nm*nt,1])            
            data_t2_reshapei = data_t2_reshapei[~np.isnan(data_t11_reshape)][:, np.newaxis]
            data_t12_reshape = data_t12_reshape[~np.isnan(data_t12_reshape)][:, np.newaxis]
            data_t11_reshape = data_t11_reshape[~np.isnan(data_t11_reshape)][:, np.newaxis]
            data_t1_reshape = np.concatenate((data_t11_reshape,data_t12_reshape),axis=1)
 
            #convert training output to fuzzy bins
            data_t2_reshape = convert_fuzzyBins(data_t2_reshapei[:,0],binCenters)

            xt = data_t1_reshape
            yt = data_t2_reshape
            
                      
            x_train, x_val, y_train, y_val = train_test_split(xt,yt,test_size = .175, shuffle=True, random_state = 12)
            nch = 1
            input_dim = x_train.shape[1]
            
            #MODEL: BUILD NEURAL NETWORK
            #----------------------------------------
            # define the model
            model = tf.keras.models.Sequential()
            model.add(Dense(N_UNITS, activation=ACTIVATION_FUNCTION, kernel_regularizer=tf.keras.regularizers.l2(RL2),input_shape=(input_dim,)))
            model.add(Dense(N_UNITS, activation=ACTIVATION_FUNCTION,kernel_regularizer=tf.keras.regularizers.l2(RL2)))
            model.add(Dense(len(binCenters)))
            model.add(Activation('softmax'))
                
            
            # Update the optimizer to use 'learning_rate' instead of 'lr'
            filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/model_output/'
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss=LOSS, metrics=[METRIC])
            model.summary()
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

            history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,shuffle=True, validation_data=(x_val, y_val),callbacks=[early_stopping])
            #print(history.history)
            
            
            #.............................................
            # Network Predictions 
            #.............................................
            #control, predict from x_train
            y_pred_tr = model.predict(x_train)
            nt = x_train.shape[0]
            
            #perturbation, +0.5*sigma, SIE
            x_tr_SIE = x_train[:,0]
            x_tr_SIAt = x_train[:,1]
            x_tr_SIE_sigma = np.nanstd(x_tr_SIE)
            x_tr_SIE_perturb = x_tr_SIE+0.5*x_tr_SIE_sigma*np.ones(nt,)
            x_tr_SIE_perturbation = np.concatenate((x_tr_SIE_perturb[:,np.newaxis],x_tr_SIAt[:,np.newaxis]),axis=1)
            y_pred_tr_SIE_perturbatuion = model.predict(x_tr_SIE_perturbation)
            
            #perturbation, +0.5*sigma, SIAt
            x_tr_SIE = x_train[:,0]
            x_tr_SIAt = x_train[:,1]
            x_tr_SIAt_sigma = np.nanstd(x_tr_SIAt)
            x_tr_SIAt_perturb = x_tr_SIAt+0.5*x_tr_SIAt_sigma*np.ones(nt,)
            x_tr_SIAt_perturbation = np.concatenate((x_tr_SIE[:,np.newaxis],x_tr_SIAt_perturb[:,np.newaxis]),axis=1)
            y_pred_tr_SIAt_perturbation = model.predict(x_tr_SIAt_perturbation)
            
            #.............................................
            # RMSE perturbation 
            #.............................................
            pert_rmsd_SIE = rmsd_pert(y_pred_tr_SIE_perturbatuion,y_pred_tr)
            pert_rmsd_SIAt = rmsd_pert(y_pred_tr_SIAt_perturbation,y_pred_tr)
            
            pert_SIE_rmsd_mean, pert_SIE_rmsd_ci = monte_carlo_rmsd_bootstrap(y_pred_tr_SIE_perturbatuion,y_pred_tr)
            pert_SIAt_rmsd_mean, pert_SIAt_rmsd_ci = monte_carlo_rmsd_bootstrap(y_pred_tr_SIAt_perturbation,y_pred_tr)
            
            pert_rmsd_SIEi.append(pert_rmsd_SIE)
            pert_rmsd_SIAti.append(pert_rmsd_SIAt)
            pert_rmsd_SIEi_mean.append(pert_SIE_rmsd_mean)
            pert_rmsd_SIAti_mean.append(pert_SIAt_rmsd_mean)
            pert_rmsd_SIEi_CI.append(pert_SIE_rmsd_ci)
            pert_rmsd_SIAti_CI.append(pert_SIAt_rmsd_ci)
        pert_rmsd_SIEij.append(pert_rmsd_SIEi)
        pert_rmsd_SIAtij.append(pert_rmsd_SIAti)
        pert_rmsd_SIEij_mean.append(pert_rmsd_SIEi_mean)
        pert_rmsd_SIAtij_mean.append(pert_rmsd_SIAti_mean)
        pert_rmsd_SIEij_CI.append(pert_rmsd_SIEi_CI)
        pert_rmsd_SIAtij_CI.append(pert_rmsd_SIAti_CI)
    pert_rmsd_SIEijk.append(pert_rmsd_SIEij)
    pert_rmsd_SIAtijk.append(pert_rmsd_SIAtij)
    pert_rmsd_SIEijk_mean.append(pert_rmsd_SIEij_mean)
    pert_rmsd_SIAtijk_mean.append(pert_rmsd_SIAtij_mean)
    pert_rmsd_SIEijk_CI.append(pert_rmsd_SIEij_CI)
    pert_rmsd_SIAtijk_CI.append(pert_rmsd_SIAtij_CI)

pert_SIE = np.array(pert_rmsd_SIEijk)[:,0,0]
pert_SIAt = np.array(pert_rmsd_SIAtijk)[:,0,0]
pert_SIE_mean = np.array(pert_rmsd_SIEijk_mean)
pert_SIA_mean = np.array(pert_rmsd_SIAtijk_mean)
pert_SIE_CI = np.array(pert_rmsd_SIEijk_CI)[:,0,0,:]
pert_SIA_CI = np.array(pert_rmsd_SIAtijk_CI)[:,0,0,:]
pert_SIA_CI_upper = pert_SIA_CI[:,1]-pert_SIA_mean[:,0,0]
pert_SIA_CI_lower = pert_SIA_mean[:,0,0]-pert_SIA_CI[:,0]
pert_SIE_CI_upper = pert_SIE_CI[:,1]-pert_SIE_mean[:,0,0]
pert_SIE_CI_lower = pert_SIE_mean[:,0,0]-pert_SIE_CI[:,0]

pert_SIA_CI = np.concatenate((pert_SIA_CI_lower[:,np.newaxis],pert_SIA_CI_upper[:,np.newaxis]),axis=1)
pert_SIE_CI = np.concatenate((pert_SIE_CI_lower[:,np.newaxis],pert_SIE_CI_upper[:,np.newaxis]),axis=1)           
        
            
plt.figure(figsize=(12,4))
plt.errorbar(np.arange(1,13),pert_SIE,pert_SIE_CI.T, fmt='o', capsize=5,color='blue',label='importance SIE')
plt.errorbar(np.arange(1,13),pert_SIAt,pert_SIA_CI.T, fmt='o', capsize=5,color='red',label='importance SIAt')
plt.ylabel('importance')
plt.title('PERT') 
plt.axis('tight')  
plt.legend()    
            
filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/XAI/'
savepath = filepath+'analysis_perturbation_output_JAN-DEC_tau0_T0.npz'
np.savez(savepath, pert_SIE = pert_SIE, pert_SIAt = pert_SIAt, pert_SIE_CI = pert_SIE_CI, pert_SIA_CI = pert_SIA_CI)