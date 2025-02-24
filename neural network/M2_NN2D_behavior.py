
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
#time frames, tf
for i in range(12):
    y_pred_ics_ij = []
    x_ics_ij = []  
    #moving mean, T
    for j in range(10):
        y_pred_ics_ijh = []  
        x_ics_ijh = []  
        #lag time, tau
        for h in range(10):
            
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
            
            #MODEL PREDICTIONS: ICs
            initial_conditions = np.arange(-3,3.1,0.1)
            ics = initial_conditions[:,np.newaxis]
            y_pred_ics_g = []
            x_ics_g = []
            for g in range(61):
                ics_siat = initial_conditions[g]*np.ones((61,1))
                x_ics = np.concatenate((ics,ics_siat),axis = 1)
                y_pred_ics = model.predict(x_ics)
                y_pred_ics_g.append(y_pred_ics)
                x_ics_g.append(x_ics)
            y_pred_ics_ijh.append(y_pred_ics_g)
            x_ics_ijh.append(x_ics_g)
        y_pred_ics_ij.append(y_pred_ics_ijh)
        x_ics_ij.append(x_ics_ijh)
    y_pred_ics_i.append(y_pred_ics_ij)
    x_ics_i.append(x_ics_ij)


model_prediction_ics = np.array(y_pred_ics_i)
x_ics = np.array(x_ics_i)
#------------------------------------------------------------
#------------------------------------------------------------



#------------------------------------------------------------
#------------------------------------------------------------
#SAVE PREDICTION ICS
#------------------------------------------------------------
#------------------------------------------------------------
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_2D_siearea2sie_bins_ICs_2.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',model_prediction_ics.shape[0]) #no. time frames
    file.createDimension('ntau',model_prediction_ics.shape[1]) #hindcast lag
    file.createDimension('nT',model_prediction_ics.shape[2]) #no. moving means
    file.createDimension('nt',model_prediction_ics.shape[3]) #no. ICs
    file.createDimension('nic',model_prediction_ics.shape[4]) #no. ICs
    file.createDimension('nb',model_prediction_ics.shape[5]) #no. bins
    file.createDimension('ni',x_ics.shape[5]) #no. bins

    #create variables
    #model input and output
    inptics = file.createVariable('x_ics','f4',('tf','ntau','nT','nt','nic','ni'))
    predics = file.createVariable('model_prediction_ics','f4',('tf','ntau','nT','nt','nic','nb'))

    #write data to variables 
    inptics[:] = x_ics
    predics[:] = model_prediction_ics

    
#------------------------------------------------------------
#------------------------------------------------------------

