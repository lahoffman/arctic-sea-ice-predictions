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
import innvestigate
tf.compat.v1.disable_eager_execution()


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
# II. TEST DATA: OBSERVATIONS, 1979-2014
#------------------------------------------------------
#------------------------------------------------------

# a. NSIDC - SIEXTENTN
#------------------------------------------------------------
#residual from cmip6 ensemble mean (1979-2014) + ssp585 (2015-2024)
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
sie_observed = np.array(dataset.variables['sie_obs'])[:,:,1:]
sie_obs_te = np.array(dataset.variables['residual_mean_weighted'])[:,:,1:]
fit_obs = np.array(dataset.variables['fit_mean_weighted'])[:,:,1:]
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])

res_std = np.nanstd(sie_obs_te,axis=2)
res_mean = np.nanmean(sie_obs_te,axis=2)

# b. PIOMAS - AREA(SIVOL > THRESHOLD)
#------------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siat_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
area_observed = np.array(dataset.variables['siat_obs'])
area_obs_te = np.array(dataset.variables['residual_mean_weighted'])
area_fit_obs = np.array(dataset.variables['fit_mean_weighted'])


#standardize
#----------------------------------------
outer = []
outera = []
for i in range(17):
    inner = []
    innera = []
    for j in range(10):
        te = sie_obs_te[j,i,:]
        test = sigma_sie[i,0,0,j]
        tem = miu_sie[i,0,0,j]
        ted = np.divide(te-tem,test)       
        inner.append(ted)
        
        tea = area_obs_te[j,i,:]
        testa = sigma_siat[i,0,0,j]
        tema = miu_siat[i,0,0,j]
        teda = np.divide(tea-tema,testa)       
        innera.append(teda)
        
    outer.append(inner)
    outera.append(innera)
sie_obsi = np.array(outer)
area_obsi = np.array(outera)

#reshape test data 
#----------------------------------------
outer = []
outera = []
for i in range(17):
    inner = []
    innera = []
    for j in range(46):
        te = sie_obsi[i,:,j]
        inner.append(te)
        
        tea = area_obsi[i,:,j]
        innera.append(tea)
        
    outer.append(inner)
    outernp = np.array(outer)[np.newaxis,:,:]
    outera.append(innera)
    outernpa = np.array(outera)[np.newaxis,:,:]
    
sie_obs = np.array(outer)   
sie_test = sie_obs[5:,np.newaxis,:,:]

area_obs = np.array(outera)   
area_test = area_obs[5:,np.newaxis,:,:]
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

#define performance metrics
#----------------------------
def skill(y_true, y_pred):
    return (1 - (np.sqrt(np.mean(np.square(y_pred - y_true))))/((np.std(y_true))))

def acc(y_true,y_pred):
    return (np.sum((y_true-np.mean(y_true))*(y_pred-np.mean(y_pred))))/((np.sqrt(np.sum(np.square(y_true-np.mean(y_true)))))*(np.sqrt(np.sum(np.square(y_pred-np.mean(y_pred))))))
            
def cod(y_true,y_pred):
    return (1 - np.divide(np.nanmean(np.square(y_pred - y_true)),np.nanmean(np.square(y_true))))
   
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
filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/model_output/'
savepath = filepath+'bin_centers.npz'
np.savez(savepath, bins=binCenters)

#b. train NN
#there will be a separate NN for each:
#time frame (tf), moving mean(T), and lag time (tau)
#----------------------------------------
skill_outeri = []
corr_outeri = []
coeff_outeri = []
y_pred_outeri = []
y_test_outeri = []
y_pred_ics_i = []

rmse_outeri_obs = []
acc_outeri_obs = []
skill_outeri_obs = []
corr_outeri_obs = []
coeff_outeri_obs = []
y_pred_outeri_obs = []
y_test_outeri_obs = []

#time frames, tf
for i in range(12):
    skill_outerij = []
    corr_outerij = []
    coeff_outerij = []
    y_pred_outerij = []
    y_test_outerij = []
    
    rmse_outerij_obs = []
    acc_outerij_obs = []
    skill_outerij_obs = []
    corr_outerij_obs = []
    coeff_outerij_obs = []
    y_pred_outerij_obs = []
    y_test_outerij_obs = []
    y_pred_ics_ij = []
    
    #moving mean, T
    for j in range(1):
        skill_outerijh = []
        corr_outerijh = []
        coeff_outerijh = []
        y_pred_outerijh = []
        y_test_outerijh = []
        y_pred_ics_ijh = []
        
        rmse_outerijh_obs = []
        acc_outerijh_obs = []
        skill_outerijh_obs = []
        corr_outerijh_obs = []
        coeff_outerijh_obs = []
        y_pred_outerijh_obs = []
        y_test_outerijh_obs = []
        
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
            
            #TESTING DATA: CMIP6 MODELS 
            #----------------------
            #data at t and t+1
            #output is september, t+1
            data2te = siete[8,:,:,j]
            data_t2te = data2te[:,tij[h]:]
            
            #input is other timeframes
            data11te = siete[i,:,:,j]
            data12te = areate[i,:,:,j]
            
            #input is same year for: 
            #J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
            data_t11ii = []
            data_t12ii = []
            data_t11iite = []
            data_t12iite = []
            if t0[i] == 0:
                data_t11 = data11[:,:-tij[h]]
                data_t12 = data12[:,:-tij[h]]
                data_t11te = data11te[:,:-tij[h]]
                data_t12te = data12te[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_t11 = data11[:,1:]
                    data_t12 = data12[:,1:]
                    data_t11te = data11te[:,1:]
                    data_t12te = data12te[:,1:]
                else:
                    data_t11 = data11[:,1:-tmj[h]]
                    data_t12 = data12[:,1:-tmj[h]]
                    data_t11te = data11te[:,1:-tmj[h]]
                    data_t12te = data12te[:,1:-tmj[h]]
            
            
            nt = data_t11.shape[1]
            nm = data_t11.shape[0]
            data_t11_reshape = np.reshape(data_t11,[nm*nt,1])
            data_t12_reshape = np.reshape(data_t12,[nm*nt,1])
            data_t2_reshapei = np.reshape(data_t2,[nm*nt,1])            
            data_t2_reshapei = data_t2_reshapei[~np.isnan(data_t11_reshape)][:, np.newaxis]
            data_t12_reshape = data_t12_reshape[~np.isnan(data_t12_reshape)][:, np.newaxis]
            data_t11_reshape = data_t11_reshape[~np.isnan(data_t11_reshape)][:, np.newaxis]
            data_t1_reshape = np.concatenate((data_t11_reshape,data_t12_reshape),axis=1)
            
            
            ntte = data_t11te.shape[1]
            nmte = data_t11te.shape[0]
            data_t11_reshapete = np.reshape(data_t11te,[nmte*ntte,1])
            data_t12_reshapete = np.reshape(data_t12te,[nmte*ntte,1])
            data_t2_reshapeite = np.reshape(data_t2te,[nmte*ntte,1])            
            data_t2_reshapeite = data_t2_reshapeite[~np.isnan(data_t11_reshapete)][:, np.newaxis]
            data_t12_reshapete = data_t12_reshapete[~np.isnan(data_t12_reshapete)][:, np.newaxis]
            data_t11_reshapete = data_t11_reshapete[~np.isnan(data_t11_reshapete)][:, np.newaxis]
            data_t1_reshapete = np.concatenate((data_t11_reshapete,data_t12_reshapete),axis=1)
            
            
            #convert training output to fuzzy bins
            data_t2_reshape = convert_fuzzyBins(data_t2_reshapei[:,0],binCenters)
            data_t2_reshapete = convert_fuzzyBins(data_t2_reshapeite[:,0],binCenters)
           
            #TESTING DATA: OBS
            #----------------------
            #output is september, t+1
            data_test2 = sie_test[8,:,:,j]
            data_test_t2 = data_test2[:,tij[h]:]
            
            #input is other timeframes
            data_test11 = sie_test[i,:,:,j]
            data_test12 = area_test[i,:,:,j]
            
            #input is same year for: 
            #J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
            data_test_t11ii = []
            data_test_t12ii = []
            if t0[i] == 0:
                data_test_t11 = data_test11[:,:-tij[h]]
                data_test_t12 = data_test12[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_test_t11 = data_test11[:,1:]
                    data_test_t12 = data_test12[:,1:]
                else:
                    data_test_t11 = data_test11[:,1:-tmj[h]]
                    data_test_t12 = data_test12[:,1:-tmj[h]]
                    
            nt_test = data_test_t11.shape[1]
            nm_test = data_test_t11.shape[0]
            nmt = nm_test*nt_test
            data_test_t11_reshape = np.reshape(data_test_t11,[nm_test*nt_test,1])
            data_test_t12_reshape = np.reshape(data_test_t12,[nm_test*nt_test,1])
            data_test_t2_reshape = np.reshape(data_test_t2,[nm_test*nt_test,1])
            
            data_test_t2_reshape = data_test_t2_reshape[~np.isnan(data_test_t11_reshape)]
            data_test_t12_reshape = data_test_t12_reshape[~np.isnan(data_test_t11_reshape)]
            data_test_t11_reshape = data_test_t11_reshape[~np.isnan(data_test_t11_reshape)]
            
            
            #convert output to fuzzy classification
            data_test_t2_reshape = convert_fuzzyBins(data_test_t2_reshape,binCenters)
            
            data_test_t2_reshape = data_test_t2_reshape[:, np.newaxis]
            data_test_t11_reshape = data_test_t11_reshape[:, np.newaxis]
            data_test_t12_reshape = data_test_t12_reshape[:, np.newaxis]
            data_test_t1_reshape = np.concatenate((data_test_t11_reshape,data_test_t12_reshape),axis=1)
            
            #observations
            xte_obs = data_test_t1_reshape 
            yte_obs = data_test_t2_reshape 
            
            #partition into train, test, validation
            xte = data_t1_reshapete
            yte = data_t2_reshapete
            
            xt = data_t1_reshape
            yt = data_t2_reshape
            
                      
            x_train, x_val, y_train, y_val = train_test_split(xt,yt,test_size = .175, shuffle=True, random_state = 12)
            nch = 1
            input_dim = x_train.shape[1]
            
            filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/model_output/'
            loadpath = filepath+'model_2DNN_siesia2sie_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.h5'
            model = tf.keras.models.load_model(loadpath)
            model.summary()
            
            model_no_softmax = innvestigate.model_wo_softmax(model)
            model_no_softmax.summary()
            mn = np.arange(23)
            
            filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/XAI/'
            analyzer = innvestigate.create_analyzer('gradient',model_no_softmax)
            analysis = analyzer.analyze(x_train)
            savepath = filepath+'analysis_gradient_output_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.npz'
            np.savez(savepath, analysis=analysis, x_train=x_train, y_train=y_train)
            
            
            x_train_reshaped = x_train.reshape(-1, 1, 1, 2)
            analyzer = innvestigate.create_analyzer('smoothgrad',model_no_softmax)
            analysis = analyzer.analyze(x_train)
            savepath = filepath+'analysis_smoothgrad_output_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.npz'
            np.savez(savepath, analysis=analysis, x_train=x_train, y_train=y_train)
           
          
            analyzer = innvestigate.create_analyzer('deep_taylor',model_no_softmax)
            analysis = analyzer.analyze(x_train)  
            savepath = filepath+'analysis_deep_taylor_output_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.npz'
            np.savez(savepath, analysis=analysis, x_train=x_train, y_train=y_train)

            
            x_train_reshaped = x_train.reshape(-1, 1, 1, 2)
            analyzer = innvestigate.create_analyzer('integrated_gradients',model_no_softmax)
            analysis = analyzer.analyze(x_train)  
            savepath = filepath+'analysis_integrated_gradients_output_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.npz'
            np.savez(savepath, analysis=analysis, x_train=x_train, y_train=y_train)
            
            
            analyzer = innvestigate.create_analyzer('lrp.z',model_no_softmax)
            analysis = analyzer.analyze(x_train) 
            savepath = filepath+'analysis_lrpz_output_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.npz'
            np.savez(savepath, analysis=analysis, x_train=x_train, y_train=y_train)
            
            
            analyzer = innvestigate.create_analyzer('lrp.epsilon',model_no_softmax)
            analysis = analyzer.analyze(x_train) 
            savepath = filepath+'analysis_lrpepsilon_output_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.npz'
            np.savez(savepath, analysis=analysis, x_train=x_train, y_train=y_train)

            analyzer = innvestigate.create_analyzer('input_t_gradient',model_no_softmax)
            analysis = analyzer.analyze(x_train) 
            savepath = filepath+'analysis_input_t_gradient_output_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.npz'
            np.savez(savepath, analysis=analysis, x_train=x_train, y_train=y_train)
            