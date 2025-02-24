
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

#SIEXTENTN
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
siete = np.concatenate((sie_training[:,ts[0]:te[0],:,:],sie_training[:,ts[1]:te[1],:,:],sie_training[:,ts[2]:te[2],:,:],sie_training[:,ts[3]:te[3],:,:]),axis=1)
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
    outer.append(inner)
sie_obsi = np.array(outer)


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

    outer.append(inner)
    outernp = np.array(outer)[np.newaxis,:,:]
    
sie_obs = np.array(outer)   
sie_test = sie_obs[5:,np.newaxis,:,:]
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
LEARNING_RATE = .001 # Learning rate (think step size)
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
filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_1D_sie2sie/model_output/'
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
    for j in range(10):
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
        for h in range(10):
            
            #TRAINING DATA: CMIP6 MODELS 
            #----------------------
            #data at t and t+1
            #output is september, t+1
            data2 = sie[8,:,:,j]
            data_t2 = data2[:,tij[h]:]
            
            #input is other timeframes
            data11 = sie[i,:,:,j]
            
            #TESTING DATA: CMIP6 MODELS 
            #----------------------
            #data at t and t+1
            #output is september, t+1
            data2te = siete[8,:,:,j]
            data_t2te = data2te[:,tij[h]:]
            
            #input is other timeframes
            data11te = siete[i,:,:,j]
            
            #input is same year for: 
            #J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
            data_t11ii = []
            data_t12ii = []
            data_t11iite = []
            data_t12iite = []
            if t0[i] == 0:
                data_t11 = data11[:,:-tij[h]]
                data_t11te = data11te[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_t11 = data11[:,1:]
                    data_t11te = data11te[:,1:]
                else:
                    data_t11 = data11[:,1:-tmj[h]]
                    data_t11te = data11te[:,1:-tmj[h]]
            
            
            nt = data_t11.shape[1]
            nm = data_t11.shape[0]
            data_t11_reshape = np.reshape(data_t11,[nm*nt,1])
            data_t2_reshapei = np.reshape(data_t2,[nm*nt,1])            
            data_t2_reshapei = data_t2_reshapei[~np.isnan(data_t11_reshape)][:, np.newaxis]
            data_t11_reshape = data_t11_reshape[~np.isnan(data_t11_reshape)][:, np.newaxis]
            data_t1_reshape = data_t11_reshape
            
            
            ntte = data_t11te.shape[1]
            nmte = data_t11te.shape[0]
            data_t11_reshapete = np.reshape(data_t11te,[nmte*ntte,1])
            data_t2_reshapeite = np.reshape(data_t2te,[nmte*ntte,1])            
            data_t2_reshapeite = data_t2_reshapeite[~np.isnan(data_t11_reshapete)][:, np.newaxis]
            data_t11_reshapete = data_t11_reshapete[~np.isnan(data_t11_reshapete)][:, np.newaxis]
            data_t1_reshapete = data_t11_reshapete
            
            
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
            
            #input is same year for: 
            #J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)
            data_test_t11ii = []
            data_test_t12ii = []
            if t0[i] == 0:
                data_test_t11 = data_test11[:,:-tij[h]]
            elif t0[i] == 1:
                if h == 0:
                    data_test_t11 = data_test11[:,1:]
                else:
                    data_test_t11 = data_test11[:,1:-tmj[h]]
                    
            nt_test = data_test_t11.shape[1]
            nm_test = data_test_t11.shape[0]
            nmt = nm_test*nt_test
            data_test_t11_reshape = np.reshape(data_test_t11,[nm_test*nt_test,1])
            data_test_t2_reshape = np.reshape(data_test_t2,[nm_test*nt_test,1])
            
            data_test_t2_reshape = data_test_t2_reshape[~np.isnan(data_test_t11_reshape)]
            data_test_t11_reshape = data_test_t11_reshape[~np.isnan(data_test_t11_reshape)]
            
            
            #convert output to fuzzy classification
            data_test_t2_reshape = convert_fuzzyBins(data_test_t2_reshape,binCenters)
            
            data_test_t2_reshape = data_test_t2_reshape[:, np.newaxis]
            data_test_t11_reshape = data_test_t11_reshape[:, np.newaxis]
            data_test_t1_reshape = data_test_t11_reshape
            
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
            
            #MODEL: BUILD NEURAL NETWORK
            #----------------------------------------
            # define the model
            model = tf.keras.models.Sequential()
            model.add(Dense(N_UNITS, activation=ACTIVATION_FUNCTION, kernel_regularizer=tf.keras.regularizers.l2(RL2),input_shape=(input_dim,)))
            model.add(Dense(N_UNITS, activation=ACTIVATION_FUNCTION,kernel_regularizer=tf.keras.regularizers.l2(RL2)))
            model.add(Dense(len(binCenters)))
            model.add(Activation('softmax'))
                
            
            # Update the optimizer to use 'learning_rate' instead of 'lr'
            filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_1D_sie2sie/model_output/'
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss=LOSS, metrics=[METRIC])
            #model.compile(optimizer='sgd',loss=LOSS, metrics=[METRIC])
            model.summary()
            savepath = filepath+'model_1DNN_sie2sie_'+time_frames[i]+'_tau'+str(h)+'_T'+str(j)+'.h5'
            model.save(savepath)
            
            #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

            history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,shuffle=True, validation_data=(x_val, y_val))
            #print(history.history)
            
            
            #evaluate model: plot the loss during training 
            plot1 = plt.figure(1,figsize=(18,9))
            plt.subplot(1,2,1)
            plt.plot(history.history['loss'],label='train')
            plt.plot(history.history['val_loss'],label='validation')
            plt.ylim(0.1,1)
            plt.xlabel('Epoch')
            plt.ylabel(LOSS)
            plt.legend(loc='lower right')
            
            plt.subplot(1,2,2)
            plt.plot(history.history['root_mean_squared_error'],label='train')
            plt.plot(history.history['val_root_mean_squared_error'],label='validation')
            plt.ylim(0,.1)
            plt.xlabel('Epoch')
            plt.ylabel(METRIC)
            plt.legend(loc='lower right')
            plt.show()
            
            
            #.............................................
            # Network Predictions 
            #.............................................
            
            #MODEL PREDICTIONS & EVAL: 'imperfect model' test case
            y_true = yte
            y_pred = model.predict(xte)
            
            skillpm = skill(y_true,y_pred)
            corrpm = acc(y_true,y_pred)
            codpm = cod(y_true,y_pred)
            #print('skill:',skillpm)
            #print('correlation:',corrpm)
            
            #Make the prediction from the model, test 
            predictions = model.predict(xte)
            pred_expectedte =  np.dot(predictions,binCenters)
            true_expectedte = np.dot(yte,binCenters)

            '''
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.plot(true_expectedte, pred_expectedte,'ro')
            plt.plot(true_expectedte,true_expectedte ,'k')
            plt.xlabel('True ')
            plt.ylabel('Predicted ')
            plt.title('Network-Predicted forced response, test') 
            plt.axis('tight')
            plt.show()
            '''
            
            #MODEL PREDICTIONS: observations
            y_true_obs = yte_obs
            y_pred_obs = model.predict(xte_obs)
            
            predictions = model.predict(xte_obs)
            pred_expectedobs =  np.dot(predictions,binCenters)
            true_expectedobs = np.dot(yte_obs,binCenters)

            '''
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.plot(true_expectedobs, pred_expectedobs,'ro')
            plt.plot(true_expectedobs,true_expectedobs ,'k')
            plt.xlabel('True ')
            plt.ylabel('Predicted ')
            plt.title('Network-Predicted forced response, test') 
            plt.axis('tight')
            plt.show()
            '''
            
            #MODEL PREDICTIONS: ICs
            initial_conditions = np.arange(-3,3.05,0.05)
            ics = initial_conditions[:,np.newaxis]
            zeros = np.zeros((121,1))
            x_ics = ics
            y_pred_ics = model.predict(x_ics)
            
            y_pred_ics_ijh.append(y_pred_ics)
            y_pred_outerijh_obs.append(y_pred_obs)
            y_test_outerijh_obs.append(y_true_obs)        
            skill_outerijh.append(skillpm)
            corr_outerijh.append(corrpm)
            coeff_outerijh.append(codpm)
            y_pred_outerijh.append(y_pred)
            y_test_outerijh.append(y_true)
        y_pred_ics_ij.append(y_pred_ics_ijh)
        y_pred_outerij_obs.append(y_pred_outerijh_obs)
        y_test_outerij_obs.append(y_test_outerijh_obs)     
        skill_outerij.append(skill_outerijh)
        corr_outerij.append(corr_outerijh)
        coeff_outerij.append(coeff_outerijh)
        y_pred_outerij.append(y_pred_outerijh)
        y_test_outerij.append(y_test_outerijh)  
    y_pred_ics_i.append(y_pred_ics_ij)
    y_pred_outeri_obs.append(y_pred_outerij_obs)
    y_test_outeri_obs.append(y_test_outerij_obs)  
    skill_outeri.append(skill_outerij)
    corr_outeri.append(corr_outerij)
    coeff_outeri.append(coeff_outerij)
    y_pred_outeri.append(y_pred_outerij)
    y_test_outeri.append(y_test_outerij)

#save: model input, prediction, and performance for 'perfect model' case
correlation_perfectmodel = np.array(corr_outeri)#[:,0,0]
skill_perfectmodel= np.array(skill_outeri)#[:,0,0]
#model_prediction = np.array(y_pred_outeri)#[:,:,:,:,0]
#model_input = np.array(y_test_outeri)#[:,:,:,:,0]
#------------------------------------------------------------
#------------------------------------------------------------



#------------------------------------------------------------
#------------------------------------------------------------
# V. DEFINE PREDICTION TIME
#------------------------------------------------------------
#------------------------------------------------------------
#time frames:J, F, M, A, M, J, J, A, XS, XO, XN, XD
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


nobs = np.array(y_pred_outeri_obs[0][0][0]).shape[0]
npm = np.array(y_pred_outeri[0][0][0]).shape[0]

#save model output, observations
#------------------------------

ypics = []
ypppm = []
yttpm =[]
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
    ypouterics = []
    ypouterpm = []
    ytouterpm = []
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
        ypinnerics = []
        ypinnerpm = []
        ytinnerpm = []
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
            ypobs = np.array(y_pred_outeri_obs[k][i][j])
            ytobs = np.array(y_test_outeri_obs[k][i][j])[:,0,:]
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
            codd = coefficient_of_determination(ytobsexpected,binCenters,ypobs)
            
            #pad with nans for size consistency in np.append
            #year range depends on time frame
            #time = prediction time
            #is there data or NaN in input
            nanpadding1D = np.full((nobs-ntt,),np.nan)
            nanpadding2D = np.full((nobs-ntt,23),np.nan)
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
                      
            
            #perfect model reshape
            ypobspm = np.array(y_pred_outeri[k][i][j])
            ytobspm = np.array(y_test_outeri[k][i][j])

            nttpm = ypobspm.shape[0]

            #pad with nans for size consistency in np.append
            #year range depends on time frame
            #time = prediction time
            #is there data or NaN in input
            nanpadding1Dpm = np.full((npm-nttpm,),np.nan)
            nanpadding2Dpm = np.full((npm-nttpm,23),np.nan)
            ypobs_paddedpm = np.concatenate((nanpadding2Dpm,ypobspm),axis=0)
            ytobs_paddedpm = np.concatenate((nanpadding2Dpm,ytobspm),axis=0)
        
            #initial conditions
            ypicspm = np.array(y_pred_ics_i[k][i][j])
            nttic = ypicspm.shape[0]
            nanpadding1Dic = np.full((121-nttic,),np.nan)
            nanpadding2Dic = np.full((121-nttic,23),np.nan)
            yics_padded = np.concatenate((nanpadding2Dic,ypicspm),axis=0)

            ypinnerics.append(yics_padded)
            ypinnerpm.append(ypobs_paddedpm)
            ytinnerpm.append(ytobs_paddedpm)
            ypinner.append(ypobs_padded)
            ypeinner.append(ypobserr_padded)
            ytinner.append(ytobs_padded)
            skillj.append(skill_obs)
            corrj.append(correlation_obs)
            accj.append(acc_obs)
            rmsej.append(rmse_obs) 
            codj.append(codd)
            relj.append(rel)
        ypouterics.append(ypinnerics)
        ypouterpm.append(ypinnerpm)
        ytouterpm.append(ytinnerpm)
        ypouter.append(ypinner)
        ypeouter.append(ypeinner)
        ytouter.append(ytinner)
        skilli.append(skillj)
        corri.append(corrj)
        acci.append(accj)
        rmsei.append(rmsej)
        codi.append(codj)
        reli.append(relj)
    ypics.append(ypouterics)
    ypppm.append(ypouterpm)
    yttpm.append(ytouterpm)
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
           
model_prediction = np.array(ypppm)
model_input = np.array(yttpm) 

nn_prediction_mean_obs = np.array(ypp)
nn_prediction_err_obs = np.array(yppe)
nn_input_obs = np.array(ytt)
#nn_probability_obs = np.array(yprob)

model_prediction_ics = np.array(ypics)


#------------------------------------------------------------
#------------------------------------------------------------
#SAVE PREDICTION ICS
#------------------------------------------------------------
#------------------------------------------------------------
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_1D_sie2sie_ICs.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',model_prediction_ics.shape[0]) #no. time frames
    file.createDimension('ntau',model_prediction_ics.shape[1]) #hindcast lag
    file.createDimension('nT',model_prediction_ics.shape[2]) #no. moving means
    file.createDimension('nt',model_prediction_ics.shape[3]) #no. timesteps, obs
    file.createDimension('nb',model_prediction_ics.shape[4]) #no. bins
    file.createDimension('ni',x_ics.shape[1]) #no. bins

    #create variables
    #model input and output
    inptics = file.createVariable('x_ics','f4',('nt','ni'))
    predics = file.createVariable('model_prediction_ics','f4',('tf','ntau','nT','nt','nb'))

    #write data to variables 
    inptics[:] = x_ics
    predics[:] = model_prediction_ics

    
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
#save model probabilities, observations
#------------------------------------------------------------
#------------------------------------------------------------

ypp = []
ytt = []
#time frames, tf
for k in range(12):
    ypouter = []
    ytouter = []     
    #moving mean, T
    for i in range(10):
        ypinner = []
        ypeinner = []
        ytinner = []
        yprobinner = []
        #lag time, tau
        for j in range(10):
            ypobs = np.array(y_pred_outeri_obs[k][i][j])
            ytobs = np.array(y_test_outeri_obs[k][i][j])[:,0,:]
            ytime = yearpred[k,i,:]
            
            #pad with nans for size consistency in np.append
            #year range depends on time frame
            #time = prediction time
            #is there data or NaN in input
            ntt = ypobs.shape[0]
            nanpadding2D = np.full((nobs-ntt,23),np.nan)
            ypprob_padded = np.concatenate((nanpadding2D,ypobs),axis=0)
            ytprob_padded = np.concatenate((nanpadding2D,ytobs),axis=0)
            
                      
            ypinner.append(ypprob_padded)
            ytinner.append(ytprob_padded)
        ypouter.append(ypinner)
        ytouter.append(ytinner)
    ypp.append(ypouter)
    ytt.append(ytouter)
nn_probability_pred = np.array(ypp)
nn_probability_input = np.array(ytt)
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
# VIII. EVALUATE PERSISTENCE
#------------------------------------------------------------
#------------------------------------------------------------

#performance, persistence
#------------------------------
def corr_ps(y_true_obs,y_pred_obs):
    return (np.nansum((y_true_obs-np.nanmean(y_true_obs))*(y_pred_obs-np.nanmean(y_pred_obs))))/((np.sqrt(np.nansum(np.square(y_true_obs-np.nanmean(y_true_obs)))))*(np.sqrt(np.nansum(np.square(y_pred_obs-np.nanmean(y_pred_obs))))))

def rmse_ps(y_true_obs,y_pred_obs):
    return np.sqrt(np.divide(np.nansum(np.square(y_pred_obs-y_true_obs)),nmt))

def coeff_det_ps(y_true_obs,y_pred_obs):
    return 1 - np.divide(np.nanmean(np.square(y_pred_obs - y_true_obs)),np.nanmean(np.square(y_true_obs)))

tmj = np.arange(0,10)

pscorri = []
psrmsei= []
pscoeffdi = []
for i in range(12):
    pscorrj = []
    psrmsej = []
    pscoeffdj = []
    for j in range(10):
        pscorrh = []
        psrmseh = []
        pscoeffdh = []
        for h in range(1,11):
            nt1 = sie_test.shape[1]
            nt2 = sie_test.shape[2]
                       
            #output is september
            persistence_output = np.reshape(sie_test[8,:,h:,j],(np.multiply(nt1,[nt2-h])))
            
            #input is same year for: 
            #J, F, M, A, M, J, J, A, XS, XO, XN, XD
            t0 = [1,1,1,1,1,1,1,1,0,0,0,0] #1 = use t+1 (same year as SEP); 0 = use t (previous year from SEP)

            if t0[i] == 0:
                ps_t1 = sie_test[i,:,:-h,j]
            elif t0[i] == 1:
                if h == 1:
                    ps_t1 = sie_test[i,:,1:,j]
                else:
                    ps_t1 = sie_test[i,:,1:-tmj[h-1],j]
           
            persistence_input = np.reshape(ps_t1,(np.multiply(nt1,[nt2-h])))        
            ps_corr = corr_ps(persistence_input,persistence_output)
            ps_rmse = rmse_ps(persistence_input,persistence_output)
            ps_coeffd = coeff_det_ps(persistence_input,persistence_output)
            pscorrh.append(ps_corr)
            psrmseh.append(ps_rmse)
            pscoeffdh.append(ps_coeffd)
        pscorrj.append(pscorrh)
        psrmsej.append(psrmseh)
        pscoeffdj.append(pscoeffdh)
    pscorri.append(pscorrj)
    psrmsei.append(psrmsej)
    pscoeffdi.append(pscoeffdj)
        
persistence_acc = np.array(pscorri)
persistence_rmse = np.array(psrmsei)
persistence_cod= np.array(pscoeffdi)
#------------------------------------------------------------
#------------------------------------------------------------


'''
#------------------------------------------------------------
#------------------------------------------------------------
#SAVE PREDICTION (PROBABILSTIC AND DETERMINISTIC) AND PERFORMANCE TO NETCDF
#------------------------------------------------------------
#------------------------------------------------------------
savepath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/NN_1D_sie2sie_performance.nc'
with nc.Dataset(savepath,'w') as file:
    #create dimensions
    file.createDimension('tf',nn_acc.shape[0]) #no. time frames
    file.createDimension('ntau',nn_acc.shape[1]) #hindcast lag
    file.createDimension('nT',nn_acc.shape[2]) #no. moving means
    file.createDimension('nt',nn_prediction_mean_obs.shape[3]) #no. timesteps, obs
    file.createDimension('ntm',model_input.shape[3]) #no. timesteps, model
    file.createDimension('nb',model_input.shape[4]) #no. bins
  

    #create variables
    #NN performance
    acc = file.createVariable('nn_acc','f4',('tf','nT','ntau')) 
    rmse = file.createVariable('nn_rmse','f4',('tf','nT','ntau'))
    coeffdet = file.createVariable('nn_cod','f4',('tf','nT','ntau'))  
    rel = file.createVariable('nn_rel','f4',('tf','nT','ntau'))
    
    #persistence performance
    accps = file.createVariable('persistence_acc','f4',('tf','nT','ntau')) 
    codps = file.createVariable('persistence_cod','f4',('tf','nT','ntau')) 
    rmseps = file.createVariable('persistence_rmse','f4',('tf','nT','ntau')) 
    
    #model input and output
    inpt = file.createVariable('nn_input_obs','f4',('tf','ntau','nT','nt'))
    pred = file.createVariable('nn_prediction_mean_obs','f4',('tf','ntau','nT','nt'))
    prederr = file.createVariable('nn_prediction_err_obs','f4',('tf','ntau','nT','nt'))
    inptmodel = file.createVariable('model_input','f4',('tf','ntau','nT','ntm','nb'))
    predmodel = file.createVariable('model_prediction','f4',('tf','ntau','nT','ntm','nb'))
    probpred = file.createVariable('nn_probability_pred','f4',('tf','ntau','nT','nt','nb'))
    probinpout = file.createVariable('nn_probability_input','f4',('tf','ntau','nT','nt','nb'))

    #write data to variables
    acc[:] = nn_acc
    rmse[:] = nn_rmse
    coeffdet[:] = nn_cod
    rel[:] = nn_rel
    
    accps[:] = persistence_acc
    codps[:] = persistence_cod
    rmseps[:] = persistence_rmse
    
    inpt[:] = nn_input_obs
    pred[:] = nn_prediction_mean_obs
    prederr[:] = nn_prediction_err_obs
    inptmodel[:] = model_input
    predmodel[:] = model_prediction
    probpred[:] = nn_probability_pred
    probinput = nn_probability_input
    
#------------------------------------------------------------
#------------------------------------------------------------
'''

'''

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,8))
titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

ax1.plot(np.arange(1,13),Acorrelation_obs[5:,])
ax1.plot(np.arange(0,14),np.zeros(14),color='black',linewidth=1)
ax1.set_xticks(np.arange(1,13),labels=titles[5:],rotation=45)
ax1.set_ylabel('correlation')

ax2.plot(np.arange(1,13),Aroot_mse_obs[5:,])
ax2.plot(np.arange(0,14),np.zeros(14),color='black',linewidth=1)
ax2.set_xticks(np.arange(1,13),labels=titles[5:],rotation=45)
ax2.set_ylabel('rmse')

ax3.plot(np.arange(1,13),Arel[5:,])
ax3.plot(np.arange(0,14),np.ones(14),color='black',linewidth=1)
ax3.set_xticks(np.arange(1,13),labels=titles[5:],rotation=45)
ax3.set_ylabel('reliability')
'''

'''    
#-----------------------------------------------
#-----------------------------------------------
#plot time frame versus averaging time
#xaxis is time frame (JAN-DEC)
#each figure is a different hindcast lag
#-----------------------------------------------
#-----------------------------------------------
mm1 = 5 #JAN
mm2 = 17 #DEC

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
    
    metric1 = coefficient_determination_obs
    metric1ps = persistence_coeffd
    metric2 = root_mse_obs
    metric2ps = persistence_rmse
    
    # correlation
    contour_levels = np.linspace(-1, 1, 50)  # Adjust levels as needed
    data1 = np.transpose(metric1[mm1:mm2,:,i])
    persistence = np.transpose(metric1ps[mm1:mm2,:,i])
    maski = data1 > persistence
    data_rel = np.transpose(metric2[mm1:mm2,:,i])
    data_corr = np.transpose(metric1[mm1:mm2,:,i])
    
    
    data11 = np.concatenate((data1[:,8:12],data1[:,0:8]),axis=1)
    maski11 = np.concatenate((maski[:,8:12],maski[:,0:8]),axis=1)
    #mask11 = np.concatenate((mask[:,8:12],mask[:,0:8]),axis=1) 
    
    if i == 0:
        d1 = np.flip(data11,axis=1)
        mi1 = np.flip(maski11,axis=1)
        #m1 = np.flip(mask11,axis=1)
    else:
        d1 = np.append(d1,np.flip(data11,axis=1),axis=1)
        mi1 = np.append(mi1,np.flip(maski11,axis=1),axis=1)
        #m1 = np.append(m1,np.flip(mask11,axis=1),axis=1)
    
    

          
    #rmse
    contour_levels = np.linspace(0, 2, 30)  # Adjust levels as needed
    data2 = np.transpose(metric2[mm1:mm2,:,i])
    num_elements_greater_than_two = np.sum(data2 > 2)
    masked_Z = np.ma.masked_greater(data2, 2)
    data_rel = np.transpose(metric2[mm1:mm2,:,i])
    data_corr = np.transpose(metric1[mm1:mm2,:,i])
    persistence = np.transpose(metric2ps[mm1:mm2,:,i])
    #mask = ((0.8 < data_rel) & (data_rel < 1.2)) & (data_corr > 0.1) & (data_corr > persistence)
    #maskint = mask.astype(int)
    #X, Y = np.meshgrid(xx,ff)
    
    data22 = np.concatenate((data2[:,8:12],data2[:,0:8]),axis=1)
    #mask22 = np.concatenate((mask[:,8:12],mask[:,0:8]),axis=1) 
    
    if i == 0:
        d2 = np.flip(data22,axis=1)
        #m2 = np.flip(mask22,axis=1)
    else:
        d2= np.append(d2,np.flip(data22,axis=1),axis=1)
        #m2= np.append(m2,np.flip(mask22,axis=1),axis=1)
       
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
#mask11 = m1
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
#axs[0].scatter(X[mask11], Y[mask11], s=300,color='maroon')

if num_elements_less_than_minus_one > 0:
    axs[0].contourf(xx,ff, data11, levels=[data11.min(), -1], colors='dimgrey')

cb1 = fig.colorbar(cp1, ax=axs[0],aspect=10)
cb1.set_label('correlation', fontsize=label_mid) 
cb1.set_ticks(custom_ticks1)
cb1.ax.tick_params(labelsize=label_small)
axs[0].set_xticks(xx)
axs[0].set_xticklabels(ticki,fontsize=label_small)
axs[0].set_yticklabels(np.arange(1,11),fontsize=label_small)


data22 = d2
#mask22 = m2

#reliability
contour_levels = np.linspace(0, 2, 30)  # Adjust levels as needed
num_elements_greater_than_two = np.sum(data22 > 2)
cp2 = axs[1].contourf(xx, ff, data22, levels=contour_levels, cmap='BrBG_r',vmin=0, vmax=2)
axs[1].contour(xx,ff,data22,levels = [0.5],colors='darkolivegreen')
axs[1].contour(xx,ff,data22,levels = [1.5],colors='peru')
#axs[1].scatter(X[mask22], Y[mask22],s=300, color='peru')
if num_elements_greater_than_two > 0:
    axs[1].contourf(xx,ff, data22, levels=[2,data22.max()], colors='saddlebrown')
     
cb2 = fig.colorbar(cp2, ax=axs[1],aspect=10)
cb2.set_label('rmse', fontsize=label_mid) 
cb2.set_ticks(custom_ticks2)
cb2.ax.tick_params(labelsize=label_small)
axs[1].set_xticks(xx)
axs[1].set_xticklabels(ticki,fontsize=label_small)

axs[1].set_yticklabels(np.arange(1,11),fontsize=label_small)
axs[1].set_ylabel('                                      averaging time [year]', fontsize=label_big)
axs[1].set_xlabel('hindcast lag [previous month]', fontsize=label_big,labelpad=100)
'''


