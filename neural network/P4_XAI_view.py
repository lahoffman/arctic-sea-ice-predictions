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
print(tf.__version__)

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

time_frames = ['yearly','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
#11-12: July
#7-8: March
start_mo = 11
end_mo = 12
tau = 0 
T = 0

for j in range(start_mo,end_mo):
    filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/model_output/'
    loadpathp = filepath+'prediction_output_'+time_frames[j]+'_tau'+str(tau)+'_T'+str(T)+'.npz'
    #loadpathp = filepath+'prediction_output.npz'
    data = np.load(loadpathp)
    pred_tr = data['pred_tr']
    true_tr = data['true_tr']
    pred_va = data['pred_va']
    true_va = data['true_va']
    pred_te = data['pred_te']
    true_te = data['true_te']
    pred_obs = data['pred_obs']
    true_obs = data['true_obs']
    pred = data['pred']


    xai_method = ['gradient','smoothgrad','integrated_gradients','deep_taylor','input_t_gradient','lrpepsilon','lrpz']
    #xai_method = ['gradient','deep_taylor','input_t_gradient','lrpepsilon','lrpz']
    ii = np.reshape(np.arange(1,22),[3,7])
    
    err = np.abs(pred_tr-true_tr)
    err2 = np.tile(err,[2,1]).T
    threshold = .2

    plt.figure(figsize=(32,8))
    for k in range(7):
        filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/XAI/'
        loadpath = filepath+'analysis_'+xai_method[k]+'_output_'+time_frames[j]+'_tau'+str(tau)+'_T'+str(T)+'.npz'
        
        data = np.load(loadpath)
        analysis = data['analysis']
        x_train = data['x_train']
        y_train = data['y_train'] 
        inputSIE = x_train[:,0]
        inputSIAt = x_train[:,1]
        outputSIE = true_tr
        
        analysis[err2>threshold] = np.nan
        analysisSIE = analysis[:,0]
        analysisSIAt = analysis[:,1]

        bins = np.arange(-4.,4.,0.1)
        siebins = np.float64(bins)
        siabins = np.float64(bins)
        bini = np.arange(-3.9,4,0.1)
        nb = bins.shape[0]
        
        inputSIE = inputSIE.astype(np.float64)
        inputSIAt = inputSIAt.astype(np.float64)
        
        sie_out_bin_avg = np.zeros((len(bins) - 1, len(bins) - 1))
        sie_rel_bin_avg = np.zeros((len(bins) - 1, len(bins) - 1))
        siat_rel_bin_avg = np.zeros((len(bins) - 1, len(bins) - 1))
        
        for i in range(nb-1):
            for h in range(nb-1):
                sie_bin_min, sie_bin_max = siebins[i], siebins[i+1]
                siat_bin_min, siat_bin_max = siabins[h], siabins[h+1]

                sie_indices = (inputSIE >= sie_bin_min) & (inputSIE < sie_bin_max)
                siat_indices = (inputSIAt >= siat_bin_min) & (inputSIAt < siat_bin_max)
                common_indices = sie_indices & siat_indices
                
                if np.any(common_indices):
                    sie_rel_bin_avg[i,h] = np.nanmean(analysisSIE[common_indices])
                    siat_rel_bin_avg[i,h] = np.nanmean(analysisSIAt[common_indices])
                    sie_out_bin_avg[i,h] = np.nanmean(outputSIE[common_indices])
                else:
                    sie_rel_bin_avg[i,h] = np.nan
                    siat_rel_bin_avg[i,h] = np.nan
                    sie_out_bin_avg[i,h] = np.nan
        
        #CONTOUR: SIE vs SIAt w/ COLORS IN RELEVANCE
        plt.subplot(3,7,ii[0,k])
        plt.contourf(bini,bini,sie_rel_bin_avg,cmap='seismic',levels=20,vmin=-.75, vmax=.75)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        mean_relevance = np.nanmean(sie_rel_bin_avg)
        text = f"mean relevance: {mean_relevance:.2f}"
        plt.text(1.0, 1.0, text, fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

        if k == 0:
            plt.ylabel('SIAt')
        if k == 6:
            plt.colorbar(label='Relevance (SIE)')
        plt.title(xai_method[k])

        plt.subplot(3,7,ii[1,k])
        plt.contourf(bini,bini,siat_rel_bin_avg,cmap='seismic',levels=20,vmin=-.75, vmax=.75)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        mean_relevance = np.nanmean(siat_rel_bin_avg)
        text = f"mean relevance: {mean_relevance:.2f}"
        plt.text(1.0, 1.0, text, fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

        if k == 0:
            plt.ylabel('SIAt')
        if k == 6:
            plt.colorbar(label='Relevance (SIAt)')
        
        plt.subplot(3,7,ii[2,k])
        plt.contourf(bini,bini,sie_out_bin_avg,cmap='seismic',levels=20,vmin=-2, vmax=2)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        plt.xlabel('SIE')
        if k == 0:
            plt.ylabel('SIAt')
        if k == 6:
            plt.colorbar(label='SIE, SEP')
    
    
    
    #TIME SERIES RELEVANCE
    import matplotlib.cm as cm
    
    #k = XAI method
    for k in range(3,4):
        filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/XAI/'
        loadpath = filepath+'analysis_'+xai_method[k]+'_output_'+time_frames[j]+'_tau'+str(tau)+'_T'+str(T)+'.npz'
        
        data = np.load(loadpath)
        analysis = data['analysis']
        x_train = data['x_train']
        y_train = data['y_train'] 
        inputSIE = x_train[:,0]
        inputSIAt = x_train[:,1]
        outputSIE = true_tr
        
        analysisnp = np.concatenate((analysis,np.full((111,2),np.nan)),axis=0)
        analysisrsie = np.reshape(analysis[:20008,0],[122,164])
        analysisrsia = np.reshape(analysis[:20008,1],[122,164])
        
        inputsienp = np.concatenate((inputSIE,np.full((111,),np.nan)),axis=0)
        inputsianp = np.concatenate((inputSIAt,np.full((111,),np.nan)),axis=0)
        outputsienp = np.concatenate((outputSIE,np.full((111,),np.nan)),axis=0)
        inputsier = np.reshape(inputSIE[:20008,],[122,164])
        inputsiar = np.reshape(inputSIAt[:20008,],[122,164])
        outputsier = np.reshape(outputSIE[:20008,],[122,164])
        
        analysissiem = np.nanmean(analysisrsie,axis=0)
        analysissiam = np.nanmean(analysisrsia,axis=0)
        inputsiem = np.nanmean(inputsier,axis=0)
        inputsiam = np.nanmean(inputsiar,axis=0)
        
        analysisSIE = np.divide(analysissiem,(analysissiem+analysissiam))*100
        analysisSIAt = np.divide(analysissiam,(analysissiem+analysissiam))*100
        
        plt.figure(figsize=(24,16))
        plt.subplot(2,1,1)
        plt.plot(np.arange(1850,2014),analysisSIE,'-',label='SIE relevance',color='seagreen')
        plt.plot(np.arange(1850,2014),analysisSIAt,'-',label='SIAt relevance',color='darkorchid')
        plt.xlabel('year')
        plt.ylabel('relevance percentile (%)')
        plt.title(xai_method[k])
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(np.arange(1850,2014),inputsiem,'-',label='SIE input',color='seagreen')
        plt.plot(np.arange(1850,2014),inputsiam,'-',label='SIAt input',color='darkorchid')
        plt.xlabel('year')
        plt.ylabel('standardized internal variability')
        plt.legend()
        
        
        
        
             
        

    
   
     
    '''
    # QUADRANT MEAN
    plt.figure(figsize=(32,8))
    for i in range(7):
        loadpath = filepath+'analysis_'+xai_method[k]+'_output_'+time_frames[j]+'_tau'+str(tau)+'_T'+str(T)+'.npz'
        
        data = np.load(loadpath)
        analysis = data['analysis']
        x_train = data['x_train']
        y_train = data['y_train'] 
        
        analysissie = analysis[:,0]
        analysissiat = analysis[:,1]
        
        analysissie1 = np.nanmean(analysissie[(x_train[:,0]>0) & (x_train[:,1]>0)])
        analysissie2 = np.nanmean(analysissie[(x_train[:,0]>0) & (x_train[:,1]<=0)])
        analysissie3 = np.nanmean(analysissie[(x_train[:,0]<=0) & (x_train[:,1]<=0)])
        analysissie4 = np.nanmean(analysissie[(x_train[:,0]<=0) & (x_train[:,1]>0)])
        analysisSIE = [analysissie1,analysissie2,analysissie3,analysissie4]
        
        analysissiat1 = np.nanmean(analysissiat[(x_train[:,0]>0) & (x_train[:,1]>0)])
        analysissiat2 = np.nanmean(analysissiat[(x_train[:,0]>0) & (x_train[:,1]<=0)])
        analysissiat3 = np.nanmean(analysissiat[(x_train[:,0]<=0) & (x_train[:,1]<=0)])
        analysissiat4 = np.nanmean(analysissiat[(x_train[:,0]<=0) & (x_train[:,1]>0)])
        analysisSIAt = [analysissiat1,analysissiat2,analysissiat3,analysissiat4]
        
        x = [1,1,-1,-1]
        y = [1,-1,-1,1]


        #SCATTER: SIE vs SIAt w/ COLORS IN RELEVANCE
        caxis_min =-0.5
        caxis_max = 0.5
        plt.subplot(3,7,ii[0,i])
        plt.scatter(x,y,c=analysisSIE,cmap='seismic',vmin=caxis_min,vmax=caxis_max)
        plt.plot([0,0],[-2,2],color='k')
        plt.plot([-2,2],[0,0],color='k')
        if i == 0:
            plt.ylabel('SIAt')
        if i == 6:
            plt.colorbar(label='Relevance (SIE)')
        plt.title(xai_method[i])

        plt.subplot(3,7,ii[1,i])
        plt.scatter(x,y,c=analysisSIAt,cmap='seismic',vmin=caxis_min,vmax=caxis_max)
        plt.plot([0,0],[-2,2],color='k')
        plt.plot([-2,2],[0,0],color='k')
        if i == 0:
            plt.ylabel('SIAt')
        if i == 6:
            plt.colorbar(label='Relevance (SIAt)')
    

        plt.subplot(3,7,ii[2,i])
        plt.scatter(x_train[:,0],x_train[:,1],c=true_tr,cmap='PuOr',vmin=-4,vmax=4)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        plt.xlabel('SIE')
        if i == 0:
            plt.ylabel('SIAt')
        if i == 6:
            plt.colorbar(label='SEP SIE')
    '''
  
'''
    
#HISTOGRAM: RELEVANCE
#----------------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(analysis[:,0],bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('SIE relevance')
plt.ylabel('Frequency')
plt.title('XAI') 
plt.axis('tight')


plt.subplot(1,2,2)
plt.hist(analysis[:,1],bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('SIAt relevance')
plt.ylabel('Frequency')
plt.title('XAI') 
plt.axis('tight')
plt.show()
            
#2D HISTOGRAM: SEPTEMBER SIE vs. RELEVANCE
#----------------------------------------
plt.figure(figsize=(12,4))
x = true_tr
plt.subplot(1,2,1)
y = analysis[:,0]
hist, xedges, yedges, im = plt.hist2d(x, y, bins=(20, 20), cmap='viridis')
plt.ylabel('SIE relevance')
plt.xlabel('SIE') 

plt.subplot(1,2,2)
y = analysis[:,1]
hist, xedges, yedges, im = plt.hist2d(x, y, bins=(20, 20), cmap='viridis')
plt.ylabel('SIAt relevance')
plt.xlabel('SIE') 

#2D HISTOGRAM: INPUT vs. RELEVANCE
#----------------------------------------
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
x = x_train[:,0]
y = analysis[:,0]
hist, xedges, yedges, im = plt.hist2d(x, y, bins=(20, 20), cmap='viridis')
plt.plot([0,0],[-0.3,0.3],color='white')
plt.plot([-3,3],[0,0],color='white')
plt.ylabel('SIE relevance')
plt.xlabel('SIE') 
plt.xlim([-3,3])
plt.ylim([-0.3,0.3])

plt.subplot(2,2,2)
x = x_train[:,0]
y = analysis[:,1]
hist, xedges, yedges, im = plt.hist2d(x, y, bins=(20, 20), cmap='viridis')
plt.plot([0,0],[-0.3,0.3],color='white')
plt.plot([-3,3],[0,0],color='white')
plt.ylabel('SIAt relevance')
plt.xlabel('SIE') 
plt.xlim([-3,3])
plt.ylim([-0.3,0.3])

plt.subplot(2,2,3)
x = x_train[:,1]
y = analysis[:,0]
hist, xedges, yedges, im = plt.hist2d(x, y, bins=(20, 20), cmap='viridis')
plt.plot([0,0],[-0.3,0.3],color='white')
plt.plot([-3,3],[0,0],color='white')
plt.ylabel('SIE relevance')
plt.xlabel('SIAt') 
plt.xlim([-3,3])
plt.ylim([-0.3,0.3])

plt.subplot(2,2,4)
x = x_train[:,1]
y = analysis[:,1]
hist, xedges, yedges, im = plt.hist2d(x, y, bins=(20, 20), cmap='viridis')
plt.plot([0,0],[-0.3,0.3],color='white')
plt.plot([-3,3],[0,0],color='white')
plt.ylabel('SIAt relevance')
plt.xlabel('SIAt') 
plt.xlim([-3,3])
plt.ylim([-0.3,0.3])
'''

#monthly
time_frames = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
tfr = ['J','F','M','A','M','J','J','A','S','O','N','D']
from scipy.stats import f_oneway

#load perturbation
filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/XAI/'
loadpath = filepath+'analysis_perturbation_output_JAN-DEC_tau0_T0.npz'
data = np.load(loadpath)
pert_SIE = data['pert_SIE']
pert_SIAt = data['pert_SIAt']
pert_SIE_CI = data['pert_SIE_CI']
pert_SIA_CI = data['pert_SIA_CI']

plt.figure(figsize=(32,22))
for i in range(1):
    ii = [6]
    k = ii[i]
    
    siat_meanj = []
    sie_meanj = []
    siat_stdj = []
    sie_stdj = []
    
    siat_rel_meanj = []
    sie_rel_meanj = []
    siat_rel_stdj = []
    sie_rel_stdj = []
    p_valuej = []
    for j in range(12):
        loadpath = filepath+'analysis_'+xai_method[k]+'_output_'+time_frames[j]+'_tau'+str(tau)+'_T'+str(T)+'.npz'
        data = np.load(loadpath)
        analysis = data['analysis']
        x_train = data['x_train']
        
        f_stat, p_value = f_oneway(*[analysis[:, i] for i in range(analysis.shape[1])])
        print("F-statistic:", f_stat)
        print("p-value:", p_value)
        
        
        sie_rel_mean = np.nanmean(analysis[:,0])
        sie_rel_std = np.nanstd(analysis[:,0])
        siat_rel_mean = np.nanmean(analysis[:,1])
        siat_rel_std = np.nanstd(analysis[:,1])
        
        sie_rel_meanj.append(sie_rel_mean)
        sie_rel_stdj.append(sie_rel_std)
        siat_rel_meanj.append(siat_rel_mean)
        siat_rel_stdj.append(siat_rel_std)
        
        sie_mean = np.nanmean(x_train[:,0])
        sie_std = np.nanstd(x_train[:,0])
        siat_mean = np.nanmean(x_train[:,1])
        siat_std = np.nanstd(x_train[:,1])
        
        sie_meanj.append(sie_mean)
        sie_stdj.append(sie_std)
        siat_meanj.append(siat_mean)
        siat_stdj.append(siat_std)
        
        xai_SIE = np.array(sie_rel_meanj)
        xai_SIE_CI = np.array(sie_rel_stdj)
        xai_SIA = np.array(siat_rel_meanj)
        xai_SIA_CI = np.array(siat_rel_stdj)
        p_valuej.append(p_value)
        
    plt.subplot(3,3,1)
    #plt.errorbar(np.arange(1,13),xai_SIE,xai_SIE_CI.T, fmt='o', capsize=5,color='blue',label='importance SIE')
    #plt.errorbar(np.arange(1,13),xai_SIA,xai_SIA_CI.T, fmt='o', capsize=5,color='red',label='importance SIAt')
    plt.plot(np.arange(1,13),xai_SIE,color='blue',label='relevance SIE',linewidth=3)
    plt.plot(np.arange(1,13),xai_SIA,color='red',label='relevance SIAt',linewidth=3)
    plt.ylabel('relevance',fontsize=18)
    #plt.legend()   
    #plt.xticks(ticks = np.arange(1,13),labels=tfr) 
    plt.xticks([])
    plt.title(f'(a) {xai_method[k]}',fontsize=18)
    plt.axis('tight') 
    plt.yticks(fontsize=18)
    
    if i == 0:
        plt.subplot(3,3,4)
        plt.plot(np.arange(1,13),xai_SIE-xai_SIA,linewidth=3)
        plt.plot(np.arange(1,13),np.zeros(12),color='k')
        plt.ylabel(r'$\Delta$ relevance, SIE-SIAt',fontsize=18) 
        plt.xticks(ticks = np.arange(1,13),labels=tfr,fontsize=18) 
        plt.title('(d)',fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.subplot(3,3,3)
        plt.plot(np.arange(1,13),sie_meanj,linewidth=3,color='k')
        #plt.plot(np.arange(1,13),np.zeros(12),color='k')
        plt.ylabel('SIE',fontsize=18) 
        plt.title('(c) inputs',fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks([])

        plt.subplot(3,3,6)
        plt.plot(np.arange(1,13),siat_meanj,linewidth=3,color='k')
        #plt.plot(np.arange(1,13),np.zeros(12),color='k')
        plt.ylabel('SIAt',fontsize=18) 
        plt.xticks(ticks = np.arange(1,13),labels=tfr,fontsize=18) 
        plt.title('(f)',fontsize=18)
        plt.yticks(fontsize=18)

plt.subplot(3,3,2)
#plt.errorbar(np.arange(1,13),pert_SIE,pert_SIE_CI.T, fmt='o', capsize=5,color='blue',label='importance SIE')
#plt.errorbar(np.arange(1,13),pert_SIAt,pert_SIA_CI.T, fmt='o', capsize=5,color='red',label='importance SIAt')
plt.plot(np.arange(1,13),pert_SIE,color='blue',label='relevance SIE',linewidth=3)
plt.plot(np.arange(1,13),pert_SIAt,color='red',label='relevance SIAt',linewidth=3)
plt.ylabel('relevance',fontsize=18)
plt.title('(b) perturbation',fontsize=18) 
plt.xticks([])
plt.axis('tight')  
plt.yticks(fontsize=18)
plt.legend(fontsize=18)  

plt.subplot(3,3,5)
plt.plot(np.arange(1,13),pert_SIE-pert_SIAt,linewidth=3)
plt.plot(np.arange(1,13),np.zeros(12),color='k')
plt.ylabel(r'$\Delta$ relevance, SIE-SIAt',fontsize=18) 
plt.xticks(ticks = np.arange(1,13),labels=tfr,fontsize=18) 
plt.title('(e)',fontsize=18)
plt.yticks(fontsize=18)


'''
plt.figure(figsize=(32,8))
for j in range(12):
    filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/model_output/'
    loadpathp = filepath+'prediction_output_'+time_frames[j]+'_tau'+str(tau)+'_T'+str(T)+'.npz'
    #loadpathp = filepath+'prediction_output.npz'
    data = np.load(loadpathp)
    pred_tr = data['pred_tr']
    true_tr = data['true_tr']
    pred_va = data['pred_va']
    true_va = data['true_va']
    pred_te = data['pred_te']
    true_te = data['true_te']
    pred_obs = data['pred_obs']
    true_obs = data['true_obs']
    pred = data['pred']


    xai_method = ['gradient','smoothgrad','integrated_gradients','deep_taylor','input_t_gradient','lrpepsilon','lrpz']
    #xai_method = ['gradient','deep_taylor','input_t_gradient','lrpepsilon','lrpz']
    ii = np.reshape(np.arange(1,22),[3,7])
    
    err = np.abs(pred_tr-true_tr)
    err2 = np.tile(err,[2,1]).T
    threshold = .2

    for k in range(6,7):
        filepath = '/Users/hoffmanl/python/.venv/manuscript/d3_bins/M_NN_2D_siesiat2sie/XAI/'
        loadpath = filepath+'analysis_'+xai_method[k]+'_output_'+time_frames[j]+'_tau'+str(tau)+'_T'+str(T)+'.npz'
        
        data = np.load(loadpath)
        analysis = data['analysis']
        x_train = data['x_train']
        y_train = data['y_train'] 
        inputSIE = x_train[:,0]
        inputSIAt = x_train[:,1]
        outputSIE = true_tr
        
        analysis[err2>threshold] = np.nan
        analysisSIE = analysis[:,0]
        analysisSIAt = analysis[:,1]

        bins = np.arange(-4.,4.,0.1)
        siebins = np.float64(bins)
        siabins = np.float64(bins)
        bini = np.arange(-3.9,4,0.1)
        nb = bins.shape[0]
        
        inputSIE = inputSIE.astype(np.float64)
        inputSIAt = inputSIAt.astype(np.float64)
        
        sie_out_bin_avg = np.zeros((len(bins) - 1, len(bins) - 1))
        sie_rel_bin_avg = np.zeros((len(bins) - 1, len(bins) - 1))
        siat_rel_bin_avg = np.zeros((len(bins) - 1, len(bins) - 1))
        
        for i in range(nb-1):
            for h in range(nb-1):
                sie_bin_min, sie_bin_max = siebins[i], siebins[i+1]
                siat_bin_min, siat_bin_max = siabins[h], siabins[h+1]

                sie_indices = (inputSIE >= sie_bin_min) & (inputSIE < sie_bin_max)
                siat_indices = (inputSIAt >= siat_bin_min) & (inputSIAt < siat_bin_max)
                common_indices = sie_indices & siat_indices
                
                if np.any(common_indices):
                    sie_rel_bin_avg[i,h] = np.nanmean(analysisSIE[common_indices])
                    siat_rel_bin_avg[i,h] = np.nanmean(analysisSIAt[common_indices])
                    sie_out_bin_avg[i,h] = np.nanmean(outputSIE[common_indices])
                else:
                    sie_rel_bin_avg[i,h] = np.nan
                    siat_rel_bin_avg[i,h] = np.nan
                    sie_out_bin_avg[i,h] = np.nan
        
        #CONTOUR: SIE vs SIAt w/ COLORS IN RELEVANCE
        plt.subplot(4,12,j+1)
        plt.contourf(bini,bini,sie_rel_bin_avg,cmap='seismic',levels=20,vmin=-.75, vmax=.75)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        mean_relevance = np.nanmean(sie_rel_bin_avg)
        std_relevance = np.nanstd(sie_rel_bin_avg)
        text = fr"\miu = {mean_relevance:.2f} \pm {std_relevance:.2f}"
        plt.text(1.0, 1.0, text, fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

        if k == 0:
            plt.ylabel('SIAt')
        if k == 6:
            plt.colorbar(label='Relevance (SIE)')
        plt.title(tfr[j])

        plt.subplot(4,12,j+13)
        plt.contourf(bini,bini,siat_rel_bin_avg,cmap='seismic',levels=20,vmin=-.75, vmax=.75)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        mean_relevance = np.nanmean(siat_rel_bin_avg)
        std_relevance = np.nanstd(siat_rel_bin_avg)
        text = fr"\miu = {mean_relevance:.2f} \pm {std_relevance:.2f}"
        plt.text(1.0, 1.0, text, fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

        if k == 0:
            plt.ylabel('SIAt')
        if k == 6:
            plt.colorbar(label='Relevance (SIAt)')
            
        plt.subplot(4,12,j+25)
        plt.contourf(bini,bini,sie_rel_bin_avg-siat_rel_bin_avg,cmap='seismic',levels=20,vmin=-.75, vmax=.75)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        mean_relevance = np.nanmean((sie_rel_bin_avg-siat_rel_bin_avg))
        text = f"mean relevance: {mean_relevance:.2f}"
        plt.text(1.0, 1.0, text, fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

        if k == 0:
            plt.ylabel('SIAt')
        if k == 6:
            plt.colorbar(label='Relevance (SIAt)')
        
        plt.subplot(4,12,j+37)
        plt.contourf(bini,bini,sie_out_bin_avg,cmap='seismic',levels=20,vmin=-2, vmax=2)
        plt.plot([0,0],[-5,5],color='k')
        plt.plot([-5,5],[0,0],color='k')
        plt.xlabel('SIE')
        if k == 0:
            plt.ylabel('SIAt')
        if k == 6:
            plt.colorbar(label='SIE, SEP')
'''