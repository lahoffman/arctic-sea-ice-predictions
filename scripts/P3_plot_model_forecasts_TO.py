#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:13:58 2024

@author: hoffmanl
"""
  
#------------------------------------------------------
#------------------------------------------------------   
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
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

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
#------------------------------------------------------
# I. LOAD TRANSFER OPERATOR
#------------------------------------------------------
#------------------------------------------------------
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/M_transfer_operator_3sigma_sie2sie_mon2sep_normTR_TVT_cmip6_185001-201412_vX.nc'
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
#------------------------------------------------------
#------------------------------------------------------



#------------------------------------------------------
#------------------------------------------------------
# II. LOAD TEST DATA: NSIDC OBSERVATIONS
#------------------------------------------------------
#------------------------------------------------------

#[MMM, demeaned, detrended] residual from cmip6 historical (1979-2014) + residual cmip6 ssp585 (2015-2024)
load_path = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/D_siextentn_TO_test_data_obs_residual_cmip6_hist_ssp585_deMT_fromMMM_nonstd_197901202412_vX.nc'
dataset =  nc.Dataset(load_path,'r')
sie_original = np.array(dataset.variables['sie_obs'])
sie_obs_te = np.array(dataset.variables['residual_mean_weighted']) #[10,17,45]
fit_obs = np.array(dataset.variables['fit_mean_weighted']) #[10,17,45]
sie_mean = np.array(dataset.variables['sie_mean'])
sie_std = np.array(dataset.variables['sie_std'])


#normalize residual by mean and stdev of training data
outer = []
outeri = []
for i in range(17):
    inner = []
    inneri = []
    for j in range(10):
        te = sie_obs_te[j,i,:] 
        test = sigma_train[0,i,0,j]
        tem = miu_train[0,i,0,j]
        ted = np.divide((te-tem),test)
        
        tei = sie_obs_te[j,i,:]       
        inner.append(ted) #standardized
        inneri.append(tei) #non-standardized
    outer.append(inner)
    outeri.append(inneri)

#******************************************************************************************************************************************************
#ON/OFF for STANDARIZED OBS
sie_obs = np.array(outer)[:,:,1:] #standardized
#sie_obsi = np.array(outeri)[:,:,np.newaxis,:] #non-standardized
#******************************************************************************************************************************************************

'''
#rearrange shape of data
outer = []
for i in range(17):
    inner = []
    for j in range(47):
        te = sie_obsi[i,:,0,j]
        inner.append(te)
    outer.append(inner)
sie_obs = np.array(outer)   

#set residual to one unit standard deviation
outer = []
outerstd = []
outermean = []
for i in range(17):
    inner = []
    innerstd = []
    innermean = []
    for j in range(10):
        te = sie_obs_te[j,i,:] 
        test = np.nanstd(sie_obs_te[j,i,:]) 
        tem = np.nanmean(sie_obs_te[j,i,:])
        ted = np.divide((te-tem),test)       
        inner.append(ted)
        innerstd.append(test)
        innermean.append(tem)
    outer.append(inner)
    outerstd.append(innerstd)
    outermean.append(innermean)
sie_obs = np.array(outer)
residual_std = np.array(outerstd)
residual_mean = np.array(outermean)
'''

test_data = sie_obs[:,np.newaxis,:,:] #2002-2020
test_original = sie_original[:,np.newaxis,:,:]
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# III. LOAD TRAINING DATA: CMIP6 MODELS
#------------------------------------------------------
#------------------------------------------------------
loadpath_sie = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/siextentn_SImon_models_historical_rXi1p1f1_ARCTIC_TF_T_RMMM_NORM2_mon2sep_gn_185001-201412.nc'
dataset_sie =  nc.Dataset(loadpath_sie,'r')
years = dataset_sie.variables['unique_years']
sie_model = sie_training
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# BIN INDICES
#------------------------------------------------------
#------------------------------------------------------
#which bin does the initial condition fall in
#find bin indices
#------------------
def find_bin_index(value, bin_boundaries):
    # Iterate through bin boundaries
    for i in range(len(bin_boundaries) - 1):
        # Check if the value falls within the current bin boundaries
        if bin_boundaries[i] <= value < bin_boundaries[i + 1]:
            return i  # Return the index of the bin
    # If the value is outside all bins, return None or -1 (depending on preference)
    return np.nan

#assign time series to bins
#------------------
def assign_time_series_to_bins(time_series, bin_boundaries):
    bin_indices = []
    # Iterate through the time series
    for value in time_series:
        # Find the bin index for the current time step
        bin_index = find_bin_index(value, bin_boundaries)
        bin_indices.append(bin_index)
    return bin_indices
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
# PREDICTED POSSIBILITIES: BIN VALUES [MIDDLE]
#------------------------------------------------------
#------------------------------------------------------

predicted_poss = []
for k in range(17):
    bin_means = []
    transferbin = transfer_bins[:,k]
    for i in range(len(transferbin)-1):
        bin_mean = (transferbin[i]+transferbin[i+1])/2
        bin_means.append(bin_mean)
    predicted_possibilities = np.array(bin_means)
    predicted_possibilities[0,] = transferbin[1,]-(transferbin[2,]-transferbin[1,])/2
    predicted_possibilities[21,] = transferbin[21,]+(transferbin[2,]-transferbin[1,])/2
    predicted_poss.append(predicted_possibilities)    
xi = np.array(predicted_possibilities)
#------------------------------------------------------
#------------------------------------------------------


#------------------------------------------------------
#------------------------------------------------------
#PREDICTIONS
#make predictions based on the residuals calculated from the observations
#------------------------------------------------------
#------------------------------------------------------
stdevl = []
predictionl = []
binsl = []
probabilityl = []
for l in range(10):
    stdevi = []
    predictioni = []
    binsi= []
    probabilityi = []
    for i in range(17):
        
        stdevj = []
        binsj = []
        predictionj = []
        probabilityj = []
        for j in range(10):
            transferop = transfer_operator[i,j,l,:,:]
            transferbins = transfer_bins[:,i]
                      
            sie = np.reshape(test_data[i,:,j,:],[46,])
            
            sie_bins = assign_time_series_to_bins(sie,transferbins)
            nt = sie.shape[0]
            
            stdev = []
            prediction = []
            probability = []
            for k in range(nt):            
                bi = sie_bins[k]
                if ~np.isnan(bi):
                    prob_k = transferop[bi,:]
                else: 
                    prob_k = np.full([22,], np.nan)
                
                #prediction is expected value
                predictionk = np.sum(xi*prob_k) 
                stdevk = np.sqrt(np.sum(np.multiply(np.square(xi-predictionk),prob_k),axis=0))
                
                stdev.append(stdevk)
                prediction.append(predictionk)
                probability.append(prob_k)
            stdevj.append(stdev)
            predictionj.append(prediction)
            probabilityj.append(probability)
            binsj.append(sie_bins)
        stdevi.append(stdevj)
        predictioni.append(predictionj)
        probabilityi.append(probabilityj)
        binsi.append(binsj)
    stdevl.append(stdevi)
    predictionl.append(predictioni)
    probabilityl.append(probabilityi)
    binsl.append(binsi)
#------------------------------------------------------ 
#------------------------------------------------------ 



#------------------------------------------------------ 
#------------------------------------------------------ 
#~~~~~~~~~~~~~~~~~~~~~FIGURE~~~~~~~~~~~~~~~~~~~~~~~~~~~
# prediction statistics, standardized residual
#------------------------------------------------------
#------------------------------------------------------ 
years = []
for i in range(10):
    yearsi = np.arange(1979+i+1,2024+i+1)
    years.append(yearsi)
yearspred = np.array(years)

years = np.arange(1978,2025)

tf = [1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1]

yp = []
yi = []
for i in range(17):
    if tf[i] == 1:
        yearpred = yearspred
        yearinput = yearspred-1

    else:
        yearpred = yearspred
        yearinput = yearspred
        
    yp.append(yearpred)
    yi.append(yearinput)
yearpred = np.array(yp)[:,:,:]
yearinput = np.array(yi)[:,:,:]
years_fit = np.arange(1979,2025)


for j in range(12,13):
    ps = 13 #september, predicted time frame
    
    #---------------------------------
    #*********************************
    #TOGGLE TO DESIRED INTEREST
    pp = 11 #starting time frame
    year_of_interest = 2024
    #*********************************
    #---------------------------------
    
    #figure properties
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(160,96))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    kl = np.concatenate((np.arange(0,3)[:,np.newaxis],np.arange(3,6)[:,np.newaxis],np.arange(6,9)[:,np.newaxis]),axis=1)
    titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    title_letter = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    label_large = 110
    label_small = 90
    
    #prediction
    prediction_array = np.array(predictionl)
    error_array = np.array(stdevl)
    
    #time based on year of interest
    yearsinterest = yearpred[pp,0,:]
    if tf[pp] == 1:
        initialize = np.array(np.where(yearsinterest == year_of_interest-1))[0,0]
        #use this for climatology lines on first row
        timek = np.arange(yearsinterest[initialize]-1,yearsinterest[initialize]+7)
        time_input = yearsinterest[initialize]
    else:
        initialize = np.array(np.where(yearsinterest == year_of_interest))[0,0]
        #use this for climatology lines on first row
        timek = np.arange(yearsinterest[initialize]-2,yearsinterest[initialize]+6) 
        time_input = yearsinterest[initialize]
        
    
    
    #transfer bins for time frame
    bins = transfer_bins[:,pp]
    bin_width = bins[21]-bins[20]
    probabilities = np.array(probabilityl)
    total_probability= np.sum(probabilities,axis=4)
    pdfs = probabilities*100
    
    
    for k in range(3):
            
        #mov_mean index
        mov_mean = [0,1,4] #index of n+1 - year average
        mov_meani = mov_mean[k]
        mov_mean_label = mov_meani+1
        time_pred = yearpred[pp,mov_meani,initialize]
        time_pred_5 = np.arange(time_pred,time_pred+5)
    
        #data: sie_obs from input ; tt = starting year
        data = sie_obs[pp,mov_meani,initialize+1]
        
        #prediction for september from pp = starting time frame ; tt = starting year
        predk = prediction_array[:,pp,mov_meani,initialize+1]
        errork = error_array[:,pp,mov_meani,initialize+1]
        
        
        #expected value, i.e. the prediction from
        #pp = starting time frame ; tt = starting year
        bin_array = np.array(binsl)
        bink = np.array(bin_array[:,pp,mov_meani,initialize+1],dtype=int)
        expected = xi[bink]
        
        probabilityk = probabilities[mov_meani,pp,mov_meani,initialize+1,:]
        total_probability= np.sum(probabilityk,axis=0)
        pdfs = probabilityk*100
        
        #climatology statistics for september
        sie_train = np.reshape(sie_model[13,:,:,mov_meani],148*165,)
        #sie_train = sie_obs[13,mov_meani,:]
        clim_mean = np.nanmean(sie_train)
        clim_std = np.nanstd(sie_train)
        clim_mean_all = clim_mean
        clim_stdp1_all = clim_mean_all+clim_std
        clim_stdp2_all = clim_mean_all+ 2*clim_std
        clim_stdp3_all = clim_mean_all+ 3*clim_std
        clim_stdm1_all = clim_mean_all-clim_std
        clim_stdm2_all = clim_mean_all- 2*clim_std
        clim_stdm3_all = clim_mean_all- 3*clim_std
        axtick = np.array([clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all])
        axtick2 = np.array([clim_stdm3_all,clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all,clim_stdp3_all])
        
        #find bins for train data [i.e. climatological PDF]
        clim_bins = np.array(assign_time_series_to_bins(sie_train,transfer_bins[:,j]))
        nb = clim_bins.shape[0]

        #find number in each bin
        clim_pdf = []
        for h in range(22):
            ph = np.divide(np.sum(clim_bins == h),nb)
            clim_pdf.append(ph) 
        climatological_probabilities = np.array(clim_pdf)*100

        #categorize pdf: high, low, moderate, etc.
        pdf = pdfs
        binsk = bins[1:]
        high = np.round(np.sum(pdf[binsk > clim_mean]),decimals=1)
        high_moderate = np.round(np.sum(pdf[(binsk > clim_mean) & (binsk < clim_stdp1_all)]),decimals=1)
        high_intense = np.round(np.sum(pdf[(binsk > clim_stdp1_all) & (binsk< clim_stdp2_all)]),decimals=1)
        high_extreme = np.round(np.sum(pdf[binsk > clim_stdp2_all]),decimals=1)
        low = np.round(np.sum(pdf[binsk < clim_mean]),decimals=1)
        low_moderate = np.round(np.sum(pdf[(binsk < clim_mean) & (binsk > clim_stdm1_all)]),decimals=1)
        low_intense = np.round(np.sum(pdf[(binsk < clim_stdm1_all) & (binsk > clim_stdm2_all)]),decimals=1)
        low_extreme = np.round(np.sum(pdf[binsk < clim_stdm2_all]),decimals=1)
              
        #difference from climatology
        clim_diff = np.divide((pdf-climatological_probabilities),climatological_probabilities)
        clim_blue = clim_diff[clim_diff < 0]
        clim_red = clim_diff[clim_diff > 0]
        bin_red = binsk[clim_diff > 0]
        bin_blue = binsk[clim_diff < 0]
        
        #ylim for third plots
        yf = np.ceil(clim_red)
        yo = -np.ceil(-clim_blue)
        
        #figure titles and axes labels
        if k > 0:
            xt1 = np.array(timek-mov_meani,dtype=str)
            xt2 = np.array(timek,dtype=str)
            xt3 = np.array(timek+mov_meani,dtype=str)
            xt1s = [label[-2:] for label in xt1]
            xt2s = [label[-2:] for label in xt2]
            xt3s = [label[-2:] for label in xt3]
            
            #labels for subplots with time range
            xstre = [f'{cxt1}-{cxt2}' for cxt1, cxt2 in zip(xt1s,xt2s)]
            xstr = [f'{cxt1}-{cxt2}' for cxt1, cxt2 in zip(xt2,xt3)]
            
            #title for two- and five-year forecast [prediction time through prediction time + 2 or 5]
            xstrtit = xstr[2]
        else:
            #title for one year forecast [prediction time]
            xtit = timek[2]
        
        
        for l in range(3):
            ax = axes[kl[l,k]] 
            titlet = title_letter[kl[k,l]]
            
            if l ==0:
                ax.errorbar(time_input,data,0,color='blue',fmt='s',markersize=100,capsize=5,zorder=2)
                ax.errorbar(timek[2:-1],predk[0:5],errork[0:5],color='red',fmt='o',markersize=100,capsize=5,zorder=2)
                ax.plot(timek,clim_mean*np.ones([8,]),linewidth=1,color='black',zorder=1)
                ax.plot(timek,clim_std*np.ones([8,]),linewidth=1,color='black',zorder=1)
                ax.plot(timek,2*clim_std*np.ones([8,]),linewidth=1,color='black',zorder=1)
                ax.plot(timek,-clim_std*np.ones([8,]),linewidth=1,color='black',zorder=1)
                ax.plot(timek,-2*clim_std*np.ones([8,]),linewidth=1,color='black',zorder=1)
                ax.scatter(timek[2:-1],xi[bink[0:5]],s=100,marker='*',color='black',zorder=3)
                
                ax.tick_params(axis='both', labelsize=label_large)
                ax.set_xlim([timek[0],timek[-1]])                
                if k > 0:
                    ax.set_title(f'{titlet} {mov_mean_label}-year average', fontsize=label_large)
                    ax.set_xticklabels(xstre,rotation=45)
                else:
                    ax.set_title(f'Forecast trajectory for \n{titlet} {mov_mean_label}-year average', fontsize=label_large)
                if k == 1: 
                    ax.set_ylabel('SIE', fontsize=label_large)
                
            elif l == 1:
                ax.bar(bins[1:]-bin_width/2,climatological_probabilities,width=bin_width,alpha=0.7,edgecolor='black',color='grey',label='climatological')
                ax.bar(bins[1:]-bin_width/2,pdf,width=bin_width/1.5,alpha=0.7,edgecolor='black',color='red',label='forecast')
                ax.bar(data,30,width=0.01,color='blue',label='initial condition')
                ax.set_xlim(-0.85,0.85)
                ax.set_ylim(0,30)
                #ax.set_title(f'Probabilistic forecast (%) for {mov_mean_label}-year average', fontsize=20)
                ax.tick_params(axis='both', labelsize=label_large)
                ax.text(0.05,0.95,f'HIGH: {high:.1f} \nMOD: {high_moderate:.1f} \nINT:: {high_intense:.1f} \nEXT: {high_extreme:.1f} \nLOW: {low:.1f} \nMOD: {low_moderate:.1f} \nINT: {low_intense:.1f} \nEXT: {low_extreme:.1f}',fontsize=label_large,ha='left', va='top', transform=ax.transAxes)
                ax.set_xlim(axtick2[0],axtick2[-1])
                ax.set_xticks(axtick)
                ax.grid(True,axis='x')     
                if k > 0:
                    ax.set_title(f'{titlet} {mov_mean_label}-year average, {xstrtit}', fontsize=label_large)
                else:
                    ax.set_title(f'Probabilistic forecast (%) for \n{titlet}  {mov_mean_label}-year average, {xtit}', fontsize=label_large)
                if k == 2: 
                    ax.set_xlabel('SIE', fontsize=label_large)
                handles, labels = ax2.get_legend_handles_labels()
                ax2.legend(handles, labels, loc='upper right', ncol=1,fontsize=label_large)

                    
            else:
                ax.bar(bin_red-bin_width/2,clim_red*100,color='red',width=bin_width,alpha=0.7,edgecolor='black',zorder=2)
                ax.bar(bin_blue-bin_width/2,clim_blue*100,color='blue',width=bin_width,alpha=0.7,edgecolor='black',zorder=2)
                ax.plot(axtick2,np.zeros(7),linewidth=2,color='black')
                #ax.set_title('Probability of abnormal events (%)', fontsize=20)
                ax.set_xticks(axtick)
                ax.grid(True,axis='x') 
                ax.set_ylim(yo[k]*100,yf[k]*100)
                ax.tick_params(axis='both', labelsize=label_large)
                ax.set_xlim(axtick2[0],axtick2[-1])

                for g in range(6):
                    bg_color = ['cornflowerblue','lightskyblue','azure','cornsilk','moccasin','sandybrown']
                    x_start = axtick2[g]
                    x_end = axtick2[g+1]
                    y_end = yf[k]-yo[k]
                    bg_patch = Rectangle((x_start, ax.get_ylim()[0]), x_end - x_start, y_end*100, color=bg_color[g], alpha=0.5,zorder=1)
                    ax.add_patch(bg_patch)
                
                if k > 0:
                    ax.set_title(f'{titlet} {xstrtit}', fontsize=label_large)
                else:
                    ax.set_title(f'Probability of abnormal events (%) \n{titlet}  {xtit}', fontsize=label_large)
                
                if k == 2: 
                    ax.set_xlabel('SIE', fontsize=label_large)
               

    #ax2.set_ylabel('SIC anomaly', fontsize=20)
    fig.subplots_adjust(right=0.85,hspace=0.3,wspace=0.2)
    #fig.suptitle(f'PREDICTION OF SEPTEMBER SIE, INITIALIZED {titles[pp]} {time_input}', fontsize=label_large)
    
    plt.show
#------------------------------------------------------ 
#------------------------------------------------------ 


'''
#------------------------------------------------------ 
#------------------------------------------------------ 
#~~~~~~~~~~~~~~~~~~~~~FIGURE~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~PROBABILITIES SEMICIRCLE PIE CHART~~~~~~~~~~~~~~~
#------------------------------------------------------ 
#------------------------------------------------------ 
ps = 13 #september, predicted time frame

#---------------------------------
#*********************************
#TOGGLE TO DESIRED INTEREST
pp = 11 #starting time frame
year_of_interest = 2024
#*********************************
#---------------------------------



label_large = 110
label_small = 90

#prediction
prediction_array = np.array(predictionl)
error_array = np.array(stdevl)

#time based on year of interest
if tf[pp] == 1:
    initialize = np.array(np.where(yearsinput == year_of_interest-1))[0,0]
    #use this for climatology lines on first row
    timek = np.arange(yearsinput[initialize]-1,yearsinput[initialize]+7)
    time_input = yearsinput[initialize]
else:
    initialize = np.array(np.where(yearsinput == year_of_interest))[0,0]
    #use this for climatology lines on first row
    timek = np.arange(yearsinput[initialize]-2,yearsinput[initialize]+6) 
    time_input = yearsinput[initialize]
    


#transfer bins for time frame
bins = transfer_bins[:,pp]
bin_width = bins[21]-bins[20]
probabilities = np.array(probabilityl)
total_probability= np.sum(probabilities,axis=4)
pdfs = probabilities*100



k = 0  
#mov_mean index
mov_mean = [4] #index of n+1 - year average
mov_meani = mov_mean[k]
mov_mean_label = mov_meani+1
time_pred = yearpred[pp,mov_meani,initialize]
time_pred_5 = np.arange(time_pred,time_pred+5)

#data: sie_obs from input ; tt = starting year
data = sie_obs[pp,mov_meani,initialize]

#prediction for september from pp = starting time frame ; tt = starting year
predk = prediction_array[:,pp,mov_meani,initialize]
errork = error_array[:,pp,mov_meani,initialize]


#expected value, i.e. the prediction from
#pp = starting time frame ; tt = starting year
bin_array = np.array(binsl)
bink = np.array(bin_array[:,pp,mov_meani,initialize],dtype=int)
expected = xi[bink]

probabilityk = probabilities[mov_meani,pp,mov_meani,initialize,:]
total_probability= np.sum(probabilityk,axis=0)
pdfs = probabilityk*100

#climatology statistics for september
sie_train = np.reshape(sie_model[:,13,:,mov_meani],211*165,);
clim_mean = np.nanmean(sie_train)
clim_std = np.nanstd(sie_train)
clim_mean_all = clim_mean
clim_stdp1_all = clim_mean_all+clim_std
clim_stdp2_all = clim_mean_all+ 2*clim_std
clim_stdp3_all = clim_mean_all+ 3*clim_std
clim_stdm1_all = clim_mean_all-clim_std
clim_stdm2_all = clim_mean_all- 2*clim_std
clim_stdm3_all = clim_mean_all- 3*clim_std
axtick = np.array([clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all])
axtick2 = np.array([clim_stdm3_all,clim_stdm2_all,clim_stdm1_all,clim_mean_all,clim_stdp1_all,clim_stdp2_all,clim_stdp3_all])

#find bins for train data [i.e. climatological PDF]
clim_bins = np.array(assign_time_series_to_bins(sie_train,transfer_bins[:,j]))
nb = clim_bins.shape[0]

#find number in each bin
clim_pdf = []
for h in range(22):
    ph = np.divide(np.sum(clim_bins == h),nb)
    clim_pdf.append(ph) 
climatological_probabilities = np.array(clim_pdf)*100

#categorize pdf: high, low, moderate, etc.
pdf = pdfs
binsk = bins[1:]
high = np.round(np.sum(pdf[binsk > clim_mean]),decimals=1)
high_moderate = np.round(np.sum(pdf[(binsk > clim_mean) & (binsk < clim_stdp1_all)]),decimals=1)
high_intense = np.round(np.sum(pdf[(binsk > clim_stdp1_all) & (binsk< clim_stdp2_all)]),decimals=1)
high_extreme = np.round(np.sum(pdf[binsk > clim_stdp2_all]),decimals=1)
low = np.round(np.sum(pdf[binsk < clim_mean]),decimals=1)
low_moderate = np.round(np.sum(pdf[(binsk < clim_mean) & (binsk > clim_stdm1_all)]),decimals=1)
low_intense = np.round(np.sum(pdf[(binsk < clim_stdm1_all) & (binsk > clim_stdm2_all)]),decimals=1)
low_extreme = np.round(np.sum(pdf[binsk < clim_stdm2_all]),decimals=1)

sizes = [high_extreme/2,high_intense/2,high_moderate/2,low_moderate/2,low_intense/2,low_extreme/2,50]
colors = ['sandybrown','moccasin','cornsilk','azure','lightskyblue','cornflowerblue','white']

import matplotlib.patches as patches
from matplotlib.patches import Wedge

fig = plt.figure(figsize=(20,20))
ax = plt.gca()

labels = ['EXTREME \nLOW','INTENSE \nLOW','MODERATE \nLOW','MODERATE \nHIGH','INTENSE \nHIGH','EXTREME \nHIGH']
sizes2 = [high_extreme/100,high_intense/100,high_moderate/100,low_moderate/100,low_intense/100,low_extreme/100,50/100]
sizes3 = [low_extreme/100,low_intense/100,low_moderate/100,high_moderate/100,high_intense/100,high_extreme/100]
xx = [-0.65,-0.6,-0.375,0.01,0.25,0.35]
yy = [0.07,0.35,0.825,0.65,0.35,0.07]
xx2 = [-1.1,-0.85,-0.35,0.35,0.85,1.1]
yy2 = [0.25,0.75,1.05,1.05,0.75,0.25]
rot = [78,50,15,-15,-50,-78]
cc = ['white','black','black','black','black','black']
sizes = [50/6,50/6,50/6,50/6,50/6,50/6,50]
colors = ['firebrick','rosybrown','lightsalmon','lightblue','cornflowerblue','darkblue','white']
colors2 = ['grey','grey','grey','grey','grey','grey','grey']
wedges = plt.pie(sizes,colors=colors2)
for i in range(7):
    wedge = wedges[0]
    theta1 = wedge[i].theta1
    theta2 = wedge[i].theta2
    center,r = wedge[i].center, wedge[i].r
    size2 = sizes2[i]
    
    theta_mid = (theta1 + theta2) / 2
    theta_third = (theta_mid+theta1)/2
    theta_twothird = (theta_mid+theta2)/2
    theta_patch = theta_mid - theta1
    colori = colors[i]
    
    wedge_patch1 = patches.Wedge(center,r/3,theta1-10, theta2+10, width=r/2, facecolor='white', edgecolor='none')
    wedge_patch2 = patches.Wedge(center,r, theta1, theta2, width=r-size2-r/3, facecolor=colori, edgecolor='none')
    ax.add_patch(wedge_patch2)
    ax.add_patch(wedge_patch1)
    if i == 1:
        ax.text(xx[i-1], yy[i-1],f'{round(sizes3[i-1],2)*100:.1f} %', fontsize=40,color=cc[i-1])
        ax.text(xx2[i-1], yy2[i-1],labels[i-1], fontsize=35,color='black',rotation=rot[i-1], ha='center', va='center')
    if i > 1:
        ax.text(xx[i-1], yy[i-1],f'{round(sizes3[i-1],3)*100:.1f} %', fontsize=40,color=cc[i-1])
        ax.text(xx2[i-1], yy2[i-1],labels[i-1], fontsize=35,color='black',rotation=rot[i-1], ha='center', va='center')


#------------------------------------------------------ 
#------------------------------------------------------ 
'''