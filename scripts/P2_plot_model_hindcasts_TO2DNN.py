#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:00:50 2024

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
#PLOT: SIC prediction in YEAR for different hindcast times
# 1-YEAR MOVING MEAN
#------------------------------------------------------
#------------------------------------------------------ 
# Create a figure and gridspec layout
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(48, 30))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
kl = np.concatenate((np.arange(0,2)[:,np.newaxis],np.arange(2,4)[:,np.newaxis],np.arange(4,6)[:,np.newaxis]),axis=1)



for l in range(2):
    if l == 0:
        # LOAD TO 
        #------------------------------------------------------
        #------------------------------------------------------
        loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_TO_julyinit_2012_2020.nc'
        #loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_TO_julyinit_2007_2019.nc'
        dataset =  nc.Dataset(loadpath,'r')
        data = np.array(dataset.variables['datato'])
        predk = np.array(dataset.variables['predto'])
        errk = np.array(dataset.variables['errto'])
        clim_mean = np.array(dataset.variables['climmeanto'])
        clim_stdp1_all = np.array(dataset.variables['climstdp1to'])
        clim_stdp2_all = np.array(dataset.variables['climstdp2to'])
        clim_stdm1_all = np.array(dataset.variables['climstdm1to'])
        clim_stdm2_all = np.array(dataset.variables['climstdm2to'])
        unique_years = np.arange(1978,2033)
        co = 'r'
    elif l == 1:
        # LOAD 2DNN
        #------------------------------------------------------
        #------------------------------------------------------
        loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_2DNN_julyinit_2012_2020.nc'
        #loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_extreme_2DNN_julyinit_2007_2019.nc'
        dataset =  nc.Dataset(loadpath,'r')
        data = np.array(dataset.variables['datann'])
        predk = np.array(dataset.variables['prednn'])
        errk = np.array(dataset.variables['errnn'])
        clim_mean = np.array(dataset.variables['climmeannn'])
        clim_stdp1_all = np.array(dataset.variables['climstdp1nn'])
        clim_stdp2_all = np.array(dataset.variables['climstdp2nn'])
        clim_stdm1_all = np.array(dataset.variables['climstdm1nn'])
        clim_stdm2_all = np.array(dataset.variables['climstdm2nn'])
        co = 'seagreen'
    for j in range(2):
        ps = 13 #september, predicted time frame
        yearsinterest = [2012,2020]
        #yearsinterest = [2007,2019]
        #---------------------------------
        #*********************************
        #TOGGLE TO DESIRED INTEREST
        pp = 0 #starting time frame
        year_of_interest = yearsinterest[j]
        #*********************************
        #---------------------------------
        
        tf = [1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1]
        
        #index of prediction year
        timeX = np.array(np.where(unique_years==year_of_interest))[0,0]

        #index of prediction year for time before now; index of previous year for time now (i.e. unknown 'data')
        if(timeX<46):
            timeX2 = timeX 
        else:
            timeX2 = 45
        
        #lagtime based on starting timeframe
        if tf[pp] == 0:
            lagtime = np.arange(0,10)
        else:
            lagtime = np.arange(1,11)
            
        #index of prediction year
        predyears = np.arange(1980,2024)
        fityears = np.arange(1979,2024)
        linfityears = np.arange(1979,2034)
        timefitX = np.array(np.where(fityears==year_of_interest))[0,0]
        timelinfitX = np.array(np.where(linfityears==year_of_interest))[0,0]

        #index of prediction year for time before now; index of previous year for time now (i.e. unknown 'data')
        if(timefitX<46):
            timefit = timefitX 
            timelinfit = timelinfitX
        else:
            timefit = 45
            timelinfit = 45
        
        #lagtime based on starting timeframe
        if tf[pp] == 0:
            lagtime = np.arange(0,10)
        else:
            lagtime = np.arange(1,11)
     
            
        initialize = timefitX-lagtime
        
        titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
        title_letters = ['(a)','(d)','(b)','(e)','(c)','(f)']
        labelk  = ['1-year mean','2-year mean','5-year mean']

        text_large = 50
        text_small = 30

        for k in range(3):

            for i in range(10):
                
                #title
                titles = f'{title_letters[kl[j,k]]} {yearsinterest[j]}, {labelk[k]}' 
    
                mov_mean = [0,1,4] #index of n+1 - year average
                mov_meani = mov_mean[k]
                mov_mean_label = mov_meani+1
            
                ax = axes[kl[j,k]] 
                
                #if statement so legend only on first set
                if i == 0:
                    if l == 0:
                        line = ax.plot(np.arange(-1,12),data[j,k,i]*np.ones(13),linewidth=5,color='blue',label='obs')
                        line2 = ax.plot(np.arange(-1,12),clim_mean[j,k,i]*np.ones(13),linewidth=5,color='gray',label='linear trend')
                        line3 = ax.plot(np.arange(-1,12),clim_mean[j,k,i]*np.ones(13),linewidth=3,color='gray',label='linear stdev')
                        errorbars = ax.errorbar(lagtime[i],predk[j,k,i],errk[j,k,i],color=co,fmt='o',markersize=30,capsize=10,label='pred, TO')
                        ax.plot(np.arange(-1,12),clim_mean[j,k,i]*np.ones([13,]),linewidth=1,color='lightgray',zorder=1,label='_nolegend_')
                        #ax.text(0.55,0.01,f'MEAN: {np.round(predk,decimals=2)} \nSTDEV: {np.round(errork,decimals=2)} \nMEDIAN: {np.round(medk,decimals=2)}  \nLOW ERR BOUND: {np.round(cilk,decimals=2)}  \nHIGH ERR BOUND: {np.round(cihk,decimals=2)} \nLINEAR: {np.round(clim_mean,decimals=2)} \nSEP 2023: {4.37} ' ,fontsize=30,ha='left', va='bottom', transform=ax.transAxes)
                    else:
                        errorbars = ax.errorbar(lagtime[i],predk[j,k,i],errk[j,k,i],color=co,fmt='o',markersize=30,capsize=10,label='pred, NN')
                   
                else:   
                    line = ax.plot(np.arange(-1,12),data[j,k,i]*np.ones(13),linewidth=5,color='blue',label='_nolegend_')
                    line2 = ax.plot(np.arange(-1,12),clim_mean[j,k,i]*np.ones(13),linewidth=5,color='gray',label='_nolegend_')
                    errorbars = ax.errorbar(lagtime[i],predk[j,k,i],errk[j,k,i],color=co,fmt='o',markersize=30,capsize=10,label='_nolegend_')
                    ax.plot(np.arange(-1,12),clim_mean[j,k,i]*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
                    
                ax.plot(np.arange(-1,12),clim_stdp1_all[j,k,i]*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
                ax.plot(np.arange(-1,12),clim_stdp2_all[j,k,i]*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
                ax.plot(np.arange(-1,12),clim_stdm1_all[j,k,i]*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
                ax.plot(np.arange(-1,12),clim_stdm2_all[j,k,i]*np.ones([13,]),linewidth=3,color='lightgray',zorder=1,label='_nolegend_')
                #ax.set_title(f'Hindcast for {mov_mean_label}-year average', fontsize=20)
                ax.grid(axis='x')
                ax.tick_params(axis='both', labelsize=text_small)
                ax.set_xlim([0,11])
                ax.set_xticks(np.arange(0,11))
                ax.set_title(titles,fontsize=text_large)
                #ax.set_ylim([4,7])
             
        ax3.set_ylabel('sea ice extent [$\mathregular{10^6   km^2}$]', fontsize=text_large)
        fig.text(0.4, 0.08,'hindcast lag, N [years]', fontsize=text_large)
        #ax1.legend(fontsize=30) #,loc='lower left')
        handles, labels = ax1.get_legend_handles_labels()
        fig.subplots_adjust(right=0.85)  # Adjust the right margin to make room for the colorbar
        if l == 1:
            fig.legend(handles, labels, loc='lower center', ncol=5,fontsize=text_large)
        #fig.suptitle(f'PREDICTION OF SEPTEMBER {year_of_interest} SIE, INITIALIZED {titles[pp]} {year_of_interest}-N YEARS', fontsize=30)
    #------------------------------------------------------
    #------------------------------------------------------ 
   


#------------------------------------------------------
#------------------------------------------------------ 
#PLOT: HINDCAST TIME SERIES FOR RILES
# prediction must be initialized before 2014
#***NOTE: first point on this plot is the same as the first on the next plot
#------------------------------------------------------
#------------------------------------------------------ 

#coefficient of determination
#------------------------------
def cod(y_true,y_pred):
    return (1 - np.divide(np.nanmean(np.square(y_pred - y_true)),np.nanmean(np.square(y_true))))
   
#RMSE
#------------------------------
def rmse(y_true_obs,y_pred_obs):
    nmt = y_true_obs.shape[0]
    return np.sqrt(np.divide(np.nansum(np.square(y_pred_obs-y_true_obs)),nmt))

#ACC
#------------------------------
def acc(y_true,y_pred):
    return (np.nansum((y_true-np.nanmean(y_true))*(y_pred-np.nanmean(y_pred))))/((np.sqrt(np.nansum(np.square(y_true-np.nanmean(y_true)))))*(np.sqrt(np.nansum(np.square(y_pred-np.nanmean(y_pred))))))
       

from matplotlib.cm import ScalarMappable

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(60,40))
label_large = 80
label_small = 50
label_mid = 60


for l in range(2):
    if l == 0:
        # LOAD TO 
        #------------------------------------------------------
        #------------------------------------------------------
        loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_TO_JUL2000.nc'
        #loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_TO_JUL2012.nc'
        dataset =  nc.Dataset(loadpath,'r')
        time = np.array(dataset.variables['timeto'])
        timeextend = np.array(dataset.variables['timeextendto'])
        timeinput = np.array(dataset.variables['timeinputto'])
        timedurpred = np.array(dataset.variables['timedurpredto'])
        data_residual = np.array(dataset.variables['data_residualto'])
        data_residual_predtime = np.array(dataset.variables['data_residual_predtimeto'])
        pred_residual = np.array(dataset.variables['prediction_residualto'])
        error_residual = np.array(dataset.variables['error_residualto'])
        data_residual_init = np.array(dataset.variables['data_residual_initto'])
        fitko = np.array(dataset.variables['fitto'])
        linfitk = np.array(dataset.variables['linfitto'])
        linfitkp1 = np.array(dataset.variables['linfitp1to'])
        linfitkp2 = np.array(dataset.variables['linfitp2to'])
        linfitkm1 = np.array(dataset.variables['linfitm1to'])
        linfitkm2 = np.array(dataset.variables['linfitm2to'])
        predk = np.array(dataset.variables['predkto'])
        errork = np.array(dataset.variables['errorkto'])
        data = np.array(dataset.variables['datakto'])
        co = 'red'
    elif l == 1:
        # LOAD 2DNN
        #------------------------------------------------------
        #------------------------------------------------------
        loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_2DNN_JUL2000.nc'
        #loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_2DNN_JUL2012.nc'
        dataset =  nc.Dataset(loadpath,'r')
        time = np.array(dataset.variables['timenn'])
        timeextend = np.array(dataset.variables['timeextendnn'])
        timeinput = np.array(dataset.variables['timeinputnn'])
        timedurpred = np.array(dataset.variables['timedurprednn'])
        data_residual = np.array(dataset.variables['data_residualnn'])
        data_residual_predtime = np.array(dataset.variables['data_residual_predtimenn'])
        pred_residual = np.array(dataset.variables['prediction_residualnn'])
        error_residual = np.array(dataset.variables['error_residualnn'])
        data_residual_init = np.array(dataset.variables['data_residual_initnn'])
        fitko = np.array(dataset.variables['fitnn'])
        linfitk = np.array(dataset.variables['linfitnn'])
        linfitkp1 = np.array(dataset.variables['linfitp1nn'])
        linfitkp2 = np.array(dataset.variables['linfitp2nn'])
        linfitkm1 = np.array(dataset.variables['linfitm1nn'])
        linfitkm2 = np.array(dataset.variables['linfitm2nn'])
        predk = np.array(dataset.variables['predknn'])
        errork = np.array(dataset.variables['errorknn'])
        data = np.array(dataset.variables['dataknn'])
        co = 'seagreen'
    
    # Create a figure and gridspec layout
    for j in range(2):   
        ps = 13 #september, predicted time frame
        
        #---------------------------------
        #*********************************
        #TOGGLE TO DESIRED INTEREST
        pp = 11 #starting time frame
        year_of_interest = 2000
        #*********************************
        #---------------------------------
     
        textx = [0.15,0.15,0.15]
        texty = [0.7,0.44,0.19]
        textyd = [0.675,0.415,0.165]
        textydd = [0.65,0.39,0.14]
        
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        kl = np.concatenate((np.arange(0,2)[:,np.newaxis],np.arange(2,4)[:,np.newaxis],np.arange(4,6)[:,np.newaxis]),axis=1)
        titles = ['YEARLY MEAN','JFM','AMJ','JAS','OND','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
        title_letter = ['(a)','(d)','(b)','(e)','(c)','(f)']
    
        
        for k in range(3):
            for i in range(10):
            
                ti = np.where(time[k,i,:] == year_of_interest)[0][0]
                mov_mean = [0,1,4] #index of n+1 - year average
                mov_meani = mov_mean[k]
                mov_mean_label = mov_meani+1
                ii = 0
           
                #figure properties
                tlet = title_letter[kl[j,k]]
                titles = f' {tlet} Hindcast for {mov_mean_label}-year average'
                
                if j == 0:
                    #figure titles and axes labels        
                    ax = axes[kl[j,k]] 
    
                    #plot forced + residual, obs
                    ax.plot(time[k,i,:],data_residual[k,i,:],linewidth=5,color='black',label='_nolegend_')               
                   
                    #plot forced + residual, obs for prediction time
                    ax.plot(time[k,i,ti:ti+10],data_residual[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')         
                    
                    #plot linear fit
                    lte = np.shape(timeextend)[2]
                    ax.plot(timeextend[k,i,:],np.zeros([lte,1]),linewidth=5,color='grey',label='_nolegend_')
                   
                    #plot prediction
                    ax.errorbar(timedurpred[k,i],pred_residual[k,i],error_residual[k,i],color=co,fmt='o',markersize=30,capsize=10,label='_nolegend_')
                    ax.set_xlim([1979,2023])
                       
                    #plot input sie
                    ax.scatter(time[k,i,ti],data_residual_init[k,i],marker='s',s=1000,color='blue',label='_nolegend_')
                   
                    #plot linear fit   
                    ax.plot(timeextend[k,i,:],np.ones([lte,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.plot(timeextend[k,i,:],-1*np.ones([lte,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.plot(timeextend[k,i,:],2*np.ones([lte,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.plot(timeextend[k,i,:],-2*np.ones([lte,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.plot(timeextend[k,i,:],-3*np.ones([lte,1]),linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.set_xlim([1979,2023])
                   
                    if i == 0:
                        ax.set_xlim([1979,2023])
                    
                    nti = 10
                    accres = np.round(acc(data_residual[k,1,ti:ti+nti],pred_residual[k,:nti]),2)
                    rmseres = np.round(rmse(data_residual[k,1,ti:ti+nti],pred_residual[k,:nti]),2)
                    codres = np.round(cod(data_residual[k,1,ti:ti+nti],pred_residual[k,:nti]),2)
                    
                    nti = 7
                    accres6 = np.round(acc(data_residual[k,1,ti:ti+nti],pred_residual[k,:nti]),2)
                    rmseres6 = np.round(rmse(data_residual[k,1,ti:ti+nti],pred_residual[k,:nti]),2)
                    codres6 = np.round(cod(data_residual[k,1,ti:ti+nti],pred_residual[k,:nti]),2)
                    
        
                    accformatted = f'{accres:0.2f}'
                    rmseformatted = f'{rmseres:0.2f}'
                    codformatted = f'{codres:0.2f}'
                    accformatted6 = f'{accres6:0.2f}'
                    rmseformatted6 = f'{rmseres6:0.2f}'
                    codformatted6 = f'{codres6:0.2f}'
         
                    fig.text(textx[k], texty[k],'ACC  (7-year)   $\mathregular{R^2}$  (7-year)', fontsize=label_mid)
                    
                    if l == 0:
                        col = 'r'
                        fig.text(textx[k], textyd[k],f'{accformatted}   ({accformatted6})   {codformatted}    ({codformatted6})', fontsize=label_mid,color=col)
                    elif l == 1:
                        col = 'seagreen'
                        fig.text(textx[k], textydd[k],f'{accformatted}   ({accformatted6})   {codformatted}    ({codformatted6})', fontsize=label_mid,color=col)
                        
                    ax.set_title(titles, fontsize=label_large)
                    ax.grid(axis='x')
                    ax.tick_params(axis='both', labelsize=label_small)
                    ax.set_xlim([1979,2023])
                    #ax1.legend(fontsize=label_large)
                    
                elif j == 1:
                   
                    #figure titles and axes labels        
                    ax = axes[kl[j,k]] 
                    
                    #if i for legend on/off
                    if i == 0:
                        if l == 0:
                            #plot forced (i.e. fit)
                            ax.plot(time[k,i,:-1],fitko[k,i,:],linewidth=5,color='orchid',label='forced')
                    
                            #plot forced + residual, obs
                            ax.plot(time[k,i,:],data[k,i,:],linewidth=5,color='black',label='obs')          
                            
                            #plot forced + residual, obs for prediction time
                            ax.plot(time[k,i,ti:ti+10],data[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')               
                            
                            #plot linear fit
                            ax.plot(timeextend[k,i,:],linfitk[k,i,:],linewidth=5,color='grey',label='linear fit, obs')
                            
                            #plot prediction
                            ax.errorbar(timedurpred[k,i],predk[k,i],errork[k,i],color=co,fmt='o',markersize=25,capsize=10,label='pred, TO')
                        elif l == 1:
                            ax.errorbar(timedurpred[k,i],predk[k,i],errork[k,i],color=co,fmt='o',markersize=25,capsize=10,label='pred, NN')
                              
                        ax.set_ylim([2,9])
                    
                    else:
                        #plot fit
                        #ax.plot(timeextend[:-1],fitk[:-1],linewidth=5,color='orchid',label='_nolegend_')
                        
                        #plot forced + residual, obs
                        ax.plot(time[k,i,:],data[k,i,:],linewidth=5,color='black',label='_nolegend_')               
                       
                        #plot forced + residual, obs for prediction time
                        ax.plot(time[k,i,ti:ti+10],data[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')         
                        
                        #plot linear fit
                        #ax.plot(timeextend,linfitk,linewidth=5,color='grey',label='_nolegend_')
                       
                        #plot prediction
                        ax.errorbar(timedurpred[k,i],predk[k,i],errork[k,i],color=co,fmt='o',markersize=25,capsize=10,label='_nolegend_')
                        ax.set_xlim([1979,2023])
                        ax.set_ylim([2,9])
                       
                    
                    #plot linear fit   
                    ax.plot(timeextend[k,i,:],linfitkp1[k,i,:],linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.plot(timeextend[k,i,:],linfitkp2[k,i,:],linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.plot(timeextend[k,i,:],linfitkm1[k,i,:],linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.plot(timeextend[k,i,:],linfitkm2[k,i,:],linewidth=1,color='grey',linestyle='--',label='_nolegend_')
                    ax.set_xlim([1979,2023])
                    
                   
                    if i == 0:
                        ax.set_xlim([1979,2023])
                        
                    ax.set_title(titles, fontsize=label_large)
                    ax.grid(axis='x')
                    ax.tick_params(axis='both', labelsize=label_small)
                    ax.set_xlim([1979,2023])
   
                ax3.set_xlim([1979,2023])
                ax3.set_ylabel('sea ice extent, residual', fontsize=label_large)
                ax4.set_ylabel('sea ice extent [$\mathregular{10^6 km^2}$]', fontsize=label_large)
                #ax2.legend(fontsize=label_large)
                fig.subplots_adjust(top=0.85,right=0.9,hspace=0.22,wspace=0.1)  # Adjust the right margin to make room for the colorbar
                handles, labels = ax2.get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=5,fontsize=label_large)
#------------------------------------------------------
#------------------------------------------------------ 





#------------------------------------------------------
#------------------------------------------------------
#PLOT: HINDCAST TIME SERIES, JJA INITIALIZED
#------------------------------------------------------
#------------------------------------------------------ 

# LOAD 2DNN
#------------------------------------------------------
#------------------------------------------------------
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_historicalJJAS_2DNN.nc'
dataset =  nc.Dataset(loadpath,'r')
accnn = np.array(dataset.variables['acci'])
accdetrendnn = np.array(dataset.variables['accdetrended'])
rmsenn = np.array(dataset.variables['rmsei'])
rmsedetrendnn = np.array(dataset.variables['rmsedetrended'])
timepnn = np.array(dataset.variables['timep'])
timemnn = np.array(dataset.variables['timem'])
sie_monnn = np.array(dataset.variables['siemsep'])
sie_prednn = np.array(dataset.variables['siepred'])


# LOAD TO 
#------------------------------------------------------
#------------------------------------------------------
loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_historicalJJAS_TO.nc'
dataset =  nc.Dataset(loadpath,'r')
accto = np.array(dataset.variables['acci'])
accdetrendto = np.array(dataset.variables['accdetrended'])
rmseto = np.array(dataset.variables['rmsei'])
rmsedetrendto = np.array(dataset.variables['rmsedetrended'])
timepto = np.array(dataset.variables['timep'])
timemto = np.array(dataset.variables['timem'])
sie_monto = np.array(dataset.variables['siemsep'])
sie_predto = np.array(dataset.variables['siepred'])
accps = np.array(dataset.variables['accpsi'])
accdetrendps = np.array(dataset.variables['accpsdetrended'])
rmseps = np.array(dataset.variables['rmsepsi'])
rmsedetrendps = np.array(dataset.variables['rmsepsdetrended'])

# PLOT
#------------------------------------------------------
#------------------------------------------------------ 

fig = plt.figure(figsize=(80,60))
label_large = 100
label_small = 80
textx = [0.15,0.58,0.15,0.58]
texty = [0.6,0.6,0.21,0.21]
textyd = [0.58,0.58,0.19,0.19]
textydd = [0.56,0.56,0.17,0.17]
textyps = [0.54,0.54,0.15,0.15]
textxps = [0.15,0.58,0.15,0.58]

mon = np.arange(9,13)
tit = ['(a) June 1st Initialized','(b) July 1st Initialized','(c) August 1st Initialized','(d) September 1st Initialized']


for i in range(2):  
    if i == 0:
        acc = accto[:,0]
        rmse = rmseto[:,0]
        accdetrend = accdetrendto[:,0]
        rmsedetrend = rmsedetrendto[:,0]
        accps = accps[:,0]
        rmseps = rmseps[:,0]
        accdetrendps = accdetrendps[:,0]
        rmsedetrendps = rmsedetrendps[:,0]
        sie_mon = sie_monto
        sie_pred = sie_predto
        timem = timemto
        timepi = timepto[:,:]
        col = 'r'
    elif i == 1:
        acc = accnn
        rmse = rmsenn
        accdetrend = accdetrendnn
        rmsedetrend = rmsedetrendnn
        sie_mon = sie_monto
        sie_pred = sie_prednn
        timem = timemnn
        timepi = timepnn
        col = 'seagreen'
    
    for j in range(4):


        plt.subplot(2,2,j+1)
        pmon=mon[j]
        timep = timepi[j,:]
        accj = acc[j]
        rmsej = rmse[j]
        accdetrendj = accdetrend[j]
        rmsedetrendj = rmsedetrend[j]
        accpsj = accps[j]
        rmsepsj = rmseps[j]
        accpsdetrendj = accdetrendps[j]
        rmsepsdetrendj = rmsedetrendps[j]
        
        
        if i == 0:
            plt.plot(timem,sie_mon,color='k', linewidth=8.0, marker='o', markersize=5,label='OBS')
            plt.plot(timepi[j,:],sie_pred[j,:],color= col, linewidth=6.0, marker='o', markersize=5,label='PRED, TO')
        elif i ==1:
            plt.plot(timem[1:],sie_mon,color='k', linewidth=8.0, marker='o', markersize=5,label='')
            plt.plot(timep,sie_pred[j,:],color= col, linewidth=6.0, marker='o', markersize=5,label='PRED,2DNN')
        plt.title(tit[j],fontsize=label_large)
        plt.xticks(np.arange(1980,2026,10),fontsize=label_small)
        plt.yticks(np.arange(2.5,8,.5),fontsize=label_small)
        plt.ylabel('Sea Ice Extent (M km$^2$)',fontsize=label_large)
        plt.grid(which='major', linestyle=':', linewidth='0.5', color='black')
        #plt.legend(fontsize=label_small,loc='upper right')
        accformatted = f'{accj:0.2f}'
        rmseformatted = f'{rmsej:0.2f}'
        accdetrendedformatted = f'{accdetrendj:0.2f}'
        rmsedetrendedformatted = f'{rmsedetrendj:0.2f}'
        accpsformatted = f'{accpsj:0.2f}'
        rmsepsformatted = f'{rmsepsj:0.2f}'
        accpsdetrendedformatted = f'{accpsdetrendj:0.2f}'
        rmsepsdetrendedformatted = f'{rmsepsdetrendj:0.2f}'
        fig.text(textx[j], texty[j],'ACC (detrend) RMSE (detrend)', fontsize=label_small)
        if j == 0:
            plt.legend(fontsize=label_small)
        if i == 0:
            fig.text(textx[j], textyd[j],f'{accformatted} ({accdetrendedformatted})       {rmseformatted} ({rmsedetrendedformatted})', fontsize=label_small,color=col)
            fig.text(textxps[j], textyps[j],f'{accpsformatted} ({accpsdetrendedformatted})       {rmsepsformatted} ({rmsepsdetrendedformatted})', fontsize=label_small,color='gray')
        elif i == 1:
            fig.text(textx[j], textydd[j],f'{accformatted} ({accdetrendedformatted})       {rmseformatted} ({rmsedetrendedformatted})', fontsize=label_small,color=col)
#------------------------------------------------------
#------------------------------------------------------ 








