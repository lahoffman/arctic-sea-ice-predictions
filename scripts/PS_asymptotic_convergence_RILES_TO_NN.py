#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:41:36 2024

@author: hoffmanl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:04:02 2024

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
import matplotlib.cm as cm
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

#cor = ['saddlebrown','maroon','firebrick','red','tomato','salmon','lightsalmon','peachpuff','orange','gold']
#cog = ['darkslategrey','teal','cadetblue','lightseagreen','mediumaquamarine','mediumseagreen','seagreen','green','darkgreen','darkolivegreen']
cor = cm.get_cmap('plasma',11)
cog = cm.get_cmap('viridis',11)

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(60,40))
label_large = 80
label_small = 50
label_mid = 60

for m in range(11):
    yearstart = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]


    # LOAD TO 
    #------------------------------------------------------
    #------------------------------------------------------
    loadlead = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_TO_JUL' 
    loadtail = '.nc'
    loadyear = str(yearstart[m])
    loadpath = loadlead+loadyear+loadtail
    #loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_TO_JUL2005.nc'
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
    co= cor(m)

    
    # Create a figure and gridspec layout
    for j in range(2):   
        ps = 13 #september, predicted time frame
        
        #---------------------------------
        #*********************************
        #TOGGLE TO DESIRED INTEREST
        pp = 11 #starting time frame
        year_of_interest = yearstart[m]
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
                    #ax.plot(time[k,i,ti:ti+10],data_residual[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')         
                    
                    #plot linear fit
                    lte = np.shape(timeextend)[2]
                    ax.plot(timeextend[k,i,:],np.zeros([lte,1]),linewidth=5,color='grey',label='_nolegend_')
                   
                    #plot prediction
                    #ax.errorbar(timedurpred[k,i],pred_residual[k,i],error_residual[k,i],color=co,fmt='o',markersize=30,capsize=10,label='_nolegend_')
                    ax.plot(timedurpred[k,:],pred_residual[k,:],color=co,linewidth=5,label='_nolegend_')
                    ax.set_xlim([1979,2023])
                       
                    #plot input sie
                    ax.scatter(time[k,i,ti],data_residual_init[k,i],marker='s',s=1000,color= co,label='_nolegend_')
                   
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
         
                    #fig.text(textx[k], texty[k],'ACC  (7-year)   $\mathregular{R^2}$  (7-year)', fontsize=label_mid)
                    
                    #if l == 0:
                    #    col = 'r'
                    #    fig.text(textx[k], textyd[k],f'{accformatted}   ({accformatted6})   {codformatted}    ({codformatted6})', fontsize=label_mid,color=col)
                    #elif l == 1:
                    #    col = 'seagreen'
                    #    fig.text(textx[k], textydd[k],f'{accformatted}   ({accformatted6})   {codformatted}    ({codformatted6})', fontsize=label_mid,color=col)
                        
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

                        #plot forced (i.e. fit)
                        ax.plot(time[k,i,:-1],fitko[k,i,:],linewidth=5,color='orchid',label='forced')
                
                        #plot forced + residual, obs
                        ax.plot(time[k,i,:],data[k,i,:],linewidth=5,color='black',label='obs')          
                        
                        #plot forced + residual, obs for prediction time
                        #ax.plot(time[k,i,ti:ti+10],data[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')               
                        
                        #plot linear fit
                        ax.plot(timeextend[k,i,:],linfitk[k,i,:],linewidth=5,color='grey',label='linear fit, obs')
                        
                        #plot prediction
                        #ax.errorbar(timedurpred[k,i],predk[k,i],errork[k,i],color=co,fmt='o',markersize=25,capsize=10,label='pred, TO')
                        ax.plot(timedurpred[k,:],predk[k,:],color=co,linewidth=5)

                        ax.set_ylim([2,9])
                    
                    else:
                        #plot fit
                        #ax.plot(timeextend[:-1],fitk[:-1],linewidth=5,color='orchid',label='_nolegend_')
                        
                        #plot forced + residual, obs
                        ax.plot(time[k,i,:],data[k,i,:],linewidth=5,color='black',label='_nolegend_')               
                       
                        #plot forced + residual, obs for prediction time
                        #ax.plot(time[k,i,ti:ti+10],data[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')         
                        
                        #plot linear fit
                        #ax.plot(timeextend,linfitk,linewidth=5,color='grey',label='_nolegend_')
                       
                        #plot prediction
                        ax.plot(timedurpred[k,:],predk[k,:],color=co,linewidth=5)  
                        #ax.errorbar(timedurpred[k,i],predk[k,i],errork[k,i],color=co,fmt='o',markersize=25,capsize=10,label='_nolegend_')
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
                #if m ==0 & k == 0 & j ==0 & i ==0:
                    #fig.legend(handles, labels, loc='lower center', ncol=5,fontsize=label_large)
#------------------------------------------------------
#------------------------------------------------------ 



fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(60,40))
label_large = 80
label_small = 50
label_mid = 60

for m in range(11):
    yearstart = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]


    # LOAD 2DNN
    #------------------------------------------------------
    #------------------------------------------------------
    loadlead = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/hindcast_RILES_2DNN_JUL' 
    loadtail = '.nc'
    loadyear = str(yearstart[m])
    loadpath = loadlead+loadyear+loadtail
    #loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d1/hindcast_RILES_TO_JUL2005.nc'
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
    #data = np.array(dataset.variables['dataknn'])
    co= cog(m)

    
    # Create a figure and gridspec layout
    for j in range(2):   
        ps = 13 #september, predicted time frame
        
        #---------------------------------
        #*********************************
        #TOGGLE TO DESIRED INTEREST
        pp = 11 #starting time frame
        year_of_interest = yearstart[m]
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
                    #ax.plot(time[k,i,ti:ti+10],data_residual[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')         
                    
                    #plot linear fit
                    lte = np.shape(timeextend)[2]
                    ax.plot(timeextend[k,i,:],np.zeros([lte,1]),linewidth=5,color='grey',label='_nolegend_')
                   
                    #plot prediction
                    #ax.errorbar(timedurpred[k,i],pred_residual[k,i],error_residual[k,i],color=co,fmt='o',markersize=30,capsize=10,label='_nolegend_')
                    ax.plot(timedurpred[k,:],pred_residual[k,:],color=co,linewidth=5,label='_nolegend_')
                    ax.set_xlim([1979,2023])
                       
                    #plot input sie
                    ax.scatter(time[k,i,ti],data_residual_init[k,i],marker='s',s=1000,color= co,label='_nolegend_')
                   
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
         
                    #fig.text(textx[k], texty[k],'ACC  (7-year)   $\mathregular{R^2}$  (7-year)', fontsize=label_mid)
                    
                    #if l == 0:
                    #    col = 'r'
                    #    fig.text(textx[k], textyd[k],f'{accformatted}   ({accformatted6})   {codformatted}    ({codformatted6})', fontsize=label_mid,color=col)
                    #elif l == 1:
                    #    col = 'seagreen'
                    #    fig.text(textx[k], textydd[k],f'{accformatted}   ({accformatted6})   {codformatted}    ({codformatted6})', fontsize=label_mid,color=col)
                        
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

                        #plot forced (i.e. fit)
                        ax.plot(time[k,i,:],fitko[k,i,:],linewidth=5,color='orchid',label='forced')
                
                        #plot forced + residual, obs
                        ax.plot(time[k,i,:],data[k,i,:],linewidth=5,color='black',label='obs')          
                        
                        #plot forced + residual, obs for prediction time
                        #ax.plot(time[k,i,ti:ti+10],data[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')               
                        
                        #plot linear fit
                        ax.plot(timeextend[k,i,:],linfitk[k,i,:],linewidth=5,color='grey',label='linear fit, obs')
                        
                        #plot prediction
                        #ax.errorbar(timedurpred[k,i],predk[k,i],errork[k,i],color=co,fmt='o',markersize=25,capsize=10,label='pred, TO')
                        ax.plot(timedurpred[k,:],predk[k,:],color=co,linewidth=5,label='10-year prediction')

                        ax.set_ylim([2,9])
                    
                    else:
                        #plot fit
                        #ax.plot(timeextend[:-1],fitk[:-1],linewidth=5,color='orchid',label='_nolegend_')
                        
                        #plot forced + residual, obs
                        ax.plot(time[k,i,:],data[k,i,:],linewidth=5,color='black',label='_nolegend_')               
                       
                        #plot forced + residual, obs for prediction time
                        #ax.plot(time[k,i,ti:ti+10],data[k,i,ti:ti+10],linewidth=6,color='blue',label='_nolegend_')         
                        
                        #plot linear fit
                        #ax.plot(timeextend,linfitk,linewidth=5,color='grey',label='_nolegend_')
                       
                        #plot prediction
                        ax.plot(timedurpred[k,:],predk[k,:],color=co,linewidth=5)  
                        #ax.errorbar(timedurpred[k,i],predk[k,i],errork[k,i],color=co,fmt='o',markersize=25,capsize=10,label='_nolegend_')
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
                #if m ==0 & k == 0 & j ==0 & i ==0:
                    #fig.legend(handles, labels, loc='lower center', ncol=5,fontsize=label_large)