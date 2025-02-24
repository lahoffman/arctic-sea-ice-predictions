#!/usr/bin/env python3
# -*- coding: utf-8 -*-



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


loadpath = '/Users/hoffmanl/Documents/data/manuscripts/ms1/d3/performance_CI_TO1D_NN2D_PM.nc'
dataset =  nc.Dataset(loadpath,'r')
to_cod = dataset.variables['to_cod']
to_rel = dataset.variables['to_rel']
to_cod_CI = dataset.variables['to_cod_CI']
to_rel_CI  = dataset.variables['to_rel_CI']

nn_cod = dataset.variables['nn_cod']
nn_rel = dataset.variables['nn_rel']
nn_cod_CI  = dataset.variables['nn_cod_CI']
nn_rel_CI  = dataset.variables['nn_rel_CI']

cod_ps = dataset.variables['cod_ps']

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
        cod = np.array(to_cod)
        rel = np.array(to_rel)
        relCI = np.array(to_rel_CI)
        codCI = np.array(to_cod_CI)
        
    elif j == 1:
        cod = np.array(nn_cod)
        rel= np.array(nn_rel)
        relCI = np.array(nn_rel_CI)
        codCI = np.array(nn_cod_CI)
        
    else:
        cod = np.array(nn_cod)-np.array(to_cod)
        rel = np.sqrt(np.square(np.array(to_rel)-1))-np.sqrt(np.square(np.array(nn_rel)-1))
        
    
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
        data_rel_CI_low = np.round(np.transpose(relCI[mm1:mm2,:,i,0]),1)
        data_rel_CI_high = np.round(np.transpose(relCI[mm1:mm2,:,i,1]),1)
        data_cod_CI_low = np.transpose(codCI[mm1:mm2,:,i,0])
        data_cod_CI_high = np.transpose(codCI[mm1:mm2,:,i,1])
        
        #mask = ((0.8 < data_rel) & (data_rel < 1.2)) & (data_corr > 0.1) & (data_corr > persistence)
        mask = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high)) & (data_cod_CI_low > 0) & (data_corr > persistence)
        X, Y = np.meshgrid(xx,ff)
        num_elements_less_than_minus_one = np.sum(data1 < -1)
        masked_Z = np.ma.masked_less(data1, -1)
        maskcod = (data_cod_CI_low > 0) & (data_corr > persistence)
        maskrel = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high))
        
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
        mask = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high)) & (data_cod_CI_low > 0) & (data_corr > persistence)
        maskint = mask.astype(int)
        X, Y = np.meshgrid(xx,ff)
        maskcod = (data_cod_CI_low > 0) & (data_corr > persistence)
        maskrel = ((data_rel_CI_low <= 1) & (1 <= data_rel_CI_high))
        
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
        cb1.set_label(r'$cod_{nn} - cod_{to}$', fontsize=label_big) 
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
        cb2.set_label(r'$\sqrt{(rel_{to}-1)^2} - \sqrt{(rel_{nn}-1)^2}$', fontsize=label_big) 
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
    fig.text(0.23, 0.335,r'(e) $\Delta (NN, TO)$', fontsize=label_big)
    fig.text(0.63, 0.335,r'(f) $\Delta (NN, TO)$', fontsize=label_big)
    fig.text(0.2, 0.605,'(c) neural network (NN)', fontsize=label_big)
    fig.text(0.6, 0.605,'(d) neural network (NN)', fontsize=label_big)
    fig.text(0.2, 0.875,'(a) transfer operator (TO)', fontsize=label_big)
    fig.text(0.6, 0.875,'(b) transfer operator (TO)', fontsize=label_big)

#------------------------------------------------------
#------------------------------------------------------

percentskillfulNN = np.sum(maskcod1[:,0:60])/600
percentskillfulTO = np.sum(maskcod0[:,0:60])/600
percentskillfulTO10 = np.sum(maskcod0)/1200
oneyearjulyTO_cod = to_cod[6,0,0]
oneyearjulyTOrel = to_rel[6,0,0]

maskcod1reshape = np.reshape(maskcod1[:,0:60],(10,5,12))
maskcod0reshape = np.reshape(maskcod0[:,0:60],(10,5,12))
monthlymask1 = np.sum(maskcod1reshape,axis=(0,1))
monthlymask0 = np.sum(maskcod0reshape,axis=(0,1))