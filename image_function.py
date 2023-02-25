# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:51:34 2023

@author: DAVID
"""

from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import gamma, factorial
from scipy.optimize import curve_fit
from scipy import signal 
import math
from scipy.fft import fft, fftfreq, fftshift,rfft,rfftfreq, ifft
from scipy.stats import norm
import scipy.stats as ss
from scipy.stats import levy  
from scipy import interpolate
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
import  yfinance  as yf
import plotly.graph_objects as go
from astropy.io import fits

back_V = 3470
#error_length_t = 1600
error_length_t = 1450
'''
from Astro_Image import back_back
background_V = back_back()
back_V = background_V.Value
'''

'''
Remove Edge
'''        
def remove_edge(x,top=0,b1=0,b2=0,left=0,rifht=0):
    for i in range(0,2570):
        for n in range(4511,4611):
            x[n,i]=0
    #bottom
    for i in range(0,2570):
        for n in range(0,25):
            x[n,i]=0

    #bottom left
    for i in range(0,425):
        for n in range(0,120):
            x[n,i]=0
    for i in range(0,120):
        for n in range(0,420):
            x[n,i]=0       
    # left
    for i in range(0,115):
        for n in range(4515,4611):
            x[n,i]=0      
    #right
    for i in range(2478,2570):
        for n in range(0,4611):
            x[n,i]=0

    for i in range(0,26):
        for n in range(0,4611):
            x[n,i]=0
    


    for i in range(0,97):
        for n in range(0,4611):
            x[n,i]=0

    for i in range(0,2570):
        for n in range(0,100):
            x[n,i]=0
    

    for i in range(0,2570):
        for n in range(4515,4611):
            x[n,i]=0  

    for i in range(2165,2476):
        for n in range(4428,4515):
            x[n,i]=0        

    for i in range(2353,2476):
        for n in range(4210,4513):
            x[n,i]=0        
    
    for i in range(2476,2570):
        for n in range(0,4611):
            x[n,i]=0        


    return x            

'''
Set the background value and remove it 
'''

def back(x,back):
    xnew = x.copy()
    print('copy success')
    ii,jj = np.shape(x)
    print('Removing backgroud signal, please wait')
    for i in range(0,ii):
        for j in range(0,jj):
            if x[i,j] < back :
                xnew[i,j] = 0            
    print('Successfully Removed Backgroud')
    return xnew


def count(x,h):
    xnew = x.copy()
    print('copy success')
    lenx,leny = np.shape(x)
    mask = np.zeros((lenx,leny))
    print('Creating mask signals, please wait')
    maxpoints=[]
    for i in range(0,lenx):
        for j in range(0,leny):
            if xnew[i,j]> back_V and mask[i,j] == 0:
                xmax = xnew[i,j]
                x_x = i
                j_j = j
                for ii in range(i-h,i+h):
                    for jj in range(j-h,j+h):
                        mask[i:i+h,j:j+h] = 1
                        if xnew[ii,jj] > xmax:
                            xmax = xnew[ii,jj] 
                            x_x = ii
                            j_j = jj
                maxpoints.append([x_x,j_j])
    return maxpoints

def sperate_array(x):
    x_max = []
    y_max = []
    for i in range(0,len(x)):
        maxcountx,maxcounty  =  x[i]
        x_max.append(maxcountx)
        y_max.append(maxcounty)
    return x_max,y_max


def split_cluster(x):    
    clustering = DBSCAN(eps=10,min_samples=5).fit(x)
    print('Density clustering machine learning algorithm is running, please wait')
    labels = clustering.labels_
    unique_labels = set(labels)    
    unique_labels = list(unique_labels)
    seprated_cluster = []
    for i in unique_labels:
        #print(i)
        seprated_cluster.append([])
        for ii in range(0,len(labels)):        
            if labels[ii] == unique_labels[i]:   
                seprated_cluster[i].append(x[ii])
    print('Density clustering machine learning algorithm completed')
    return seprated_cluster

def error_label(cluster):    
    e_mask_signal = np.zeros_like(cluster)
    for i in range(0,len(cluster)):
        xi,yi = sperate_array(cluster[i])
        if i == len(cluster)-1 and len(cluster) >1:
            e_mask_signal[i] = 1
        elif len(cluster[i]) > error_length_t:
            e_mask_signal[i] = 2
    return e_mask_signal
'''
def plot_sperated_clusters(s_cluster):
    for i in range(0,len(s_cluster)):
        xi,yi = sperate_array(s_cluster[i])
        if i == len(s_cluster)-1 and len(s_cluster) >1:
            plt.scatter(yi,xi,color= 'black',label='Noise')
        elif len(s_cluster[i]) > 200000:
            plt.scatter(yi,xi,color= 'black',label='Detection Error')
        else:
            #plt.scatter(yi,xi,label=str(i))
            plt.scatter(yi,xi)
'''
def plot_sperated_clusters(s_cluster,inten):
    for i in range(0,len(s_cluster)):
        xi,yi = sperate_array(s_cluster[i])
        if i == len(s_cluster)-1 and len(s_cluster) > 1:
            plt.scatter(yi,xi,color= 'black',label='Noise',zorder=1)
        elif len(s_cluster[i]) > error_length_t:
            plt.scatter(yi,xi,color= 'black',label='Detection Error',zorder=2)
        else:
            #plt.scatter(yi,xi,label=str(i))
            sz=1
            c=inten[xi,yi]
            plt.scatter(yi,xi,sz,c,cmap='Reds',zorder=3)

'''
Background Flux 
'''


def calculate_back(x):    
    print('Calculating Background intensity')
    for i in range(0,len(x)):
        n = 0
        back_s = 0
        if x[i] < back_V:
            back_s += x[i]
            n += 1
    I_background = back_s/n
    print('Average background intensity is '+ str(I_background))
    return I_background
    

def cali(x,MAGZPT):
    if x == 0 or x < 0:
        return 'error encoutered, detected magnitude is 0 or negative'
    else:        
        mag = MAGZPT - 2.5 *math.log10(x)
        return mag
'''
def cali(x,MAGZPT):
    mag = MAGZPT - 2.5 *math.log10(x)
    return mag
'''

def mag(cluster,intensity,er_mask_signal):
    IT = []
    #print(len(cluster))
    for i in range(0,len(cluster)):
        if er_mask_signal == 0:
            px,py = cluster[i] 
            #print(px,py)
            IT.append(intensity[px,py] - back_V)
    total_mag = sum(IT)
    return total_mag,IT


def find_magnitude(cluster,intensity,err_mask_signal,MAGZPT):
    print('Finding magnitude of the splited cluters, please wait')
    cali_mag = []
    for nstar in range(0,len(cluster)):
        mag_v = mag(cluster[nstar],intensity,err_mask_signal[nstar])
        mag_ca = cali(mag_v[0],MAGZPT)
        if err_mask_signal[nstar] == 0:
            cali_mag.append(mag_ca)
    print('Magnitude of the splited cluters are founded')
    return cali_mag

'''
Fitness of the graph
'''

def gaussian_fit(x,c,mu,sigma):
    dummy1 = c / (sigma * np.sqrt(2*np.pi))
    dummy2 = np.exp( (-1/2)* ((x-mu)/(sigma))*((x-mu)/(sigma)) )
    return dummy1*dummy2

    
def makeGaussian(size, fwhm = 3, center=None):
    N=500
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    distri=np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    for i in range(0,N):
        for n in range(0,N):
            if distri[n,i]<0.01:
                distri[n,i]=0

    return distri
'''
def clusterAndNoise(eps_v,x):   
    clustering = DBSCAN(eps=5,min_samples=10).fit(x)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    return [n_clusters_,n_noise_]
'''
'''
𝑙𝑜𝑔 𝑁 ( 𝑚 ) = 0 . 6 𝑚 + 𝑐𝑜𝑛𝑠𝑡𝑎𝑛𝑡
'''

def get_N(x,m):
    N = 0
    mx,my = np.shape(x)
    print('Counting Number of stars at magnitude '+ str(m))
    for i in range(0,mx):
        for j in range(0,my):
            intensity = x[i,j]
            if intensity > m:
                N += 1
    print('Star Counting finished')
    N_log = np.log10(N)
    return N_log



def log10it(x):
    y = []
    for i in range(0,len(x)):
        if x[i] >0:
            y.append(math.log10(x[i]))
    print('Checking the number of the stars')
    return y

def linear(x,c):
    y =  -math.log10(0.6) * x + c
    #y = np.log10(y)
    return y

def llinear(x,c):
    y =  (0.6) * x + c
    #y = np.log10(y)
    return y

def newlinear(x,a,c):
    y =  a * x + c
    #y = np.log10(y)
    return y
