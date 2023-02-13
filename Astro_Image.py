# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:44:28 2023

@author: Yang
"""

#import package
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

'''
Load data, visuilization, and Guassian fit
'''

hdulist = fits.open("C:/Users/DAVID/Desktop/Lab/Image/A1_mosaic/A1_mosaic.fits") 

header_0 = hdulist[0].header
#print(header_0)

data_0 = hdulist[0].data
#print(data_0)

data_f = data_0.flatten()

data_0_x = data_0.shape[1]
data_0_y = data_0.shape[0]

plt.figure()
plt.imshow(data_0, cmap='gray')
plt.colorbar()
plt.show()
#%%  remove edges    (4611, 2570) 

data_1=data_0

#top
for i in range(0,2570):
    for n in range(4511,4611):
        data_1[n,i]=0
#bottom
for i in range(0,2570):
    for n in range(0,25):
        data_1[n,i]=0
#bottom left
for i in range(0,425):
    for n in range(0,120):
        data_1[n,i]=0
for i in range(0,120):
    for n in range(0,420):
        data_1[n,i]=0       
# left
for i in range(0,115):
    for n in range(4515,4611):
        data_1[n,i]=0      
#right
for i in range(2478,2570):
    for n in range(0,4611):
        data_1[n,i]=0
        

'''
Set the background value and remove it 
'''

def back(x,back):
    xnew = x.copy()
    print('copy success')
    ii,jj = np.shape(data_1)
    print('Removing backgroud signal, please wait')
    for i in range(0,ii):
        for j in range(0,jj):
            if x[i,j] < back :
                xnew[i,j] = 0            
    print('Successfully Removed Backgroud')
    return xnew


'''
Analysis of the whole data set
'''

def count(x,h):
    xnew = x.copy()
    print('copy success')
    lenx,leny = np.shape(x)
    mask = np.zeros((lenx,leny))
    print('Creating mask signals, please wait')
    maxpoints=[]
    for i in range(0,lenx):
        for j in range(0,leny):
            if xnew[i,j]> 4000 and mask[i,j] == 0:
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
    clustering = DBSCAN(eps=3).fit(x)
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
        if i == len(cluster)-1:
            e_mask_signal[i] = 1
        elif len(cluster[i]) > 5000:
            e_mask_signal[i] = 2
    return e_mask_signal


'''
Background Flux 
'''


def calculate_back(x):    
    print('Calculating Background intensity')
    for i in range(0,len(x)):
        n = 0
        back_s = 0
        if x[i] < 4000:
            back_s += x[i]
            n += 1
    I_background = back_s/n
    return I_background
    
MAGZPT = header_0["MAGZPT"]
MAGZRR = header_0["MAGZRR"]

def cali(x):
    if x == 0 or x < 0:
        return 'error encoutered, detected magnitude is 0 or negative'
    else:        
        mag = MAGZPT - 2.5 *math.log10(x)
        return mag

def mag(cluster,intensity,er_mask_signal):
    IT = []
    #print(len(cluster))
    for i in range(0,len(cluster)):
        if er_mask_signal == 0:
            px,py = cluster[i]
            #print(px,py)
            IT.append(intensity[px,py])
    total_mag = sum(IT)
    return total_mag,IT


def find_magnitude(cluster,intensity,err_mask_signal):
    print('Finding magnitude of the splited cluters, please wait')
    cali_mag = []
    for nstar in range(0,len(cluster)):
        mag_v = mag(cluster[nstar],intensity,err_mask_signal[nstar])
        mag_ca = cali(mag_v[0])
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

#%%


data_1_back = back(data_1,4000)        
print(data_1_back)

plt.figure()
x_contourf, y_contourf = np.shape(data_1_back)
contours = plt.contourf(range(0,y_contourf),range(0,x_contourf),data_1_back, colors='black')
plt.contourf(range(0,y_contourf),range(0,x_contourf),data_1_back)  
plt.clabel(contours, inline=True, fontsize=8) 
plt.colorbar()    
plt.show()


data_test1 = data_1_back
maxcount = count(data_test1,1)
splited_cluster = split_cluster(maxcount)
error_mask_signal = error_label(splited_cluster)



plt.figure()
x_contourf_test, y_contourf_test = np.shape(data_test1)
contours = plt.contourf(range(0,y_contourf_test),range(0,x_contourf_test),data_test1, colors='black')
plt.contourf(range(0,y_contourf_test),range(0,x_contourf_test),data_test1)  
plt.clabel(contours, inline=True, fontsize=8) 
plt.colorbar()    
for i in range(0,len(splited_cluster)):
    xi,yi = sperate_array(splited_cluster[i])
    if i == len(splited_cluster)-1:
        plt.scatter(yi,xi,label='Noise')
    elif len(splited_cluster[i]) > 5000:
        plt.scatter(yi,xi,label='Detection Error')
    else:
        plt.scatter(yi,xi,label=str(i))
plt.legend()
plt.show()


clustering = DBSCAN(eps=3).fit(maxcount)
labels = clustering.labels_
unique_labels = set(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


intensity_background = calculate_back(data_f)
print('Average background intensity is '+ str(intensity_background))

all_mag = find_magnitude(splited_cluster,data_1,error_mask_signal)



#%%
#Clean data
data=[]
for i in range(0,len(data_f)):
    dummydata = data_f[i]
    if dummydata > 3300 and dummydata < 3600:
        data.append(dummydata)
    
#print(data)
print('\n Image Loading success')

#visuilize Histograme
datav = plt.hist(data,density = True, cumulative =False,  bins = 100,label='Histogram')
print('\n Visulization success')

x_data = datav[1][1:]
y_data = datav[0]
y_peak = max(y_data)

guess = [y_peak,3400,0.5]
fit_gau, cov_gau = curve_fit(gaussian_fit, x_data, y_data, guess, maxfev = 10000)
y_fitted = gaussian_fit(x_data, fit_gau[0], fit_gau[1],fit_gau[2])

plt.grid()
datav = plt.hist(data,density = True, cumulative =False,  bins = 100,label='Histogram')
plt.plot(x_data,y_fitted, label = "Gaussian fit ")
plt.legend()
plt.show()

print('\n Gaussian fit success')
print('The mean value of the Gaussian fit is ' + str(fit_gau[1]) +' with the error of ' + str(np.sqrt(cov_gau[1][1])) +' and the sigma is ' + str(fit_gau[2]) +' with the error of ' + str(np.sqrt(cov_gau[2][2])))

#%%
N=500
np.zeros((N,)*2)
plt.figure()
plt.imshow(makeGaussian(500, 10, center=(240,230)))
plt.colorbar()
plt.show()