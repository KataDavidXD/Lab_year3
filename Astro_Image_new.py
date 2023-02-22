# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:44:28 2023

@author: Yang
"""

#import package
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
from astropy.io import fits
from image_function import remove_edge,back,sperate_array,count,split_cluster,error_label,calculate_back,cali,mag
from image_function import plot_sperated_clusters,find_magnitude,gaussian_fit,makeGaussian,clusterAndNoise,get_N
from image_function import log10it,linear

'''
Load data, visuilization, and Guassian fit
'''

hdulist = fits.open("C:/Users/DAVID/Desktop/Lab/Image/A1_mosaic/A1_mosaic.fits") 

header_0 = hdulist[0].header
#print(header_0)

data_0 = hdulist[0].data
#print(data_0)

MAGZPT = header_0["MAGZPT"]
MAGZRR = header_0["MAGZRR"]

data_f = data_0.flatten()

plt.figure()
plt.imshow(data_0, cmap='gray')
plt.colorbar()
plt.show()

'''
Remove edges
'''
data_1=data_0.copy()
data_1 = remove_edge(data_1)

    
class back_back():
    def __init__(self):
        self.Value = 3475
        
'''
Set the background value and remove it 
'''
background_V = back_back()
back_V = background_V.Value
data_1_back = back(data_1,back_V)        
#%%
'''
Analysis of the whole data set
'''

plt.figure()
x_contourf, y_contourf = np.shape(data_1_back)
contours = plt.contourf(range(0,y_contourf),range(0,x_contourf),data_1_back, colors='black')
plt.contourf(range(0,y_contourf),range(0,x_contourf),data_1_back,cmap='gist_heat')  
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
plt.contourf(range(0,y_contourf_test),range(0,x_contourf_test),data_test1,cmap='gist_heat')  
plt.clabel(contours, inline=True, fontsize=8) 
plt.colorbar()    
sperated_plot = plot_sperated_clusters(splited_cluster)
plt.legend()
plt.show()

print('The number of clusters is '+str(len(splited_cluster)))


all_mag = find_magnitude(splited_cluster,data_1,error_mask_signal,MAGZPT)

plt.figure()
mag_hist=plt.hist(all_mag, 20,  histtype='step', cumulative=True, label='Empirical')
plt.show()

xdata = mag_hist[1]

x_data=np.array([])
for i in range(0,len(xdata)-1):
    dummyv = np.array([(xdata[i]+xdata[i+1])/2])
    if dummyv < 20:
        x_data = np.append(x_data,dummyv)

y_data = mag_hist[0]
y_data = y_data[:len(x_data)]
plt.figure()
plt.title('Number of stars and magnitude at the background intensity of ' + str(back_V))
y_data = log10it(y_data)
plt.scatter(x_data,y_data, color = "k")

fit_linear, cov_linear = curve_fit(linear, x_data, y_data, 0, maxfev = 1000)
plt.plot(x_data,linear(x_data,fit_linear[0]),label='Theoretic Line')
plt.grid()
plt.xlabel('Magnitude')
plt.ylabel('Number of the cluster in Log10')
plt.legend()
plt.show()
#%%
def new_linear(x,a,c):
    y = a * x + c
    return y

guess =[0.6,0]
fit_linear, cov_linear = curve_fit(new_linear, x_data, y_data, *guess, maxfev = 1000)

plt.plot(x_data,new_linear(x_data,fit_linear[0],fit_linear[1]),label='Theoretic Line')
plt.grid()
plt.xlabel('Magnitude')
plt.ylabel('Number of the cluster in Log10')
plt.legend()
plt.show()


