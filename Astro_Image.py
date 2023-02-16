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
#%%
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

    
global back_V
back_V = 3499
        
'''
Set the background value and remove it 
'''
data_1_back = back(data_1,back_V)        

'''
Analysis of the whole data set
'''
#%%
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

n_clusters_noise = clusterAndNoise(3,maxcount)

intensity_background = calculate_back(data_f)

all_mag = find_magnitude(splited_cluster,data_1,error_mask_signal,MAGZPT)
'''

def remove(data, mean):
    return data-mean

def collect1():
    data=data_1.copy()
    test=back(data,i)
    maxcount=count(test,1)
    splited_cluster = split_cluster(maxcount)
    print(len(splited_cluster))
    test = remove(test,3418)
    m = find_magnitude(splited_cluster,test,error_label(splited_cluster))
    return m

m=collect1()

'''
#%% 

mag_hist=plt.hist(all_mag, 30,  histtype='step', cumulative=True, label='Empirical')
x_data = mag_hist[1][1:]
y_data = mag_hist[0]

plt.figure()
y_data = log10it(y_data)
plt.scatter(x_data,y_data, color = "k")

fit_linear, cov_linear = curve_fit(linear, x_data, y_data, 0, maxfev = 1000)
plt.plot(x_data,linear(x_data,fit_linear[0]),label='Theoretic Line')
plt.xlabel('Magnitude')
plt.ylabel('Number of the cluster')
plt.legend()
plt.show()







#%%
'''
Ignore all below
'''

#%%
def intm(x):
    im = []
    for i in range(0,len(x)):
        im.append((round(x[i],2)))
    return im
        
int_m = intm(all_mag)

def count_int_m(x):
    m = np.unique(x)
    N=[]
    M=[]
    for i in m:
        M.append(i)
        n = 0
        for j in range(0,len(m)):
            if m[j] == i:
                n+= 1
        N.append(n)
    return M,N

int_N_m = count_int_m(all_mag)



plt.figure()
plt.xlabel('Magnitude')
plt.ylabel('Number of cluster at that magnitude')
plt.grid()
mag_hist = plt.hist(all_mag, 25, density=True, histtype='step',cumulative=True, label='Empirical')
plt.yscale('log')
plt.show()


x_data = mag_hist[1][1:]
y_data = mag_hist[0]
plt.scatter(x_data,y_data, color = "k")

def linear(x,c):
    y = 0.6 * x + c
    return y


fit_linear, cov_linear = curve_fit(linear, x_data, y_data, 0, maxfev = 1000)
plt.plot(x_data,linear(x_data,0))
plt.plot(x_data,linear(x_data,fit_linear[0]))
plt.show()



#%%
#Test
#number_30000 = get_N(data_0,30000)

data0_MAX = data_0.max()
data0_MIN = data_0.min()
step = int((data0_MAX - data0_MIN)/1000)

Number_ALL = [] 
for mag_interval in np.linspace(data0_MIN,data0_MAX,step):
    number = get_N(data_0,mag_interval)
    Number_ALL.append(number)










#%%
'''
Previous Work for Gaussian
'''
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

plt.figure()
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