
from keras import backend as K

import numpy as np
from scipy import io as sio
import cv2
import os
import glob
from scipy.ndimage import gaussian_filter
#K.clear_session()
#import matplotlib.pyplot as plt
from scipy import signal

class flow_based_patch_generator(object):
    def __init__(self, temporal, spatial, thres):
        self.thres = thres
        self.framediff = temporal
        self.gray = temporal.squeeze()#cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.spatial = spatial.squeeze()
        self.patch_generator()
        

    def normalize_kernel(self,kernel):
        return(kernel/np.sum(kernel))
    
    def gaussian_kernel2d(self,n,sigma):
        Y, X = np.indices((n,n)) - int(n/2)
        gaussian_kernel = 1/(2*np.pi * sigma **2) * np.exp(-(X**2 + Y**2) / (2*sigma**2))
        return (self.normalize_kernel(gaussian_kernel))
    
    def local_mean(self,image, kernel):
        return (signal.convolve2d(image, kernel, 'same'))
    def local_deviation(self,image, local_mean, kernel):
        
        sigma = image **2
        sigma = signal.convolve2d(sigma, kernel, 'same')
        return (np.sqrt(np.abs(local_mean**2 - sigma)))
    
    def calculate_mscn_coefficients(self,image, kernel_size =7, sigma =7/6):
        C = 1/0xff
        kernel = self.gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = signal.convolve2d(image, kernel, 'same')
        local_var = self.local_deviation(image, local_mean, kernel)
        return( (image - local_mean)/(local_var + C), local_var )
        
    def calculate_sigma(self, image, kernel_size =7, sigma =7/6):
        kernel = self.gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = signal.convolve2d(image, kernel, 'same')
        local_var = self.local_deviation(image, local_mean, kernel)
        return (local_var)
    
    def extract_patches(self, framediff, sigma_spa):
    
        patches_diff = []
        sharpness_spa = []
        row = np.shape(framediff)[0]
        col = np.shape(framediff)[1]
        size = 96
        for j in range(0,(row-size),96):
            for k in range(0,(col-size),96):
                patches_diff.append(framediff[j:j+size,k:k+size])                
                sharpness_spa.append(np.mean(sigma_spa[j:j+size,k:k+size]))
#                sharpness_tmp.append(np.mean(sigma_tmp[j:j+size,k:k+size]))
        return (patches_diff,  np.array(sharpness_spa))
    

    def patch_generator(self):
#        img_mscn,sigma = self.calculate_mscn_coefficients(self.gray/0xff)
        
        sigma_spatial = self.calculate_sigma(self.spatial/0xff)
#        sigma_temporal = self.calculate_sigma(self.gray/0xff)
        
    #    patches_mscn, sigma_patches = extract_patches(img_mscn, sigma)
        framediff_list,sigma_patches_spatial = self.extract_patches(self.framediff, sigma_spatial)#,  sigma_temporal)
                                                                                          
        sigma_spatio_temporal = 1*sigma_patches_spatial# + 0*sigma_patches_temporal
        
#        idx = np.where((sigma_patches_spatial>=self.thres*np.max(sigma_patches_spatial)) & \
#                       (sigma_patches_temporal>=0.1*np.max(sigma_patches_temporal)))[0]
        
        idx = np.where(sigma_spatio_temporal>=self.thres*np.max(sigma_spatio_temporal))[0]
        
        output_diff, output_sigma = [], []
        for k in range(len(idx)):
            output_diff.append(framediff_list[idx[k]])
#            output_sigma.append(sigma_patches_spatial[idx[k]])
            output_sigma.append(sigma_spatio_temporal[idx[k]])
            
            
        output_diff = np.array(output_diff)
        
        output_sigma  = np.array(output_sigma)
    #        plt.imshow(patches_mscn[idx[0][k]])
            
    #        np.save('/home/neel/Desktop/sharp_patches/'+ str(count).zfill(6), np.expand_dims(patches_mscn[idx[0][k]], -1))
    
        return(output_diff, output_sigma)
