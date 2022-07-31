
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

class sharp_patch_generator(object):
    def __init__(self,image, thres):
        self.thres = thres
        self.original = image
        self.gray = image.squeeze()#cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #
        self.patch_generator()
    
    def image_colorfulness(self, image):
    	# split the image into its respective RGB components
    	(B, G, R) = cv2.split(image.astype("float"))
    	# compute rg = R - G
    	rg = np.absolute(R - G)
    	# compute yb = 0.5 * (R + G) - B
    	yb = np.absolute(0.5 * (R + G) - B)
    	# compute the mean and standard deviation of both `rg` and `yb`
    	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
    	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
    	# combine the mean and standard deviations
    	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    	# derive the "colorfulness" metric and return it
    	return stdRoot + (0.3 * meanRoot)

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
      
    def extract_patches(self,img_mscn, sigma):
    
        patches = []
        sharpness = []
        row = np.shape(img_mscn)[0]
        col = np.shape(img_mscn)[1]
        size = 96
        for j in range(0,(row-size),96):
            for k in range(0,(col-size),96):
                patches.append(img_mscn[j:j+size,k:k+size])
                sharpness.append(np.mean(sigma[j:j+size,k:k+size]))
    #    patches = np.array(patches)        
    #    patches = patches.astype('float32')/255
    #    dist_patches = np.clip(dist_patches, a_min = 0, a_max = 1)
        return (patches, sharpness)
    

    def patch_generator(self):
        img_mscn,sigma = self.calculate_mscn_coefficients(self.gray/0xff)
#        patches_mscn, sigma_patches = self.extract_patches(np.expand_dims(img_mscn, -1), sigma)
        patches_mscn, sigma_patches = self.extract_patches(self.original, sigma)
        idx = np.where(sigma_patches>=self.thres*np.max(sigma_patches))[0]
        
        output_patches, output_sigma = [], []
        for k in range(len(idx)):
    #        cv2.imwrite('/home/neel/Desktop/live_sharp_patches/'+ str(count).zfill(6) + '.png', patches_mscn[idx[0][k]])
            output_patches.append(patches_mscn[idx[k]])
            output_sigma.append(sigma_patches[idx[k]])
        output_patches = np.array(output_patches)
        output_sigma  = np.array(output_sigma)
    #        plt.imshow(patches_mscn[idx[0][k]])
            
    #        np.save('/home/neel/Desktop/sharp_patches/'+ str(count).zfill(6), np.expand_dims(patches_mscn[idx[0][k]], -1))
    
        return(output_patches, output_sigma)
