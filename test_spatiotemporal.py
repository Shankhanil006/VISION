import tensorflow as tf
from keras.applications import VGG19
from keras.applications.resnet50 import ResNet50
from keras import models
from keras.layers  import Conv2D, Dense,Flatten,Dropout,GlobalMaxPooling2D, GlobalAveragePooling2D, Input,Lambda, BatchNormalization
from keras import optimizers, layers
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers import GRU, LSTM, TimeDistributed,Layer,concatenate, PReLU, LeakyReLU, Concatenate
from tensorflow.keras.layers import Cropping2D
import glob
import cv2
from scipy import io as sio
import numpy as np
import os
#import matplotlib.pyplot as plt
#from cropping import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from operator import itemgetter
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from scipy.stats import spearmanr as srocc
from scipy.stats import pearsonr as plcc

import keras
from keras.optimizers import Adam, Nadam
from keras.models import load_model

from keras import regularizers
import h5py
from keras import losses

from scipy.io import loadmat
import skvideo.io
from scipy import signal
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.kernel_approximation import Nystroem
from itertools import combinations
import time
import random
import gc
from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt
from clr_multiview_contrastive_loss import *
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from sklearn import preprocessing
from scipy.stats import spearmanr,pearsonr 
from sklearn.decomposition import PCA
from scipy.io import loadmat,savemat
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gc.collect()
K.clear_session()

def distance(mu1, sigma1, mu2, sigma2):
    dist = np.matmul( np.linalg.pinv((sigma1+ sigma2)/2), (mu1 - mu2).T)
#    dist = np.matmul( np.linalg.pinv(sigma1), (mu1 - mu2).T)
    dist = np.matmul((mu1 - mu2), dist)
   
#    dist = ((mu1 - mu2))*( np.linalg.inv((sigma1 + sigma2)/2))*((mu1 - mu2).T)
    dist = np.sqrt(dist)
#    print(dist)
#    dist = (mu1 - mu2)*(mu1 - mu2).T + np.matrix.trace(sigma1 + sigma2 -2*scipy.linalg.sqrtm(sigma1*sigma2))
    return(dist.squeeze())


#syn_feat_spa = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/ablation/syntheticvsauthentic/full_authentic_FvsFD_frame_ref_feat_0x_5000.mat')['feat'].squeeze()
syn_feat_spa = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/test/FvsFD_frame_ref_feat_0x_5000.mat')['feat'].squeeze()

#syn_feat_tmp = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/ablation/syntheticvsauthentic/full_authentic_FvsFD_diff_ref_feat_0x_5000.mat')['feat'].squeeze()
syn_feat_tmp = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/test/FvsFD_diff_ref_feat_0x_5000.mat')['feat'].squeeze()

#pca_spa = PCA(n_components=8)
#pca_tmp = PCA(n_components=8)
#syn_feat_spa = pca_spa.fit_transform(syn_feat_spa)
#syn_feat_tmp = pca_tmp.fit_transform(syn_feat_tmp)


#spa_mu_pris = np.mean(syn_feat_spa, 0).reshape([1,-1])
#spa_cov_pris = np.cov(syn_feat_spa.T)
#
#tmp_mu_pris = np.mean(syn_feat_tmp, 0).reshape([1,-1])
#tmp_cov_pris = np.cov(syn_feat_tmp.T)

spa_tmp_pris= (syn_feat_spa + syn_feat_tmp)/2# np.concatenate([syn_feat_spa, syn_feat_tmp], -1)#
mu_pris = np.mean(spa_tmp_pris, 0).reshape([1,-1])
cov_pris = np.cov(spa_tmp_pris.T)
################################# authentic data ####################################
import csv
flickr_id  = []
konvid_mos = []
konvid_resnet = []
konvid_directory = '/media/ece/DATA/Shankhanil/VQA/konvid/KoNViD_1k_videos/' 
with open(konvid_directory + 'KoNViD_1k_mos.csv', 'r') as file:
    reader = csv.reader(file) 
    for row in reader:
        flickr_id.append(row[0])
        konvid_mos.append(row[1])
           
flickr_id  = flickr_id[1:]
konvid_mos = konvid_mos[1:]
konvid_dmos = np.array([5-float(i) for i in konvid_mos])

liveVQC_dir = '/media/ece/DATA/Shankhanil/VQA/LIVE_VQC/' 
video_list = sio.loadmat(liveVQC_dir+'data.mat')['video_list']
vqc_mos = sio.loadmat(liveVQC_dir+'data.mat')['mos'].squeeze()
vqc_dmos = np.array([100-float(i) for i in vqc_mos])
#names = [file for file in glob.glob(directory +'/*.yuv')]

utube_directory = '/media/ece/DATA/Shankhanil/VQA/youtube_ugc/'
vid_id = loadmat(utube_directory + 'filename.mat')['name']
utube_mos   = loadmat(utube_directory + 'filename.mat')['mos'].squeeze()
utube_dmos = [5-float(i) for i in utube_mos]
utube_dmos = np.array(utube_dmos)

qcomm_directory = '/media/ece/DATA/Shankhanil/VQA/live_qualcomm/'
vid_names = loadmat('/media/ece/DATA/Shankhanil/VQA/live_qualcomm/live_qualcommData.mat')['video_names']
qcomm_mos = loadmat('/media/ece/DATA/Shankhanil/VQA/live_qualcomm/live_qualcommData.mat')['scores'].squeeze()
qcomm_dmos = np.array([100-float(i) for i in qcomm_mos])

directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/'
with open(directory + 'live_video_quality_seqs.txt') as f:
    video_names = f.readlines()
live_video_list = [x.split('.')[0] for x in video_names] 
with open(directory + 'live_video_quality_data.txt') as f:
    video_dmos = f.readlines()
live_dmos_list = [float(x.split('\t')[0]) for x in video_dmos] 
live_dmos = np.array(live_dmos_list)

epfl_cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/'
with open(epfl_cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_video_list = [x.split()[0] for x in file_names] 
epfl_dmos  = [float(x.split()[1]) for x in file_names]

epfl_4cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/'
with open(epfl_4cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_video_list.extend([x.split()[0] for x in file_names])
epfl_dmos.extend([float(x.split()[1]) for x in file_names])

mobile_directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/'
mobile_dmos = sio.loadmat(mobile_directory + 'dmos_live_mobile.mat')['dmos'].squeeze()
mobile_video_list = sio.loadmat(mobile_directory + 'strred_mobile.mat')['names'].squeeze()

csiq_directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
with open(csiq_directory + 'video_subj_ratings.txt') as f:
    file_names = f.readlines()
csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
csiq_dmos = [float(x.split('\t')[1]) for x in file_names]

def mscn(image, kernel_size =7, sigma =7/6):
    image = image.squeeze()
    def normalize_kernel(kernel):
        return(kernel/np.sum(kernel))
    
    def gaussian_kernel2d(n,sigma):
        Y, X = np.indices((n,n)) - int(n/2)
        gaussian_kernel = 1/(2*np.pi * sigma **2) * np.exp(-(X**2 + Y**2) / (2*sigma**2))
        return (normalize_kernel(gaussian_kernel))
    def local_deviation(image, local_mean, kernel):
        
        sigma = image **2
        sigma = signal.convolve2d(sigma, kernel, 'same')
        return (np.sqrt(np.abs(local_mean**2 - sigma)))
    
    C = 1#/0xff
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)
    
    tmp = (image - local_mean)/(local_var + C)
    return( np.expand_dims(tmp, -1) )
    
def extract_patches(img_mscn):
    scale = 2
    patches_0x, patches_1x, patches_2x, patches_3x = [], [], [], []
    row = np.shape(img_mscn)[0]
    col = np.shape(img_mscn)[1]
    size = 96
    for j in range(0,(row-size),96):
        for k in range(0,(col-size),96):
            patches_0x.append(img_mscn[j:j+size,k:k+size])
#            tmp_list = pyramid(patches_0x[-1],scale)
#            patches_1x.append(tmp_list[0])
#            patches_2x.append(tmp_list[1])
#            patches_3x.append(tmp_list[2]) 
    return (np.array(patches_0x), np.array(patches_1x), np.array(patches_2x), np.array(patches_3x))

def ema(niqe, alpha=0.1):
    weights = np.zeros(len(niqe)//6)
    for i in range(len(niqe)//6):
        weights[i] = np.exp(-alpha*i)
    weights = weights/np.sum(weights)
    
    score = np.convolve(niqe, weights, 'same')
    return(np.mean(score))

def pyramid(image, scale, res=0):
    G = image.copy()
    gpA = [G]
    for i in range(scale):
        G = cv2.pyrDown(G)
        gpA.append(G)
        
    lpA = [np.expand_dims(gpA[scale], -1)]
    for i in range(scale,0,-1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(np.expand_dims(L,-1))
        
#    return(lpA[res-1])
    return(lpA)
    
################################# authentic niqe feat #######################################################
loss_fn = contrastive_Loss_func(10)

model_0x_spa = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/test/FvsFD_frame_contrastive_0x_5000.h5')
model_0x_tmp = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/test/FvsFD_diff_contrastive_0x_5000.h5')

#model_0x_spa = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/ablation/full_authentic_FvsFD_frame_contrastive_0x_5000.h5')
#model_0x_tmp = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/ablation/full_authentic_FvsFD_diff_contrastive_0x_5000.h5')

optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
scale =16
count = 0
#data_feat = []
start = time.time()
spa_vid_score, tmp_vid_score = [], []
for i in range(len(flickr_id )//120): # vid_names  # vid_id# epfl_video_list # video_list
#    break
    video_feat_frame = []
    video_feat_framediff  = []
    
    name = flickr_id[i]
    get_video = skvideo.io.vread(konvid_directory +
                                       name + '.mp4', as_grey = True)
    videometadata = skvideo.io.ffprobe(konvid_directory +name + '.mp4')
    
#    name = video_list[i][0][0]
#    get_video = skvideo.io.vread(liveVQC_dir + 'Video/'+
#                                       name , as_grey = True) 
#    videometadata = skvideo.io.ffprobe(liveVQC_dir + 'Video/' +name)
    
#    name = str(vid_id[i]).strip()
#    get_video = skvideo.io.vread(utube_directory + 'new_ugc_videos/' +
#                                       name + '.mp4', as_grey = True)           # youtubeUGC
#    videometadata = skvideo.io.ffprobe(utube_directory + 'new_ugc_videos/' +name + '.mp4')
    
#    name = str(vid_names[i][0][0])                                              # liveQualcomm
#    get_video = skvideo.io.vread(qcomm_directory + 'Videos_mp4/'+ name.split('.')[0] + '.mp4', as_grey = True)
#    videometadata = skvideo.io.ffprobe(qcomm_directory + 'Videos_mp4/' + name.split('.')[0] + '.mp4')
##    
    frame_rate = videometadata['video']['@avg_frame_rate']
    frame_rate = int(np.round(int(frame_rate.split('/')[0])/int(frame_rate.split('/')[1])))
    frame_rate = 2
    
#    name = str(vid_names[i][0][0])                                              # liveQualcomm
#    get_video = skvideo.io.vread(qcomm_directory + 'Videos/'+ name, 1080, 1920, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
#    frame_rate = 30
    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/live_vqa/liveVideo/' +
#                                       live_video_list[i]  + '.yuv', 432, 768, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
#    frame_rate = int(live_video_list[0].split('_')[-1][:2])
    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/' +
#                                       csiq_video_list[i]  + '.yuv', 480, 832, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
    
#    frame_rate = len(get_video)//10
    
#    get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/live_mobile/LIVE_VQA_mobile/' +
#                                       str(mobile_video_list[i][0])  + '.yuv', 720, 1280, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})    
        
#    if i <72:
#        get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/' +
#                                       epfl_video_list[i] + '.yuv', 288, 352, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'}) 
#    else:
#        get_video = skvideo.io.vread('/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/' +
#                                       epfl_video_list[i] + '.yuv', 288*2, 352*2, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'}) 
#        
#    frame_rate =25
    
    spa_frame_score, tmp_frame_score = [], []
    for j in range(0, len(get_video)-1, frame_rate):
        frame = get_video[j]
        frame_diff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
#        flow = optical_flow.calc(cv2.resize(get_video[j],None, fx=1/scale, fy=1/scale) , cv2.resize(get_video[j+1],None, fx=1/scale,fy=1/scale), None)
#        dim = (np.shape(get_video[0])[0], np.shape(get_video[0])[1])
#        flow = cv2.resize(flow, dim)*scale
        
        frame_0x, frame_1x, frame_2x, frame_3x = extract_patches(frame) 
        framediff_0x, framediff_1x, framediff_2x, framediff_3x = extract_patches(frame_diff)

        frame_0x = np.expand_dims(frame_0x, 1)
        framediff_0x = np.expand_dims(framediff_0x, 1)
        
        spa_0x = (model_0x_spa.predict([frame_0x]))
        tmp_0x = (model_0x_tmp.predict([framediff_0x]))#.squeeze(0)
                    
        spa_feat = spa_0x.squeeze()#np.concatenate([spa_0x, spa_1x, spa_2x, spa_3x], axis =-1)
        tmp_feat = tmp_0x.squeeze()#np.concatenate([tmp_0x, tmp_1x, tmp_2x, tmp_3x], axis =-1)
        
#        spa_feat = pca_spa.transform(spa_feat)
#        tmp_feat = pca_tmp.transform(tmp_feat)
        
        feat = (spa_feat + tmp_feat)/2 # np.concatenate([spa_feat, tmp_feat], -1)#
        mu_dist  = np.mean(feat, 0).reshape([1,-1])
        cov_dist = np.cov(feat.T)   
        
#        spa_mu_dist  = np.mean(spa_feat, 0).reshape([1,-1])
#        spa_cov_dist = np.cov(spa_feat.T)
#    
#        tmp_mu_dist  = np.mean(tmp_feat, 0).reshape([1,-1])
#        tmp_cov_dist = np.cov(tmp_feat.T)   
        
#        spa_frame_score.append(distance(spa_mu_pris, spa_cov_pris,spa_mu_dist,spa_cov_dist))
#        tmp_frame_score.append(distance(tmp_mu_pris, tmp_cov_pris,tmp_mu_dist,tmp_cov_dist))
        spa_frame_score.append(distance(mu_pris, cov_pris, mu_dist, cov_dist))
    
    spa_vid_score.append(np.array(spa_frame_score).mean())
#    tmp_vid_score.append(np.array(tmp_frame_score).mean())
    
#    spa_vid_score.append(ema(spa_frame_score))
#    tmp_vid_score.append(ema(tmp_frame_score)) 
    
    duration = (time.time() - start)/1.0
    print(spa_vid_score[-1].mean(),duration, i)
#    print(spa_vid_score[-1], tmp_vid_score[-1],  duration,i)
 
    gc.collect()
#    break
    
x1 = np.array(spa_vid_score)
x2 = np.array(tmp_vid_score)
y = qcomm_dmos


output_dir = '/home/ece/Shankhanil/VQA/ablation/fullframe/'
#np.save(output_dir + 'konvid_spatiotemporal_0.25fr.npy', x1)
#
#output_dir = '/home/ece/Shankhanil/VQA/STEM-main/'
#mdic = {'spatial':x1, 'temporal':x2, 'niqe':z, 'dmos': vqc_dmos}
#
#savemat(output_dir + 'liveVQC_contrastive_scores_fps.mat',mdic)
    