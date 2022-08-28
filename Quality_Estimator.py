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


FvsFD_ref_frame = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/ablation/tSNE/full_authentic_FvsFD_flow_ref_feat_0x_5000.mat')['feat'].squeeze()
FvsFD_ref_diff = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/ablation/tSNE/full_authentic_FvsFD_diff_ref_feat_0x_5000.mat')['feat'].squeeze()
FDvsOF_ref_flow = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/ablation/tSNE/full_authentic_FDvsOF_flow_ref_feat_0x_5000.mat')['feat'].squeeze()
FDvsOF_ref_diff = loadmat('/media/ece/DATA/Shankhanil/VQA/contrastive_features/ablation/tSNE/full_authentic_FDvsOF_diff_ref_feat_0x_5000.mat')['feat'].squeeze()


#pca_spa = PCA(n_components=8)
#pca_tmp = PCA(n_components=8)
#syn_feat_spa = pca_spa.fit_transform(syn_feat_spa)
#syn_feat_tmp = pca_tmp.fit_transform(syn_feat_tmp)

#
#spa_mu_pris = np.mean(syn_feat_spa, 0).reshape([1,-1])
#spa_cov_pris = np.cov(syn_feat_spa.T)
#
#tmp_mu_pris = np.mean(syn_feat_tmp, 0).reshape([1,-1])
#tmp_cov_pris = np.cov(syn_feat_tmp.T)

FvsFD_feat_ref = (FvsFD_ref_frame + FvsFD_ref_diff)/2
FvsFD_mu_pris  = np.mean(FvsFD_feat_ref, 0).reshape([1,-1])
FvsFD_cov_pris = np.cov(FvsFD_feat_ref.T)  
#
FDvsOF_feat_ref = (FDvsOF_ref_flow + FDvsOF_ref_diff)/2
FDvsOF_mu_pris  = np.mean(FDvsOF_feat_ref, 0).reshape([1,-1])
FDvsOF_cov_pris = np.cov(FDvsOF_feat_ref.T)  

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
    for j in range(0,(row-size),size):
        for k in range(0,(col-size),size):
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
model_FvsFD_frame = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/ablation/full_authentic_FvsFD_frame_contrastive_0x_5000.h5')
model_FvsFD_diff = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/ablation/full_authentic_FvsFD_diff_contrastive_0x_5000.h5')
model_FDvsOF_flow = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/ablation/full_authentic_FDvsOF_flow_contrastive_0x_5000.h5')
model_FDvsOF_diff = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/ablation/full_authentic_FDvsOF_diff_contrastive_0x_5000.h5')


optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
scale = 8
count = 0
#data_feat = []
start = time.time()
F_vid_Score =[]
FvsFD_vid_score, FDvsOF_vid_score = [], []

path_to_video = ''
get_video = skvideo.io.vread(path_to_video, as_grey = True) 
videometadata = skvideo.io.ffprobe(path_to_video)
frame_rate = videometadata['video']['@avg_frame_rate']
frame_rate = int(np.round(int(frame_rate.split('/')[0])/int(frame_rate.split('/')[1])))

#FvsFD_frame_score, FDvsOF_frame_score = [], []

for j in range(0, len(get_video)-1, frame_rate):
    frame = get_video[j]
    framediff = get_video[j].astype(np.int16) - get_video[j+1].astype(np.int16)
    flow = optical_flow.calc(cv2.resize(get_video[j],None, fx=1/scale, fy=1/scale) , cv2.resize(get_video[j+1],None, fx=1/scale,fy=1/scale), None)
#        flow = cv2.resize(flow, None, fx=scale, fy=scale)*scale
    dim = (np.shape(get_video[0])[0], np.shape(get_video[0])[1])
    flow = cv2.resize(flow, dim)*scale
    
    frame, _,_,_ = extract_patches(frame) 
    framediff, _,_,_  = extract_patches(framediff)
    flow, _,_,_  = extract_patches(flow)
    
    frame = np.expand_dims(frame, 1)
    framediff = np.expand_dims(framediff, 1)
    flow = np.expand_dims(flow, 1)
    
    frame_FvsFD = (model_FvsFD_frame.predict([frame])).squeeze(1)
    diff_FvsFD = (model_FvsFD_diff.predict([framediff])).squeeze(1)
    
    flow_FDvsOF = (model_FDvsOF_flow.predict([flow])).squeeze(1)
    diff_FDvsOF = (model_FDvsOF_diff.predict([framediff])).squeeze(1)

#        spa_feat = pca_spa.transform(spa_feat)
#        tmp_feat = pca_tmp.transform(tmp_feat)
    
    FvsFD_feat = (frame_FvsFD + diff_FvsFD)/2
    
    FvsFD_mu_dist  = np.mean(FvsFD_feat, 0).reshape([1,-1])
    FvsFD_cov_dist = np.cov(FvsFD_feat.T)  
    
    FDvsOF_feat = (flow_FDvsOF + diff_FDvsOF)/2
    
    FDvsOF_mu_dist  = np.mean(FDvsOF_feat, 0).reshape([1,-1])
    FDvsOF_cov_dist = np.cov(FDvsOF_feat.T)  
    
#        FvsFD_frame_score.append(FvsFD_feat.mean(0))
#        FDvsOF_frame_score.append(FDvsOF_feat.mean(0)) 
    
    FvsFD_vid_score.appen(distance(FvsFD_mu_pris, FvsFD_cov_pris, FvsFD_mu_dist,FvsFD_cov_dist))
    FDvsOF_vid_score.append(distance(FDvsOF_mu_pris, FDvsOF_cov_pris, FDvsOF_mu_dist,FDvsOF_cov_dist))
        
    
FvsFD_vid_score = np.array(FvsFD_vid_score).mean(0)
FDvsOF_vid_score = np.array(FDvsOF_vid_score).mean(0)

VISION = FvsFD_vid_score*FDvsOF_vid_score
