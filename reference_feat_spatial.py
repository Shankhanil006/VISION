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
import keras
from keras.optimizers import Adam, Nadam
from keras.models import load_model

from keras import regularizers
import h5py
from keras import losses

from scipy.io import loadmat
import skvideo.io

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
from scipy.io import loadmat,savemat
from sharp_patch_generator import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gc.collect()
K.clear_session()

strred_directory = '/media/ece/DATA/Shankhanil/VQA/strred_files/'
##################################  LIVE VQA  #####################################

live_directory = '/media/ece/DATA/Shankhanil/VQA/live_vqa/'
with open(live_directory + 'live_video_quality_seqs.txt') as f:
    video_names = f.readlines()
live_video_list = [x.strip('.yuv\n') for x in video_names] 

with open(live_directory + 'live_video_quality_data.txt') as f:
    video_dmos = f.readlines()
live_dmos_list = [float(x.split('\t')[0]) for x in video_dmos] 
live_dmos = np.array(live_dmos_list)

#feat = np.zeros([len(video_list),2048*2])

live_seq = np.array(['pa', 'rb', 'rh', 'tr', 'st', 'sf', 'bs', 'sh', 'mc', 'pr'])
rate = [25,25,25,25,25,25,25,50,50,50]

seq_id = {0:'pa', 1:'rb', 2:'rh', 3:'tr', 4:'st', 5:'sf', 6:'bs', 7:'sh',  8:'mc', 9:'pr'}

########################## EPFL-POLIMI CIF #####################################

epfl_cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/'
with open(epfl_cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_cif_dmos = 5-np.array(epfl_cif_dmos_list)

epfl_cif_seq = np.array(['foreman','hall','mobile','mother','news','paris'])

########################## EPFL-POLIMI 4CIF #####################################

epfl_4cif_directory = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/'
with open(epfl_4cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_4cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_4cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_4cif_dmos = 5-np.array(epfl_4cif_dmos_list)

epfl_4cif_seq = np.array(['CROWDRUN','DUCKSTAKEOFF','HARBOUR','ICE','PARKJOY','SOCCER'])

############################### Mobile LIVE ###################################
mobile_directory = '/media/ece/DATA/Shankhanil/VQA/live_mobile/'
mobile_dmos = sio.loadmat(strred_directory + 'strred_mobile.mat')['dmos'].squeeze()

mobile_video_list = sio.loadmat(strred_directory + 'strred_mobile.mat')['names'].squeeze()
#mobile_seq_id = {0:'bf', 1:'dv', 2:'fc', 3:'hc', 4:'la', 5:'po', 6:'rb', 7:'sd', 8:'ss', 9:'tk'}
mobile_seq = np.array(['bf', 'dv', 'fc', 'hc', 'la', 'po', 'rb', 'sd', 'ss', 'tk'])
mobile_dist = np.array(['r1','r2','r3','r4','s14','s24','s34','t14','t124','t421','t134','t431','w1','w2','w3','w4'])

###############################  CSIQ  #############################################
csiq_directory = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
with open(csiq_directory + 'video_subj_ratings.txt') as f:
    file_names = f.readlines()
csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
csiq_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
csiq_dmos = np.array(csiq_dmos_list)

csiq_seq = np.array(['BQMall', 'BQTerrace','BasketballDrive','Cactus', \
            'Carving','Chipmunks','Flowervase','Keiba','Kimono', \
            'ParkScene','PartyScene','Timelapse'])

################## ECVQ ###########################################################
ecvq_directory = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/'
with open(ecvq_directory + 'subjective_scores_cif.txt') as f:
    file_names = f.readlines()
ecvq_video_list = [x.split()[0] for x in file_names] 
ecvq_dmos_list  = [float(x.split()[2]) for x in file_names]
ecvq_dmos = np.array(ecvq_dmos_list)

ecvq_seq = np.array(['container_ship', 'flower_garden', 'football', 'foreman', 'hall', 'mobile', 'news', 'silent'])

################## EVVQ ###########################################################
evvq_directory = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/'
with open(evvq_directory + 'subjective_scores_vga.txt') as f:
    file_names = f.readlines()
evvq_video_list = [x.split()[0] for x in file_names] 
evvq_dmos_list  = [float(x.split()[2]) for x in file_names]
evvq_dmos = np.array(evvq_dmos_list)

evvq_seq = np.array(['cartoon', 'cheerleaders', 'discussion', 'flower_garden', 'football', 'mobile', 'town_plan', 'weather'])

live_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/live_vqa/liveVideo/reference/'
epfl_cif_vid_dir  = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/reference/'
epfl_4cif_vid_dir = '/media/ece/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/reference/'
mobile_vid_dir    = '/media/ece/DATA/Shankhanil/VQA/live_mobile/LIVE_VQA_mobile/'
csiq_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
ecvq_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/ECVQ/cif_videos/reference/'
evvq_vid_dir      = '/media/ece/DATA/Shankhanil/VQA/EVVQ/vga_videos/reference/'

from natsort import natsort_keygen,natsorted, ns
#natsort_key = natsort_keygen(alg=ns.IGNORECASE)

live_filenames = natsorted([file for file in glob.glob(live_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

epfl_cif_filenames = natsorted([file  for file in glob.glob(epfl_cif_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

epfl_4cif_filenames = natsorted([file  for file in glob.glob(epfl_4cif_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

mobile_filenames = natsorted([file for file in glob.glob(mobile_vid_dir +'*org.yuv')], alg=ns.IGNORECASE)

csiq_filenames = natsorted([file for file in glob.glob(csiq_vid_dir +'*ref.yuv')], alg=ns.IGNORECASE)

ecvq_filenames = natsorted([file for file in glob.glob(ecvq_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

evvq_filenames = natsorted([file for file in glob.glob(evvq_vid_dir +'*.yuv')], alg=ns.IGNORECASE)

###################################3 Network ###########################################

def random_crop(img, random_crop_size = (112,112)):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 1
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
#    return img[y:(y+dy), x:(x+dx), :]
    return (x,y)

def image_colorfulness(image):
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
    
#############################################################
databases = {  'live'      :{'num':150, 'ref':10, 'dist': 15, 'height': 432, 'width' : 768, 'seq' : live_seq,   'files' : live_filenames,      'dmos' : live_dmos}, \
               'mobile'    :{'num':160, 'ref':10, 'dist': 16, 'height':720, 'width' :1280,  'seq' : mobile_seq,   'files' : mobile_filenames,    'dmos' : mobile_dmos}, \
               'epfl_cif'  :{'num':72,  'ref':6,  'dist': 12, 'height':288, 'width' :352, 'seq' : epfl_cif_seq,   'files' : epfl_cif_filenames,  'dmos' : epfl_cif_dmos},   \
               'epfl_4cif' :{'num':72,  'ref':6,  'dist': 12, 'height':576, 'width' : 704, 'seq' : epfl_4cif_seq,  'files' : epfl_4cif_filenames,  'dmos' : epfl_4cif_dmos},   \
               'csiq'      :{'num':216, 'ref':12, 'dist': 18, 'height':480, 'width' : 832, 'seq' : csiq_seq,       'files' : csiq_filenames,  'dmos' : csiq_dmos}, \
               'ecvq'      :{'num':90,  'ref':8,  'dist': 10, 'height':288, 'width' :  352,'seq' : ecvq_seq,       'files' : ecvq_filenames,  'dmos' : ecvq_dmos},   \
               'evvq'      :{'num':90,  'ref':8,  'dist': 9, 'height':480, 'width' : 640, 'seq' : evvq_seq,       'files' : evvq_filenames,  'dmos' : evvq_dmos} }

ecvq_num_seq = [12, 11, 10, 11, 12, 10, 12, 12]
evvq_num_seq = [12, 9, 12, 11, 11, 11, 12, 12]

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
vqc_dmos = np.array([100-float(i) for i in vqc_mos]  )
#names = [file for file in glob.glob(directory +'/*.yuv')]

utube_directory = '/media/ece/DATA/Shankhanil/VQA/youtube_ugc/'
vid_id = loadmat(utube_directory + 'filename.mat')['name']
utube_mos   = loadmat(utube_directory + 'filename.mat')['mos'].squeeze()
utube_dmos = [5-float(i) for i in utube_mos]
utube_dmos = np.array(utube_dmos)
########################################################################################

#batch_size = 4
##nframes = 16
#dist_seq = 8
#epoch = 0
#patch_size = 112
scale =2

all_train_dbs = ['live','mobile',  'epfl_4cif', 'epfl_cif', 'csiq', 'ecvq', 'evvq']#
#train_db = 'live'

output_dir = '/media/ece/DATA/Shankhanil/VQA/contrastive_features/ablation/cmc/'

loss_fn = contrastive_Loss_func(0)
#base_model= get_model()
#model = multi_gpu_model(base_model, gpus=2)

model_0x = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/ablation/FvsOF_frame_cmc_0x_3000.h5', custom_objects={'loss': loss_fn})
#model_1x = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/acm/spatial_contrastive_1x_10000.h5', custom_objects={'loss': loss_fn})
#model_2x = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/acm/spatial_contrastive_2x_10000.h5', custom_objects={'loss': loss_fn})
#model_3x = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/acm/spatial_contrastive_3x_10000.h5', custom_objects={'loss': loss_fn})

patches_feat, sigma_list = [], []
start = time.time()
for train_db in all_train_dbs:
#    for ref_id in range(databases[train_db]['ref']):
    filenames = databases[train_db]['files']
    height, width = databases[train_db]['height'], databases[train_db]['width']
    for name in filenames:
            get_video = skvideo.io.vread(name,height, width, as_grey = True, inputdict={'-pix_fmt': 'yuvj420p'})
            
            for j in range(0, len(get_video)-1, 2):
                    frame11 = get_video[j]
#                    frame12 = get_video[j + 1]
#                    frame_diff = frame11.astype(np.int16) - frame12.astype(np.int16)    
                    anchor_0x, sigma = sharp_patch_generator(frame11,0.85).patch_generator()
#                    anchor_1x,anchor_2x,anchor_3x = [], [], []
#                    for p in range(len(anchor_0x)):
#                        tmp_list = pyramid(anchor_0x[p],scale)
#                        anchor_1x.append(tmp_list[0])
#                        anchor_2x.append(tmp_list[1])
#                        anchor_3x.append(tmp_list[2])                        
                    
                        
                    anchor_0x = np.expand_dims(anchor_0x, 1)   
#                    anchor_1x = np.expand_dims(anchor_1x, 1)
#                    anchor_2x = np.expand_dims(anchor_2x, 1)
#                    anchor_3x = np.expand_dims(anchor_3x, 1)
                    
#                    patches_feat.append((model.predict([anchor_1x,anchor_2x,anchor_4x])).squeeze(0))
                    tmp_0x = (model_0x.predict([anchor_0x]))#.squeeze(0)
#                    tmp_1x = (model_1x.predict([anchor_1x]))#.squeeze(0)
#                    tmp_2x = (model_2x.predict([anchor_2x]))#.squeeze(0)
#                    tmp_3x = (model_3x.predict([anchor_3x]))#.squeeze(0)
                    
                    patches_feat.append(tmp_0x)
#                    patches_feat.append(np.concatenate([tmp_0x,tmp_1x,tmp_2x,tmp_3x], -1))
                    sigma_list.append(sigma)    
            
#            break
    end = time.time()
    duration = (end - start)/60
    print(train_db, duration)
#    break
patches_feat = np.vstack(patches_feat)
for i in range(len(sigma_list)):
    sigma_list[i] = sigma_list[i].reshape([-1,1])
sigma_list = np.vstack(sigma_list)
mdic = {'feat':patches_feat, 'sigma':sigma_list}

savemat(output_dir + 'cmc_FvsOF_frame_ref_feat_0x_3000.mat',mdic)

