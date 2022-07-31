import tensorflow as tf
#tf.enable_eager_execution()
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

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
gc.collect()
K.clear_session()

path = '/media/ece/'
strred_directory = path + '/DATA/Shankhanil/VQA/strred_files/'
##################################  LIVE VQA  #####################################

live_directory = path + '/DATA/Shankhanil/VQA/live_vqa/'
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

epfl_cif_directory = path + '/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/CIF/'
with open(epfl_cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_cif_dmos = 5-np.array(epfl_cif_dmos_list)

epfl_cif_seq = np.array(['foreman','hall','mobile','mother','news','paris'])

########################## EPFL-POLIMI 4CIF #####################################

epfl_4cif_directory = path + '/DATA/Shankhanil/VQA/epfl_vqeg_video/online_DB/decoded/4CIF/'
with open(epfl_4cif_directory + 'names_scores.txt') as f:
    file_names = f.readlines()
epfl_4cif_video_list = [x.split('\t')[0] for x in file_names] 
epfl_4cif_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
epfl_4cif_dmos = 5-np.array(epfl_4cif_dmos_list)

epfl_4cif_seq = np.array(['CROWDRUN','DUCKSTAKEOFF','HARBOUR','ICE','PARKJOY','SOCCER'])

############################### Mobile LIVE ###################################
mobile_directory = path + '/DATA/Shankhanil/VQA/live_mobile/'
mobile_dmos = sio.loadmat(strred_directory + 'strred_mobile.mat')['dmos'].squeeze()

mobile_video_list = sio.loadmat(strred_directory + 'strred_mobile.mat')['names'].squeeze()
#mobile_seq_id = {0:'bf', 1:'dv', 2:'fc', 3:'hc', 4:'la', 5:'po', 6:'rb', 7:'sd', 8:'ss', 9:'tk'}
mobile_seq = np.array(['bf', 'dv', 'fc', 'hc', 'la', 'po', 'rb', 'sd', 'ss', 'tk'])
mobile_dist = np.array(['r1','r2','r3','r4','s14','s24','s34','t14','t124','t421','t134','t431','w1','w2','w3','w4'])

###############################  CSIQ  #############################################
csiq_directory = path + '/DATA/Shankhanil/VQA/CSIQ/csiq_videos/'
with open(csiq_directory + 'video_subj_ratings.txt') as f:
    file_names = f.readlines()
csiq_video_list = [x.split('\t')[0].split('.')[0] for x in file_names] 
csiq_dmos_list  = [float(x.split('\t')[1]) for x in file_names]
csiq_dmos = np.array(csiq_dmos_list)

csiq_seq = np.array(['BQMall', 'BQTerrace','BasketballDrive','Cactus', \
            'Carving','Chipmunks','flowervase','Keiba','Kimono', \
            'ParkScene','PartyScene','Timelapse'])

################## ECVQ ###########################################################
ecvq_directory = path + '/DATA/Shankhanil/VQA/ECVQ/cif_videos/'
with open(ecvq_directory + 'subjective_scores_cif.txt') as f:
    file_names = f.readlines()
ecvq_video_list = [x.split()[0] for x in file_names] 
ecvq_dmos_list  = [float(x.split()[2]) for x in file_names]
ecvq_dmos = np.array(ecvq_dmos_list)

ecvq_seq = np.array(['container_ship', 'flower_garden', 'football', 'foreman', 'hall', 'mobile', 'news', 'silent'])

################## EVVQ ###########################################################
evvq_directory = path + '/DATA/Shankhanil/VQA/EVVQ/vga_videos/'
with open(evvq_directory + 'subjective_scores_vga.txt') as f:
    file_names = f.readlines()
evvq_video_list = [x.split()[0] for x in file_names] 
evvq_dmos_list  = [float(x.split()[2]) for x in file_names]
evvq_dmos = np.array(evvq_dmos_list)

evvq_seq = np.array(['cartoon', 'cheerleaders', 'discussion', 'flower_garden', 'football', 'mobile', 'town_plan', 'weather'])

view1_live_frame_dir      = path + '/vip_vol2/live_vqa/live_vqa_frames_gray/reference/'
view1_epfl_cif_frame_dir  = path + '/vip_vol2/epfl_vqeg_video/epfl_cif_frames_gray/reference/'
view1_epfl_4cif_frame_dir = path + '/vip_vol2/epfl_vqeg_video/epfl_4cif_frames_gray/reference/'
view1_mobile_frame_dir    = path + '/vip_vol2/live_mobile/live_mobile_frames_gray/reference/'
view1_csiq_frame_dir      = path + '/vip_vol2/CSIQ/csiq_frames_gray/reference/'
view1_ecvq_frame_dir      = path + '/vip_vol2/ECVQ/ecvq_frames_gray/reference/'
view1_evvq_frame_dir      = path + '/vip_vol2/EVVQ/evvq_frames_gray/reference/'

view2_live_frame_dir      = path + '/vip_vol2/live_vqa/live_vqa_frames_diff_gray/reference/'
view2_epfl_cif_frame_dir  = path + '/vip_vol2/epfl_vqeg_video/epfl_cif_frames_diff_gray/reference/'
view2_epfl_4cif_frame_dir = path + '/vip_vol2/epfl_vqeg_video/epfl_4cif_frames_diff_gray/reference/'
view2_mobile_frame_dir    = path + '/vip_vol2/live_mobile/live_mobile_frames_diff_gray/reference/'
view2_csiq_frame_dir      = path + '/vip_vol2/CSIQ/csiq_frames_diff_gray/reference/'
view2_ecvq_frame_dir      = path + '/vip_vol2/ECVQ/ecvq_frames_diff_gray/reference/'
view2_evvq_frame_dir      = path + '/vip_vol2/EVVQ/evvq_frames_diff_gray/reference/'

#view2_live_frame_dir      = path + '/vip_vol2/live_vqa/live_vqa_optical_flow/reference/'
#view2_epfl_cif_frame_dir  = path + '/vip_vol2/epfl_vqeg_video/epfl_cif_optical_flow/reference/'
#view2_epfl_4cif_frame_dir = path + '/vip_vol2/epfl_vqeg_video/epfl_4cif_optical_flow/reference/'
#view2_mobile_frame_dir    = path + '/vip_vol2/live_mobile/live_mobile_optical_flow/reference/'
#view2_csiq_frame_dir      = path + '/vip_vol2/CSIQ/csiq_optical_flow/reference/'
#view2_ecvq_frame_dir      = path + '/vip_vol2/ECVQ/ecvq_optical_flow/reference/'
#view2_evvq_frame_dir      = path + '/vip_vol2/EVVQ/evvq_optical_flow/reference/'

from natsort import natsort_keygen,natsorted, ns
natsort_key = natsort_keygen(alg=ns.IGNORECASE)

view1_live_filenames, view2_live_filenames = [], []
for ref in range(10):
    view1_seq_path = view1_live_frame_dir + live_seq[ref] + '/'
    view1_live_filenames.append(natsorted([file for file in glob.glob(view1_seq_path +'*.npy')], alg=ns.IGNORECASE))

    view2_seq_path = view2_live_frame_dir + live_seq[ref] + '/'
    view2_live_filenames.append(natsorted([file for file in glob.glob(view2_seq_path +'*.npy')], alg=ns.IGNORECASE))

view1_epfl_cif_filenames, view2_epfl_cif_filenames = [], []
for ref in range(6):
    view1_seq_path = view1_epfl_cif_frame_dir + epfl_cif_seq[ref] + '/'
    view1_epfl_cif_filenames.append(natsorted([file  for file in glob.glob(view1_seq_path +'*.npy')], alg=ns.IGNORECASE))

    view2_seq_path = view2_epfl_cif_frame_dir + epfl_cif_seq[ref] + '/'
    view2_epfl_cif_filenames.append(natsorted([file  for file in glob.glob(view2_seq_path +'*.npy')], alg=ns.IGNORECASE))
    
view1_epfl_4cif_filenames, view2_epfl_4cif_filenames = [], []
for ref in range(6):
    view1_seq_path = view1_epfl_4cif_frame_dir + epfl_4cif_seq[ref] + '/'
    view1_epfl_4cif_filenames.append(natsorted([file  for file in glob.glob(view1_seq_path +'*.npy')], alg=ns.IGNORECASE))

    view2_seq_path = view2_epfl_4cif_frame_dir + epfl_4cif_seq[ref] + '/'
    view2_epfl_4cif_filenames.append(natsorted([file  for file in glob.glob(view2_seq_path +'*.npy')], alg=ns.IGNORECASE))
    
view1_mobile_filenames, view2_mobile_filenames = [], []
for ref in range(10):
    view1_seq_path = view1_mobile_frame_dir + mobile_seq[ref] + '/'
    view1_mobile_filenames.append(natsorted([file for file in glob.glob(view1_seq_path +'*.npy')], alg=ns.IGNORECASE))

    view2_seq_path = view2_mobile_frame_dir + mobile_seq[ref] + '/'
    view2_mobile_filenames.append(natsorted([file for file in glob.glob(view2_seq_path +'*.npy')], alg=ns.IGNORECASE))
    
view1_csiq_filenames, view2_csiq_filenames = [], []
for ref in range(12):
    view1_seq_path = view1_csiq_frame_dir + csiq_seq[ref] + '/'
    view1_csiq_filenames.append(natsorted([file for file in glob.glob(view1_seq_path +'*.npy')], alg=ns.IGNORECASE))

    view2_seq_path = view2_csiq_frame_dir + csiq_seq[ref] + '/'
    view2_csiq_filenames.append(natsorted([file for file in glob.glob(view2_seq_path +'*.npy')], alg=ns.IGNORECASE))
    
view1_ecvq_filenames, view2_ecvq_filenames = [], []
for ref in range(8):
    view1_seq_path = view1_ecvq_frame_dir + ecvq_seq[ref] + '/'
    view1_ecvq_filenames.append(natsorted([file for file in glob.glob(view1_seq_path +'*.npy')], alg=ns.IGNORECASE))

    view2_seq_path = view2_ecvq_frame_dir + ecvq_seq[ref] + '/'
    view2_ecvq_filenames.append(natsorted([file for file in glob.glob(view2_seq_path +'*.npy')], alg=ns.IGNORECASE))
    
view1_evvq_filenames, view2_evvq_filenames = [], []
for ref in range(8):
    view1_seq_path = view1_evvq_frame_dir + evvq_seq[ref] + '/'
    view1_evvq_filenames.append(natsorted([file for file in glob.glob(view1_seq_path +'*.npy')], alg=ns.IGNORECASE))

    view2_seq_path = view2_evvq_frame_dir + evvq_seq[ref] + '/'
    view2_evvq_filenames.append(natsorted([file for file in glob.glob(view2_seq_path +'*.npy')], alg=ns.IGNORECASE))

######################################### Authentic #######################################

curr_dir = path + '/vip_vol1/LSVD_frame_curr/'
next_dir = path + '/vip_vol1/LSVD_frame_next/'
batch_seq = []
for i in range(1,26):
    batch_seq.append('yfcc-batch' + str(i))
#from natsort import natsort_keygen,natsorted, ns
#natsort_key = natsort_keygen(alg=ns.IGNORECASE)

curr_filenames, next_filenames = [], []
for ref in range(len(batch_seq)):
    curr_path = curr_dir + batch_seq[ref] + '/' + '0/'
    curr_filenames.append(natsorted([file for file in glob.glob(curr_path +'*.npy')], alg=ns.IGNORECASE))
    next_path  = next_dir + batch_seq[ref] + '/' + '0/'
    next_filenames.append(natsorted([file for file in glob.glob(next_path +'*.npy')], alg=ns.IGNORECASE))
    
    
###################################3 Network ###########################################
class Models(object):
    def __init__(self):
        self.view1 = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/test/FvsFD_frame_contrastive_0x_7000.h5')
        self.view2 = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/test/FvsFD_diff_contrastive_0x_7000.h5')
        self.wild_model = None
        
    def view1_network(self):
        if self.view1:
            return self.view1
        #    inputs = keras.Input((depth, width, height, 1))
        self.view1 = Sequential()
        self.view1.add(TimeDistributed(layers.Conv2D(filters=32, kernel_size=3,bias_initializer='random_normal', padding = 'same',  activation="relu")))
        self.view1.add(TimeDistributed(layers.Conv2D(filters=32, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view1.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view1.add(TimeDistributed(layers.BatchNormalization()))
        
        self.view1.add(TimeDistributed(layers.Conv2D(filters=64, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view1.add(TimeDistributed(layers.Conv2D(filters=64, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view1.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view1.add(TimeDistributed(layers.BatchNormalization()))
    
        self.view1.add(TimeDistributed(layers.Conv2D(filters=128, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view1.add(TimeDistributed(layers.Conv2D(filters=128, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view1.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view1.add(TimeDistributed(layers.BatchNormalization()))
    
        self.view1.add(TimeDistributed(layers.Conv2D(filters=256, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view1.add(TimeDistributed(layers.Conv2D(filters=256, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view1.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view1.add(TimeDistributed(layers.BatchNormalization()))
    
        self.view1.add(TimeDistributed(layers.GlobalAveragePooling2D()))
        
        
    #    base_model = VGG19(include_top=False, weights=None, input_shape=(None,None,1), pooling='avg')
    #    model.add(TimeDistributed(base_model))
        return self.view1
    
    def view2_network(self):
        if self.view2:
            return self.view2
        #    inputs = keras.Input((depth, width, height, 1))
        self.view2 = Sequential()
        self.view2.add(TimeDistributed(layers.Conv2D(filters=32, kernel_size=3,bias_initializer='random_normal', padding = 'same',  activation="relu")))
        self.view2.add(TimeDistributed(layers.Conv2D(filters=32, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view2.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view2.add(TimeDistributed(layers.BatchNormalization()))
        
        self.view2.add(TimeDistributed(layers.Conv2D(filters=64, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view2.add(TimeDistributed(layers.Conv2D(filters=64, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view2.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view2.add(TimeDistributed(layers.BatchNormalization()))
    
        self.view2.add(TimeDistributed(layers.Conv2D(filters=128, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view2.add(TimeDistributed(layers.Conv2D(filters=128, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view2.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view2.add(TimeDistributed(layers.BatchNormalization()))
    
        self.view2.add(TimeDistributed(layers.Conv2D(filters=256, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view2.add(TimeDistributed(layers.Conv2D(filters=256, kernel_size=3,bias_initializer='random_normal', padding = 'same',activation="relu")))
        self.view2.add(TimeDistributed(layers.MaxPool2D(pool_size=(2,2))))
        self.view2.add(TimeDistributed(layers.BatchNormalization()))
    
        self.view2.add(TimeDistributed(layers.GlobalAveragePooling2D()))
        
        
    #    base_model = VGG19(include_top=False, weights=None, input_shape=(None,None,1), pooling='avg')
    #    model.add(TimeDistributed(base_model))
        return self.view2
    
    def get_model(self):
        if self.wild_model:
            return self.wild_model
        view1_input_shape = (None, None, None, 1)
        view2_input_shape = (None, None, None, 1)
        
        view1_inputs = Input(shape=view1_input_shape)
        view2_inputs = Input(shape=view2_input_shape)
        
        view1_model = self.view1_network()
        view2_model = self.view2_network()
        
        view1_outputs = view1_model(view1_inputs)  
        view2_outputs = view2_model(view2_inputs)
        
        outputs = layers.concatenate([view1_outputs, view2_outputs], axis =1)
        self.wild_model = Model(inputs = [view1_inputs, view2_inputs], outputs = outputs)
    #    wild_model.compile(Adam(1e-4), loss = loss_fn)
        return self.wild_model    


def random_crop(img, random_crop_size = (112,112)):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 1
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
#    return img[y:(y+dy), x:(x+dx), :]
    return (x,y)

def center_crop(img, dim = (224,224)):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	height, width = img.shape[0], img.shape[1]

	# process crop width and height for max available dimension
	crop_width = dim[1] #if dim[1]<img.shape[1] else img.shape[1]
	crop_height = dim[0] #if dim[0]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
#	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return (mid_x - cw2, mid_y - ch2)

def scaling(img, flip , factor = 224):
    height, width = img.shape[0], img.shape[1]
    
    scale = max(height//224 +1, width//224 + 1)    
    scaled_img = cv2.resize(img,None, fx = 1/scale, fy = 1/scale)
    if flip ==1:
        scaled_img = cv2.flip(scaled_img,0)
    return(np.expand_dims(scaled_img,-1))
    
def pyramid(image, scale, res):
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
        
    return(lpA[res-1])
 
def cuda_opticalflow(prev_frame, next_frame):
    gpu_prev = cv2.cuda_GpuMat()
    gpu_prev.upload(prev_frame)

    gpu_next = cv2.cuda_GpuMat()
    gpu_next.upload(next_frame)
    
    gpu_flow = cv2.cuda_Opticalview2Dual_TVL1.create()
    gpu_flow = cv2.cuda_Opticalview2Dual_TVL1.calc(gpu_flow, gpu_prev, gpu_next, None)
    flow = gpu_flow.download()
    return(flow)   
    
#############################################################
databases = {  'live'      :{'num':150, 'ref':10, 'dist': 15, 'height': 432, 'width' : 768, 'seq' : live_seq,   'view1_files' : view1_live_filenames, 'view2_files' : view2_live_filenames,      'dmos' : live_dmos}, \
               'mobile'    :{'num':160, 'ref':10, 'dist': 16, 'height':720, 'width' :1280,  'seq' : mobile_seq,   'view1_files' : view1_mobile_filenames, 'view2_files' : view2_mobile_filenames,    'dmos' : mobile_dmos}, \
               'epfl_cif'  :{'num':72,  'ref':6,  'dist': 12, 'height':288, 'width' :352, 'seq' : epfl_cif_seq,   'view1_files' : view1_epfl_cif_filenames, 'view2_files' : view2_epfl_cif_filenames,  'dmos' : epfl_cif_dmos},   \
               'epfl_4cif' :{'num':72,  'ref':6,  'dist': 12, 'height':576, 'width' : 704, 'seq' : epfl_4cif_seq,  'view1_files' : view1_epfl_4cif_filenames, 'view2_files' : view2_epfl_4cif_filenames,  'dmos' : epfl_4cif_dmos},   \
               'csiq'      :{'num':216, 'ref':12, 'dist': 18, 'height':480, 'width' : 832, 'seq' : csiq_seq,       'view1_files' : view1_csiq_filenames, 'view2_files' : view2_csiq_filenames,  'dmos' : csiq_dmos}, \
               'ecvq'      :{'num':90,  'ref':8,  'dist': 10, 'height':288, 'width' :  352,'seq' : ecvq_seq,       'view1_files' : view1_ecvq_filenames, 'view2_files' : view2_ecvq_filenames,   'dmos' : ecvq_dmos},   \
               'evvq'      :{'num':90,  'ref':8,  'dist': 9, 'height':480, 'width' : 640, 'seq' : evvq_seq,       'view1_files' : view1_evvq_filenames, 'view2_files' : view2_evvq_filenames,  'dmos' : evvq_dmos} }

ecvq_num_seq = [12, 11, 10, 11, 12, 10, 12, 12]
evvq_num_seq = [12, 9, 12, 11, 11, 11, 12, 12]

batch_size = 4*2
dist_seq =11
patch_size = 224

scale = 2
#res = 3
spacing = 5

resolutions = [1, 2, 3]

all_train_dbs = ['live','mobile',  'epfl_4cif', 'epfl_cif', 'csiq', 'ecvq', 'evvq']
#resolutions = [1]
optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()  

#for res in resolutions:
 
flow_scale = 0.5
loss_fn = contrastive_Loss_func(dist_seq)
models = Models()
view1_model = models.view1_network()
view2_model = models.view2_network()
base_model= models.get_model()
##base_model = load_model('/media/ece/DATA/Shankhanil/VQA/tmp/contrastive_models/new_study/spatial_contrastive_1x_15000.h5', custom_objects={'loss': loss_fn})
model = multi_gpu_model(base_model, gpus=2)
##model.layers[-2].set_weights(base_model.get_weights())
model.compile(Adam(1e-5), loss_fn)
loss = []    

start_time = time.time()   
epoch = 7001
while(1):
    view1_train, view2_train, y_train = [], [], []
    
    if epoch%2==0 and epoch<5000:
        for _ in range(batch_size):
    #    train_db = all_train_dbs[np.mod(epoch, len(all_train_dbs))]
    #    ref_id = np.random.choice(databases[train_db]['ref'], batch_size, True)
            train_db = all_train_dbs[np.random.randint(len(all_train_dbs))]
            ref = np.random.randint(databases[train_db]['ref'])
                
            if train_db == 'ecvq' and ecvq_num_seq[ref]< 11:
                    continue
            elif train_db == 'evvq' and evvq_num_seq[ref]< 11:
                    continue
                
            view1_filenames = databases[train_db]['view1_files']    
            view2_filenames = databases[train_db]['view2_files'] 
    #    for ref in ref_id:
            crop_view1 = []
            crop_view2 = []
        
            view1_names = view1_filenames[ref] 
            view2_names = view2_filenames[ref]
            frame_id = np.random.randint(0,len(view1_names)-10,2) 
            
            flip1, flip2 = 0,0# np.random.randint(0,2,2)             # seed for random flip
            
            view1_ref_name = view1_names[frame_id[0]]
            view1_img =  np.load(view1_ref_name) 
    
            view2_ref_name = view2_names[frame_id[0]]
            view2_img = np.load(view2_ref_name)            
            
            crop_x1,crop_y1 = center_crop(view1_img, (patch_size,patch_size))
            crop_x2,crop_y2 = crop_x1,crop_y1#random_crop(view2_img, (patch_size,patch_size))
            
            view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :]
            view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
            
    #            prev_crop = pyramid(prev_crop, scale, res)
    #            next_crop = pyramid(next_crop, scale, res)
            
    #        ref_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
            crop_view1.append(view1_crop)
            
            
    #        ref_view2 = optical_flow.calc(prev_crop, next_crop, None)
            crop_view2.append(view2_crop)
            
            ref_name, view1_id = view1_ref_name.split('/')[-1].rsplit('_',1) 
            ref_name, view2_id = view2_ref_name.split('/')[-1].rsplit('_',1)
            
    #        break
            if train_db == 'live':
                live_dist_seq = np.random.choice(15, dist_seq, False)
                for seq in live_dist_seq:               
                     
                    view1_dist_name = ref_name +  str(seq+2) + '_' + view1_id
                    view2_dist_name = ref_name +  str(seq+2) + '_' + view2_id
                    
                    view1_img = np.load(view1_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view1_dist_name)
                    view2_img = np.load(view2_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view2_dist_name)
                    
                    view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                    view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #                    prev_crop = pyramid(prev_crop, scale, res)
    #                    next_crop = pyramid(next_crop, scale, res)
                    
    #                dist_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
    #                dist_view2 = optical_flow.calc(prev_crop, next_crop, None)
    
                    
                    crop_view1.append(view1_crop)
                    crop_view2.append(view2_crop)
                    
            elif train_db == 'mobile':         
                mobile_dist_seq = np.random.choice(16, dist_seq, False)
                for seq in mobile_dist_seq:
                    
                    view1_dist_name = ref_name + '_' + mobile_dist[seq] + '_' + view1_id
                    view2_dist_name = ref_name + '_' + mobile_dist[seq] + '_' + view2_id
                    
                    view1_img = np.load(view1_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view1_dist_name)
                    view2_img = np.load(view2_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view2_dist_name)
                    
                    view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                    view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #                    prev_crop = pyramid(prev_crop, scale, res)
    #                    next_crop = pyramid(next_crop, scale, res)
                    
    #                dist_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
    #                dist_view2 = optical_flow.calc(prev_crop, next_crop, None)
    
                    
                    crop_view1.append(view1_crop)
                    crop_view2.append(view2_crop)       
                    
            elif train_db == 'epfl_4cif':
                epfl_4cif_dist_seq = np.random.choice(12, dist_seq, False)
                for seq in epfl_4cif_dist_seq:               
                    view1_dist_name = ref_name + '_' + str(seq) + '_' + view1_id
                    view2_dist_name = ref_name + '_' + str(seq) + '_' + view2_id
                    
                    view1_img = np.load(view1_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view1_dist_name)
                    view2_img = np.load(view2_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view2_dist_name)
                    
                    view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                    view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #                    prev_crop = pyramid(prev_crop, scale, res)
    #                    next_crop = pyramid(next_crop, scale, res)
                    
    #                dist_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
    #                dist_view2 = optical_flow.calc(prev_crop, next_crop, None)
    
                    
                    crop_view1.append(view1_crop)
                    crop_view2.append(view2_crop)
                    
            elif train_db == 'epfl_cif':
                epfl_cif_dist_seq = np.random.choice(12, dist_seq, False)
                for seq in epfl_cif_dist_seq:               
                    view1_dist_name = ref_name + '_' + str(seq) + '_' + view1_id
                    view2_dist_name = ref_name + '_' + str(seq) + '_' + view2_id
                    
                    view1_img = np.load(view1_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view1_dist_name)
                    view2_img = np.load(view2_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view2_dist_name)
                    
                    view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                    view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #                    prev_crop = pyramid(prev_crop, scale, res)
    #                    next_crop = pyramid(next_crop, scale, res)
                    
    #                dist_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
    #                dist_view2 = optical_flow.calc(prev_crop, next_crop, None)
    
                    
                    crop_view1.append(view1_crop)
                    crop_view2.append(view2_crop)
                    
            elif train_db == 'csiq':
                csiq_dist_seq = np.random.choice(18, dist_seq, False)
                for seq in csiq_dist_seq:
                    view1_dist_name = ref_name + '_' + str(seq+1).zfill(2) + '_' + view1_id
                    view2_dist_name = ref_name + '_' + str(seq+1).zfill(2) + '_' + view2_id
                    
                    view1_img = np.load(view1_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view1_dist_name)
                    view2_img = np.load(view2_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view2_dist_name)
                    
                    view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                    view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #                    prev_crop = pyramid(prev_crop, scale, res)
    #                    next_crop = pyramid(next_crop, scale, res)
                    
    #                dist_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
    #                dist_view2 = optical_flow.calc(prev_crop, next_crop, None)
    
                    
                    crop_view1.append(view1_crop)
                    crop_view2.append(view2_crop)
                    
            elif train_db == 'ecvq':
                ecvq_dist_seq = np.random.choice(ecvq_num_seq[ref], dist_seq, False)
                for seq in ecvq_dist_seq:
                    view1_dist_name = ref_name + '_' + str(seq + 1) + '_' + view1_id
                    view2_dist_name = ref_name + '_' + str(seq + 1) + '_' + view2_id
                    
                    view1_img = np.load(view1_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view1_dist_name)
                    view2_img = np.load(view2_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view2_dist_name)
                    
                    view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                    view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #                    prev_crop = pyramid(prev_crop, scale, res)
    #                    next_crop = pyramid(next_crop, scale, res)
                    
    #                dist_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
    #                dist_view2 = optical_flow.calc(prev_crop, next_crop, None)
    
                    
                    crop_view1.append(view1_crop)
                    crop_view2.append(view2_crop)
                    
            elif train_db == 'evvq':
                evvq_dist_seq = np.random.choice(evvq_num_seq[ref], dist_seq, False)
                for seq in evvq_dist_seq:
                    view1_dist_name = ref_name + '_' + str(seq + 1) + '_' + view1_id
                    view2_dist_name = ref_name + '_' + str(seq + 1) + '_' + view2_id
                    
                    view1_img = np.load(view1_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view1_dist_name)
                    view2_img = np.load(view2_ref_name.rsplit('/',3)[0] + '/distorted/' + ref_name + '/' + view2_dist_name)
                    
                    view1_crop = view1_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                    view2_crop = view2_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #                    prev_crop = pyramid(prev_crop, scale, res)
    #                    next_crop = pyramid(next_crop, scale, res)
                    
    #                dist_view1 = prev_crop.astype('int16') - next_crop.astype('int16')
    #                dist_view2 = optical_flow.calc(prev_crop, next_crop, None)
    
                    
                    crop_view1.append(view1_crop)
                    crop_view2.append(view2_crop)
#            crop = np.concatenate([crop_view1, crop_view2], axis = 0)
            view1_train.append(crop_view1)
            view2_train.append(crop_view2)
    else:
        for _ in range(batch_size):
            ref = np.random.randint(25)  
    #    for ref in ref_id:
            crop_view1 = []
            crop_view2 = []
        
            curr_names = curr_filenames[ref]
            next_names = next_filenames[ref]
        
            frame_id = np.random.randint(0,len(curr_names),2) 
            
            curr_ref_name = curr_names[frame_id[0]]
            curr_img = np.load(curr_ref_name)
            
            next_ref_name = next_names[frame_id[0]]
            next_img = np.load(next_ref_name)
            
            crop_x1,crop_y1 = center_crop(curr_img, (patch_size,patch_size))
            crop_x2,crop_y2 = crop_x1,crop_y1#random_crop(view2_img, (patch_size,patch_size))
            
            curr_crop = curr_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :]
            next_crop = next_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]        
            
            ref_view1 = curr_crop
            crop_view1.append(ref_view1)        
            
            ref_view2 = curr_crop.astype('int16') - next_crop.astype('int16')#optical_flow.calc(curr_crop, next_crop, None)
#                ref_view2 = optical_flow.calc(cv2.resize(curr_crop,None, fx=flow_scale,fy=flow_scale),  cv2.resize(next_crop,None, fx=flow_scale,fy=flow_scale), None)
#                ref_view2 = cv2.resize(ref_view2, None, fx= 1/flow_scale, fy = 1/flow_scale)*(1/flow_scale)
            crop_view2.append(ref_view2)        
            
    #        ref_name, view1_id = curr_ref_name.split('/')[-1].rsplit('_',1) 
    #        ref_name, view2_id = next_ref_name.split('/')[-1].rsplit('_',1)       
            
            lsvd_dist_seq = np.random.choice(12, dist_seq, False)
            for seq in lsvd_dist_seq:               
                 
    #            neg_name1 = ref_name +   '_' + view1_id
    #            neg_name2 = ref_name +   '_' + view2_id
                
                curr_img = np.load(curr_ref_name.rsplit('/',2)[0] + '/' + str(seq+1) + '/' + curr_ref_name.split('/')[-1])
                next_img = np.load(next_ref_name.rsplit('/',2)[0] + '/' + str(seq+1) + '/' + next_ref_name.split('/')[-1])
    
                curr_crop = curr_img[crop_y1:(crop_y1+patch_size), crop_x1:(crop_x1+patch_size), :] 
                next_crop = next_img[crop_y2:(crop_y2+patch_size), crop_x2:(crop_x2+patch_size), :]
    
    #            prev_crop = pyramid(prev_crop, scale, res)
    #            next_crop = pyramid(next_crop, scale, res)
                
                dist_view1 = curr_crop
                
                dist_view2 = curr_crop.astype('int16') - next_crop.astype('int16')#optical_flow.calc(prev_crop, next_crop, None)
#                    dist_view2 = optical_flow.calc(cv2.resize(curr_crop,None, fx=flow_scale,fy=flow_scale),  cv2.resize(next_crop,None, fx=flow_scale,fy=flow_scale), None)
#                    dist_view2 = cv2.resize(dist_view2, None, fx= 1/flow_scale, fy =1/flow_scale)*(1/flow_scale)
                
                crop_view1.append(dist_view1)
                crop_view2.append(dist_view2)                                
                    
            crop_view1 = np.array(crop_view1)
            crop_view2 = np.array(crop_view2) 
#            
            view1_train.append(crop_view1)
            view2_train.append(crop_view2)
#        del crop_view1, crop_view2   
    
    view1_train = np.array(view1_train)#(np.array(X_train) +255)//2  
    view2_train = np.array(view2_train)
    
    y_train = np.ones([len(view1_train),1])
    
#    model.fit(X_train,y_train, batch_size = batch_size, epochs=50)
    loss.append(model.train_on_batch([view1_train, view2_train], y_train))
    epoch = epoch + 1
    
    duration = (time.time() - start_time)/60
    
    print('epoch = %4d , loss = %4.6f, time = %4.2f m' %(epoch, loss[-1], duration))
    
    output_directory = path + '/DATA/Shankhanil/VQA/'
    if epoch and epoch%1000 == 0:
#        base_model.compile(optimizer=model.optimizer, loss = loss_fn)
        
        view1_model.save(output_directory + 'tmp/contrastive_models/test/' +'FvsFD_frame_contrastive_' + str(0) + 'x_'+  str(epoch) + '.h5')
        view2_model.save(output_directory + 'tmp/contrastive_models/test/' +'FvsFD_diff_contrastive_' + str(0) + 'x_'+  str(epoch) + '.h5') #
        
        if epoch%10000==0:
            K.clear_session()
            break
#    break
