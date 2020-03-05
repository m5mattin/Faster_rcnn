################################


################################
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import argparse
parser = argparse.ArgumentParser(description='Argument')
parser.add_argument('--device', type=str,
                    help='device visible for keras (CPU: -1)')
parser.add_argument('--num_rois', type=int,
			help='num_rois')
parser.add_argument('--train_label', type=str,
			help='CSV file with annotation')
parser.add_argument('--test_label', type=str,
			help='CSV file with annotation')
parser.add_argument('--images_path', type=str,
			help='images folder')
parser.add_argument('--mode', type=str,
			help='training_mode')
parser.add_argument('--checkpoint', type=str,
			help='training_mode')

args = parser.parse_args()

device = args.device
images_path = args.images_path
mode = args.mode
checkpoint_to_load = args.checkpoint

if device:
    os.environ["CUDA_VISIBLE_DEVICES"]=device

if args.num_rois:
    num_rois = args.num_rois
else:
    num_rois = 4

base_path = os.getcwd()
train_path = args.train_label
test_path = args.test_label

import random
import pprint
import sys
import time
import numpy as np
import pickle
import math
import cv2
import copy

import tensorflow as tf
import pandas as pd

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

from new_utils import *


############################################################
                    #CONFIG SETTING
############################################################

class Config:

    def __init__(self):

        # Print the process or not
        self.verbose = True

        # Name of base network
        self.network = 'vgg'

        # Setting for data augmentation
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        # Anchor box scales
    # Note that if im_size is smaller, anchor_box_scales should be scaled
    # Original anchor_box_scales in the paper is [128, 256, 512]
        self.anchor_box_scales = [100, 175, 350]
        #self.anchor_box_scales = [64, 128, 256, 512, 1024]

        # Anchor box ratios
        #self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

        # Size to resize the smallest side of the image
        # Original setting in paper is 600. Set to 300 in here to save training time
        self.im_size = 720

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 4

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_num_train_examples = 150 
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.65

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.0
        self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = None

        self.model_path = None

############################################################
                        #Start training
############################################################


from tensorflow.python.client import device_lib

print("List of available devices on tensorflow : ")
print(device_lib.list_local_devices())

print("List of available devices on Keras : ")
print(K.tensorflow_backend._get_available_gpus())
print("#############")

# Augmentation flag
horizontal_flips = False # Augment with horizontal flips in training. 
vertical_flips = False  # Augment with vertical flips in training. 
rot_90 = False           # Augment with 90 degree rotations in training. 

output_weight_path = '../model_frcnn_vgg.hdf5' # File to store the best model

record_path_train = '../record_train.csv' # Record data (used to save the losses, classification accuracy and mean average precision)
record_path_test = '../record_test.csv'

base_weight_path = '../vgg16_weights_tf_dim_ordering_tf_kernels.h5' # Initial weights for vgg

config_output_filename = '../model_vgg_config.pickle' # config file

# Create the config
C = Config()

C.use_horizontal_flips = horizontal_flips
C.use_vertical_flips = vertical_flips
C.rot_90 = rot_90
C.record_path_train = record_path_train
C.record_path_test = record_path_test
C.model_path = output_weight_path
C.num_rois = num_rois
C.base_net_weights = base_weight_path

#--------------------------------------------------------#
# This step will spend some time to load the data        #
#--------------------------------------------------------#

train_imgs, classes_count_train, class_mapping = get_data(train_path,images_path)
test_imgs, classes_count_test, _ = get_data(test_path,images_path)

class_mapping_label = class_mapping.copy()
if mode != 'PaCO':
    print("class_mapping was {} but because mode is {} so class_mapping is :".format(class_mapping, mode))
    class_mapping.pop('others',None)
    
if 'bg' not in classes_count_train:
    classes_count_train['bg'] = 0
    class_mapping['bg'] = len(class_mapping)
    class_mapping_label['bg'] = len(class_mapping_label)
print("     - class_mapping : ",class_mapping)
C.class_mapping = class_mapping

print('Num train samples (images) : {}'.format(len(train_imgs)))
print('Num test samples (images) : {}'.format(len(test_imgs)))
print('Training sub-images per class : {}'.format(classes_count_train))
print('Test sub-images per class : {}'.format(classes_count_test))

#Save the configuration
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
    print("-------------------------------------")

# Shuffle images with seed
# random.seed(1)
# random.shuffle(train_imgs)
# Get train data generator which generate X, Y, image_data

data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length, mode=mode)
data_gen_test = get_anchor_gt(test_imgs, C, get_img_output_length, mode=mode)

############################################################
                    #Build the model
############################################################
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
rpn = rpn_layer(shared_layers, num_anchors)
classifier = classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(class_mapping))
model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# Because the google colab can only run the session several hours one time (then you need to connect again), 
# we need to save the model and load the model to continue training
if checkpoint_to_load:
    print('Continue training based on previous trained model')
    print('Loading weights from {}'.format(checkpoint_to_load))
    model_rpn.load_weights(checkpoint_to_load, by_name=True)
    model_classifier.load_weights(checkpoint_to_load, by_name=True)
    
    # Load the records
    record_df_train = pd.read_csv("../record_train.csv")
    record_df_test = pd.read_csv("../record_test.csv")

    print('Already train %dK batches'% (len(record_df_train)))

else:
# if not os.path.isfile(C.model_path):
    #If this is the begin of the training, load the pre-traind base network such as vgg-16
    try:
        print('This is the first time of your training')
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')
    
    # Create the record.csv file to record losses, acc and mAP

    record_df_train = pd.DataFrame(columns=[    'loss_rpn_cls', 'loss_rpn_regr', 
                                                'rpn_00', 'rpn_01' , 'rpn_02',
                                                'rpn_10', 'rpn_11' , 'rpn_12',
                                                'loss_class_cls', 'loss_class_regr',
                                                'class_00', 'class_01', 'class_02',
                                                'class_10', 'class_11', 'class_12',
                                                'class_20', 'class_21', 'class_22',
                                                'curr_loss','rpn_mop','rpn_moo','rpn_mob',
                                                'ap_rpn50','ap_rpn75','ap_detect50','ap_detect75'])

    record_df_test = pd.DataFrame(columns=[    'loss_rpn_cls', 'loss_rpn_regr', 
                                                'rpn_00', 'rpn_01' , 'rpn_02',
                                                'rpn_10', 'rpn_11' , 'rpn_12',
                                                'loss_class_cls', 'loss_class_regr',
                                                'class_00', 'class_01', 'class_02',
                                                'class_10', 'class_11', 'class_12',
                                                'class_20', 'class_21', 'class_22',
                                                'curr_loss','rpn_mop','rpn_moo','rpn_mob',
                                                'ap_rpn50','ap_rpn75','ap_detect50','ap_detect75'])
Learning_rate = 1e-4
# optimizer = Adam(lr=Learning_rate)
# optimizer_classifier = Adam(lr=Learning_rate)
optimizer = SGD(lr=Learning_rate)
optimizer_classifier = SGD(lr=Learning_rate)

model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(class_mapping)-1)], metrics=["accuracy"])

model_all.compile(optimizer='sgd', loss='mae')

# Training setting
total_epochs = len(record_df_train)
r_epochs = len(record_df_train)

epoch_length = len(train_imgs)
imagestest_length = len(test_imgs)
num_epochs = 100 - r_epochs
iter_num = 0

total_epochs += num_epochs

losses_train = np.zeros((epoch_length, 8))
losses_test = np.zeros((imagestest_length, 8))

rpn_accuracy_rpn_monitor_train = []
rpn_accuracy_for_epoch_train = []
rpn_accuracy_rpn_monitor_test = []
rpn_accuracy_for_epoch_test = []

rpn_overlaping_train = np.zeros((epoch_length,3,num_epochs))
rpn_overlaping_test = np.zeros((imagestest_length,3,num_epochs))

best_loss_train = np.Inf

start_time = time.time()

st_epoch = time.time()

for epoch_num in range(num_epochs):
    print("time/epoch:{}".format(st_epoch-time.time()))
    st_epoch = time.time()

    progbar_train = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
    # if (epoch_num+1) % 20 == 0:
    #     print('Reduce current learning rate by 10. \nCurrent learning rate = ',  Learning_rate, '\nNew learning rate = ', Learning_rate/10)
    #     Learning_rate = Learning_rate/10
    #     K.set_value(model_rpn.optimizer.lr, Learning_rate)
    #     K.set_value(model_classifier.optimizer.lr, Learning_rate)  
    
    r_epochs += 1
    st_step = time.time()

    # Init RPN
    rpn_confusion_matrix_train = np.zeros((2,3))
    rpn_confusion_matrix_test = np.zeros((2,3))

    # Init Class
    class_confusion_matrix_train = np.zeros((3,3))
    class_confusion_matrix_test = np.zeros((3,3))
                
    while True:
        try:
            if len(rpn_accuracy_rpn_monitor_train) == epoch_length and C.verbose:
                mean_overlapping_bboxes_train = float(sum(rpn_accuracy_rpn_monitor_train))/len(rpn_accuracy_rpn_monitor_train)
                rpn_accuracy_rpn_monitor_train = []
                if mean_overlapping_bboxes_train == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            
            # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
            print("mode : {}".format(mode))
            X_train, Y_train, img_data_train, debug_img_train, debug_num_pos_train, hard_anchors_others = next(data_gen_train)
            # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
            loss_rpn_train = model_rpn.train_on_batch(X_train, Y_train)
            # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
            P_rpn_train = model_rpn.predict_on_batch(X_train)
            # Convert rpn layer to roi bboxes
            R_train, probs = rpn_to_roi(P_rpn_train[0], P_rpn_train[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # NMS 0. 45
            R_rpn_nms, probs_nms, valid = non_max_suppression_fast(R_train,probs,overlap_thresh=0.45)
            # Get all boxes in list
            all_det_rpn = []
            if valid is True:
                for i in range(R_rpn_nms.shape[0]):
                    if probs_nms[i] > 0.5:
                        all_det_rpn.append((R_rpn_nms[i]* C.rpn_stride ,0,probs_nms[i]))
                # Calcul average precision with groundtruth
                ap_rpn_train50 = get_average_precision(all_det_rpn,img_data_train['bboxes'],0.50)
                ap_rpn_train75 = get_average_precision(all_det_rpn,img_data_train['bboxes'],0.75)
            else:
                # If no box detected ap = 0
                ap_rpn_train50 = 0
                ap_rpn_train75 = 0
            
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
            # Y1: one hot code for bboxes from above => x_roi (X)
            # Y2: corresponding labels and corresponding gt bboxes
            X2_train, Y1_train, Y2_train, IouS_train, overlap_others = calc_iou(R_train, img_data_train, C, class_mapping_label)

            # Get matches between rpn boxes and real classes (number of pig, number of others, number of neg)
            for i in range(len(Y1_train[0])):
                j = np.where(Y1_train[0,i,:]>0)
                rpn_overlaping_train[iter_num,j,epoch_num] = rpn_overlaping_train[iter_num,j,epoch_num] + 1

            # get confusion matrix for rpn boxes 
            tmp_confusion = compare_rpn_to_groundtruth(img_data_train['bboxes'],R_train*C.rpn_stride,num_boxes=300)
            rpn_confusion_matrix_train = rpn_confusion_matrix_train + tmp_confusion
            
            # If X2 is None means there are no matching bboxes
            if X2_train is None:
                rpn_accuracy_rpn_monitor_train.append(0)
                rpn_accuracy_for_epoch_train.append(0)
                continue
            
            # Find out the positive anchors and negative anchors

            neg_samples_train = np.where(Y1_train[0, :, -1] == 1)
            pos_samples_train = np.where(Y1_train[0, :, -1] == 0)

            if len(neg_samples_train) > 0:
                neg_samples_train = neg_samples_train[0]
            else:
                neg_samples_train = []

            if len(pos_samples_train) > 0:
                pos_samples_train = pos_samples_train[0]
            else:
                pos_samples_train = []
            rpn_accuracy_rpn_monitor_train.append(len(pos_samples_train))
            rpn_accuracy_for_epoch_train.append((len(pos_samples_train)))

            ###### TEST ON BATCH
            Y_detection_train = []
            samples_cross_entropy = np.zeros((300))
            for k in range(int(len(Y1_train[0])//C.num_rois)):
                
                X2, Y1, Y2, sel_samples = get_testing_batch_classifier(C, X2_train, Y1_train, Y2_train, mode, k)
                [P_cls, P_regr] = model_classifier.predict([X_train, X2])

                for j in range(len(P_cls[0])):
                         # Calcul loss per example
                        samples_cross_entropy[k*C.num_rois+j] = cross_entropy(Y1[0][j], P_cls[0][j])
                                     
                #Fill confusion matrix for classification 
                for i in range (len(sel_samples)):
                    class_predicted = np.where(P_cls[0][i] == np.amax(P_cls[0][i]))
                    class_predicted = int(class_predicted[0][0])
                    class_gt = np.where(Y1_train[0, sel_samples[i], :]== np.amax(Y1_train[0, sel_samples[i], :]))
                    class_gt = int(class_gt[0])
                    
                    if (mode != 'PaCO') :
                        if class_predicted == len(class_mapping) - 1:
                            class_predicted = class_predicted + 1

                    class_confusion_matrix_train[class_predicted, class_gt] = class_confusion_matrix_train[class_predicted, class_gt] + 1
                    # Save boxes with format (boxe, cls, regr)
                    
                    Y_detection_train.append( (X2_train[:, sel_samples[i], :][0], P_cls[0][i], P_regr[0][i]))  
            # get boxes probability
            bboxes, probs = get_detections_boxes(Y_detection_train, C, class_mapping)
            # Get Average precision with NMS : 0.45
            bboxes, probs, valid = non_max_suppression_fast(bboxes,probs,overlap_thresh=0.45)
            all_det = []
            if valid is True:
                for i in range(bboxes.shape[0]):
                    if probs[i] > 0.5:          
                        all_det.append((bboxes[i],0,probs[i]))
                ap_detect_train50 = get_average_precision(all_det,img_data_train['bboxes'],0.50)
                ap_detect_train75 = get_average_precision(all_det,img_data_train['bboxes'],0.75)
            else :
                ap_detect_train50 = 0
                ap_detect_train75 = 0
            ###### TRAIN ON BATCH
            # Get samples if HA is needed 
            hard_anchors_others,_ = nms_boxes(boxes=np.asarray(hard_anchors_others),probs=None)
            
            # Get samples if HOEM is needed            
            tmp = X2_train.copy()
            tmp[:,:,0] = X2_train[0][:,0].copy()
            tmp[:,:,1] = X2_train[0][:,1].copy()
            tmp[:,:,2] = X2_train[0][:,0].copy() +  X2_train[0][:,2].copy()
            tmp[:,:,3] = X2_train[0][:,1].copy() +  X2_train[0][:,3].copy() 
            _,sel_samples_HOEM = nms_boxes(boxes=tmp[0],probs=samples_cross_entropy,overlap_thresh=0.7,max_boxes=30)

            X2, Y1, Y2, sel_samples = get_training_batch_classifier(    C=C, 
                                                                        X2_train=X2_train, 
                                                                        Y1_train=Y1_train, 
                                                                        Y2_train = Y2_train, 
                                                                        IouS_train = IouS_train, 
                                                                        mode = mode, 
                                                                        overlap_others = overlap_others, 
                                                                        hard_anchors_others = hard_anchors_others,
                                                                        sel_samples_HOEM = sel_samples_HOEM)
            
            # for i in range(len(X2[0])):
            #     print(i,X2[0][i],Y1[0][i])

            loss_class_train = model_classifier.train_on_batch([X_train,X2], [Y1,Y2]) 
            # Loss rpn  
            losses_train[iter_num, 0] = loss_rpn_train[1]
            losses_train[iter_num, 1] = loss_rpn_train[2]
            
            # Loss classification
            losses_train[iter_num, 2] = loss_class_train[1]
            losses_train[iter_num, 3] = loss_class_train[2]

            # Average precision
            losses_train[iter_num, 4] = ap_rpn_train50
            losses_train[iter_num, 5] = ap_rpn_train75
            losses_train[iter_num, 6] = ap_detect_train50
            losses_train[iter_num, 7] = ap_detect_train75

            iter_num += 1
            progbar_train.update(iter_num, [('rpn_cls', np.mean(losses_train[:iter_num, 0])), ('rpn_regr', np.mean(losses_train[:iter_num, 1])),
                                      ('final_cls', np.mean(losses_train[:iter_num, 2])), ('final_regr', np.mean(losses_train[:iter_num, 3]))])             
            if iter_num == epoch_length:
                
                loss_rpn_cls_train = np.mean(losses_train[:, 0])
                loss_rpn_regr_train = np.mean(losses_train[:, 1])
                loss_class_cls_train = np.mean(losses_train[:, 2])
                loss_class_regr_train = np.mean(losses_train[:, 3])
                ap_rpn_train50 = np.mean(losses_train[:, 4])
                ap_rpn_train75 = np.mean(losses_train[:, 5])
                ap_detect_train50 = np.mean(losses_train[:, 6])
                ap_detect_train75 = np.mean(losses_train[:, 7])

                rpn_mop = np.mean(rpn_overlaping_train[:,0,epoch_num])
                rpn_moo = np.mean(rpn_overlaping_train[:,1,epoch_num])
                rpn_mob = np.mean(rpn_overlaping_train[:,2,epoch_num])
  
                mean_overlapping_bboxes_train = float(sum(rpn_accuracy_for_epoch_train)) / len(rpn_accuracy_for_epoch_train)
                rpn_accuracy_for_epoch_train = []  
                
                if C.verbose:
                    print('epoch {}'.format(epoch_num))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls_train))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr_train))
                    print('Loss Detector classifier: {}'.format(loss_class_cls_train))
                    print('Loss Detector regression: {}'.format(loss_class_regr_train))
                    print('Total loss: {}'.format(loss_rpn_cls_train + loss_rpn_regr_train + loss_class_cls_train + loss_class_regr_train))

                    elapsed_time = (time.time()-start_time)/60 
                curr_loss_train = loss_rpn_cls_train + loss_rpn_regr_train + loss_class_cls_train + loss_class_regr_train
                iter_num = 0

#                 if curr_loss_train < best_loss_train:
#                     if C.verbose:
#                         print('Total loss decreased from {} to {}, saving weights'.format(best_loss_train,curr_loss_train))
#                     best_loss_train = curr_loss_train
#                     model_all.save_weights(C.model_path)
#                     # with open("checkpoint/checkpoint.csv", "w") as csv_file:
#                     #     csv_file.write("epoch:{}, best_loss:{} \n".format(len(record_df_train),best_loss_train))

                #Save weight from the current model 
                checkpoint_path = "../checkpoint/model_frcnn_vgg_epoch" + mode + str(len(record_df_train))+".hdf5"
                model_all.save_weights(checkpoint_path) 
                new_row_train = {   'loss_rpn_cls':round(loss_rpn_cls_train, 3), 
                                    'loss_rpn_regr':round(loss_rpn_regr_train, 3), 
                                    'loss_class_cls':round(loss_class_cls_train, 3), 
                                    'loss_class_regr':round(loss_class_regr_train, 3), 
                                    'curr_loss':round(curr_loss_train, 3),
                                    'rpn_00':round(rpn_confusion_matrix_train[0,0], 3),
                                    'rpn_01':round(rpn_confusion_matrix_train[0,1], 3) ,
                                    'rpn_02':round(rpn_confusion_matrix_train[0,2], 3),
                                    'rpn_10':round(rpn_confusion_matrix_train[1,0], 3),
                                    'rpn_11':round(rpn_confusion_matrix_train[1,1], 3) ,
                                    'rpn_12':round(rpn_confusion_matrix_train[1,2], 3),      
                                    'class_00':round(class_confusion_matrix_train[0,0], 3),
                                    'class_01':round(class_confusion_matrix_train[0,1], 3) ,
                                    'class_02':round(class_confusion_matrix_train[0,2], 3),
                                    'class_10':round(class_confusion_matrix_train[1,0], 3),
                                    'class_11':round(class_confusion_matrix_train[1,1], 3) ,
                                    'class_12':round(class_confusion_matrix_train[1,2], 3),
                                    'class_20':round(class_confusion_matrix_train[2,0], 3),
                                    'class_21':round(class_confusion_matrix_train[2,1], 3) ,
                                    'class_22':round(class_confusion_matrix_train[2,2], 3),
                                    'rpn_mop':round(rpn_mop, 3),
                                    'rpn_moo':round(rpn_moo, 3),
                                    'rpn_mob':round(rpn_mob, 3),
                                    'ap_rpn50':round(ap_rpn_train50,3),
                                    'ap_rpn75':round(ap_rpn_train75,3),
                                    'ap_detect50':round(ap_detect_train50,3),
                                    'ap_detect75':round(ap_detect_train75,3)
                                    }  

                for num_image in range (len(test_imgs)):  
                    print("epoch {}, test image {}/{} processed".format(epoch_num, num_image+1,len(test_imgs)))

                    X_test, Y_test, img_data_test, debug_img_test, debug_num_pos_test, hard_anchors_others = next(data_gen_test)

                    loss_rpn_test = model_rpn.test_on_batch(X_test, Y_test)

                    P_rpn_test = model_rpn.predict_on_batch(X_test)

                    R_test,probs = rpn_to_roi(P_rpn_test[0], P_rpn_test[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)  
                    R_rpn_nms, probs_nms, valid = non_max_suppression_fast(R_test,probs,overlap_thresh=0.45)
                    all_det_rpn = []
                    if valid is True:
                        for i in range(R_rpn_nms.shape[0]):
                            if probs_nms[i] > 0.5:
                                all_det_rpn.append((R_rpn_nms[i]* C.rpn_stride ,0,probs_nms[i]))

                        ap_rpn_test50 = get_average_precision(all_det_rpn,img_data_test['bboxes'],0.50)
                        ap_rpn_test75 = get_average_precision(all_det_rpn,img_data_test['bboxes'],0.75)
                    else:
                        ap_rpn_test50 = 0
                        ap_rpn_test75 = 0

                    X2_test, Y1_test, Y2_test, IouS_test, overlap_others = calc_iou(R_test, img_data_test, C, class_mapping_label)
                    for i in range(len(Y1_test[0])):
                        j = np.where(Y1_test[0,i,:]>0)
                        rpn_overlaping_test[num_image,j,epoch_num] = rpn_overlaping_test[num_image,j,epoch_num] + 1
                    tmp_confusion = compare_rpn_to_groundtruth(img_data_test['bboxes'],R_test*C.rpn_stride,num_boxes=300)
                    rpn_confusion_matrix_test = rpn_confusion_matrix_test + tmp_confusion

                    if X2_test is None:
                        rpn_accuracy_rpn_monitor_test.append(0)
                        rpn_accuracy_for_epoch_test.append(0)
                        continue
                    ###### TEST ON BATCH
                    Y_detection_test = [] 
                    samples_cross_entropy = np.zeros((300))

                    for k in range(int(len(Y1_test[0])//C.num_rois)):
                        
                        X2, Y1, Y2, sel_samples = get_testing_batch_classifier(C, X2_test, Y1_test, Y2_test, mode, k)
                        [P_cls, P_regr] = model_classifier.predict([X_test, X2])
                        for i in range(len(P_cls[0])):
                            # Calcul loss per example
                            samples_cross_entropy[k*C.num_rois+i] = cross_entropy(Y1[0][i], P_cls[0][i])
                                      
                        for i in range (len(sel_samples)):
                            class_predicted = np.where(P_cls[0][i] == np.amax(P_cls[0][i]))
                            class_predicted = int(class_predicted[0][0])
                            class_gt = np.where(Y1_test[0, sel_samples[i], :]== np.amax(Y1_test[0, sel_samples[i], :]))
                            class_gt = int(class_gt[0])

                            if (mode != 'PaCO') :
                                if class_predicted == len(class_mapping) - 1:
                                    class_predicted = class_predicted + 1

                            class_confusion_matrix_test[class_predicted, class_gt] = class_confusion_matrix_test[class_predicted, class_gt] + 1
                            Y_detection_test.append( (X2_test[:, sel_samples[i], :][0], P_cls[0][i], P_regr[0][i]))
                    
                    bboxes, probs = get_detections_boxes(Y_detection_test, C, class_mapping)
                    bboxes, probs, valid = non_max_suppression_fast(bboxes,probs,overlap_thresh=0.45)
                    all_det = []
                    if valid is True:
                        for i in range(bboxes.shape[0]):
                            if probs[i] > 0.5:          
                                all_det.append((bboxes[i],0,probs[i]))

                        ap_detect_test50 = get_average_precision(all_det,img_data_test['bboxes'],0.50)
                        ap_detect_test75 = get_average_precision(all_det,img_data_test['bboxes'],0.75)
                    else:
                        ap_detect_test50 = 0
                        ap_detect_test75 = 0
                    
                    hard_anchors_others,_ = nms_boxes(boxes=np.asarray(hard_anchors_others),probs=None)
                    tmp = X2_test.copy()
                    tmp[:,:,0] = X2_test[0][:,0].copy()
                    tmp[:,:,1] = X2_test[0][:,1].copy()
                    tmp[:,:,2] = X2_test[0][:,0].copy() +  X2_test[0][:,2].copy()
                    tmp[:,:,3] = X2_test[0][:,1].copy() +  X2_test[0][:,3].copy() 
                    _,sel_samples_HOEM = nms_boxes(boxes=tmp[0],probs=samples_cross_entropy,overlap_thresh=0.7,max_boxes=30)
                    X2, Y1, Y2, sel_samples = get_training_batch_classifier(    C=C, 
                                                                                X2_train=X2_test, 
                                                                                Y1_train=Y1_test, 
                                                                                Y2_train = Y2_test, 
                                                                                IouS_train = IouS_test, 
                                                                                mode = mode, 
                                                                                overlap_others = overlap_others, 
                                                                                hard_anchors_others = hard_anchors_others,
                                                                                sel_samples_HOEM = sel_samples_HOEM)
                    loss_class_test = model_classifier.test_on_batch([X_test,X2], [Y1,Y2]) 
                    
                    
                    losses_test[num_image, 0] = loss_rpn_test[1]
                    losses_test[num_image, 1] = loss_rpn_test[2]
                    
                    losses_test[num_image, 2] = loss_class_test[1]
                    losses_test[num_image, 3] = loss_class_test[2]

                    losses_test[num_image, 4] = ap_rpn_test50
                    losses_test[num_image, 5] = ap_rpn_test75
                    losses_test[num_image, 6] = ap_detect_test50
                    losses_test[num_image, 7] = ap_detect_test75

                    if num_image == (len(test_imgs)-1):
                        
                        loss_rpn_cls_test = np.mean(losses_test[:, 0])
                        loss_rpn_regr_test = np.mean(losses_test[:, 1])
                        loss_class_cls_test = np.mean(losses_test[:, 2])
                        loss_class_regr_test = np.mean(losses_test[:, 3])
                        ap_rpn_test50 = np.mean(losses_test[:, 4])
                        ap_rpn_test75 = np.mean(losses_test[:, 5])
                        ap_detect_test50 = np.mean(losses_test[:, 6])
                        ap_detect_test75 = np.mean(losses_test[:, 7])

                        rpn_mop = np.mean(rpn_overlaping_test[:,0,epoch_num])
                        rpn_moo = np.mean(rpn_overlaping_test[:,1,epoch_num])
                        rpn_mob = np.mean(rpn_overlaping_test[:,2,epoch_num])

                        curr_loss_test = loss_rpn_cls_test + loss_rpn_regr_test + loss_class_cls_test + loss_class_regr_test

                        new_row_test = {    'loss_rpn_cls':round(loss_rpn_cls_test, 3), 
                                            'loss_rpn_regr':round(loss_rpn_regr_test, 3), 
                                            'loss_class_cls':round(loss_class_cls_test, 3), 
                                            'loss_class_regr':round(loss_class_regr_test, 3), 
                                            'curr_loss':round(curr_loss_test, 3),
                                            'rpn_00':round(rpn_confusion_matrix_test[0,0], 3),
                                            'rpn_01':round(rpn_confusion_matrix_test[0,1], 3) ,
                                            'rpn_02':round(rpn_confusion_matrix_test[0,2], 3),
                                            'rpn_10':round(rpn_confusion_matrix_test[1,0], 3),
                                            'rpn_11':round(rpn_confusion_matrix_test[1,1], 3) ,
                                            'rpn_12':round(rpn_confusion_matrix_test[1,2], 3),      
                                            'class_00':round(class_confusion_matrix_test[0,0], 3),
                                            'class_01':round(class_confusion_matrix_test[0,1], 3) ,
                                            'class_02':round(class_confusion_matrix_test[0,2], 3),
                                            'class_10':round(class_confusion_matrix_test[1,0], 3),
                                            'class_11':round(class_confusion_matrix_test[1,1], 3) ,
                                            'class_12':round(class_confusion_matrix_test[1,2], 3),
                                            'class_20':round(class_confusion_matrix_test[2,0], 3),
                                            'class_21':round(class_confusion_matrix_test[2,1], 3) ,
                                            'class_22':round(class_confusion_matrix_test[2,2], 3),
                                            'rpn_mop':round(rpn_mop, 3),
                                            'rpn_moo':round(rpn_moo, 3),
                                            'rpn_mob':round(rpn_mob, 3),
                                            'ap_rpn50':round(ap_rpn_test50, 3),
                                            'ap_rpn75':round(ap_rpn_test75, 3),
                                            'ap_detect50':round(ap_detect_test50, 3),
                                            'ap_detect75':round(ap_detect_test75, 3)
                                    }

                        record_df_train = record_df_train.append(new_row_train, ignore_index=True)
                        record_df_train.to_csv(record_path_train, index=0)
                        np.save('../overlaping_rpn_train.npy', rpn_overlaping_train)
                        record_df_test = record_df_test.append(new_row_test, ignore_index=True)
                        record_df_test.to_csv(record_path_test, index=0)
                        np.save('../overlaping_rpn_test.npy', rpn_overlaping_test)

                print("  time/step:{}".format(time.time()-st_step))
                st_step = time.time()
                break
            
        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')