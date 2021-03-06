
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import csv

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

def get_label(input_path):
    nb_total_images = 0
    nb_total_pigs = 0
    nb_total_others = 0

    with open(input_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        list_r = list(reader)
        filename = "none"
        data = {}
        i = 1

        while (i < len(list_r)):
            if list_r[i][0]!=filename:
                data[list_r[i][0]]= {}
                filename = list_r[i][0]
                nb_total_images = nb_total_images + 1
            classe = list_r[i][3]
            if classe=="pig":
                nb_total_pigs = nb_total_pigs + 1
            length_of_line = len(list_r[i])
            if classe in data[filename]:
                boxe = list_r[i][4:length_of_line]
                for j in range(0, len(boxe)): 
                    boxe[j] = int(boxe[j]) 
                data[filename][classe].append(boxe)

            else:
                data[filename][classe]=[]
                boxe = list_r[i][4:length_of_line]
                for j in range(0, len(boxe)): 
                    boxe[j] = int(boxe[j]) 
                data[filename][classe].append(boxe)
            
            i = i + 1

        return data, nb_total_images, nb_total_pigs, nb_total_others

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

############################################################
            #PARSER THE DATA FROM ANNOTATION FILE
############################################################

def get_data(input_path,images_folder):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

	Returns:
		all_data: list(filepath, width, height, list(bboxes))
		classes_count: dict{key:class_name, value:count_num} 
			e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
		class_mapping: dict{key:class_name, value: idx}
			e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
	"""
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 1
	
    with open(input_path,'r') as f:

        print('Parsing annotation files : ', input_path)

        for line in f:
            # Print process
            #sys.stdout.write('\r'+'idx=' + str(i))
            i += 1
            line_split = line.strip().split(',')

            # Make sure the info saved in annotation file matching the format (path_filename, x1, y1, x2, y2, class_name)
            # Note:
            #	One path_filename might has several classes (class_name)
            #	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
            #	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
            #   x1,y1-------------------
            #	|						|
            #	|						|
            #	|						|
            #	|						|
            #	---------------------x2,y2
            (filename,_,_,class_name,x1,y1,x2,y2) = line_split
            if filename[0]!="#" and filename != "filename" :
                filename = os.path.join(images_folder, filename)
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    if class_name == 'bg' and found_bg == False:
                        print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                        found_bg = True
                    class_mapping[class_name] = len(class_mapping)

                if filename not in all_imgs:
                    all_imgs[filename] = {}
                    
                    img = cv2.imread(filename)
                    (rows,cols) = img.shape[:2]
                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['bboxes'] = []
                    # if np.random.randint(0,6) > 0:
                    # 	all_imgs[filename]['imageset'] = 'trainval'
                    # else:
                    # 	all_imgs[filename]['imageset'] = 'test'

                all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    # make sure the bg class is last in the list
    if found_bg:
        if class_mapping['bg'] != len(class_mapping) - 1:
            key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
            val_to_switch = class_mapping['bg']
            class_mapping['bg'] = len(class_mapping) - 1
            class_mapping[key_to_switch] = val_to_switch

    return all_data, classes_count, class_mapping



############################################################
                    #VGG 16 MODEL
############################################################

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)    

def nn_base(input_tensor=None, trainable=False):


    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x

############################################################
                    #RPN LAYER
############################################################

def rpn_layer(base_layers, num_anchors):
    """Create a rpn layer
        Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
                Keep the padding 'same' to preserve the feature map's size
        Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
    Args:
        base_layers: vgg in here
        num_anchors: 9 in here

    Returns:
        [x_class, x_regr, base_layers]
        x_class: classification for whether it's an object
        x_regr: bboxes regression
        base_layers: vgg in here
    """
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

############################################################
                    #Classifier layer
############################################################

def classifier_layer(base_layers, input_rois, num_rois, nb_classes = 4):
    """Create a classifier layer
    
    Args:
        base_layers: vgg
        input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        num_rois: number of rois to be processed in one time (4 in here)

    Returns:
        list(out_class, out_regr)
        out_class: classifier layer output
        out_regr: regression layer output
    """

    #input_shape = (num_rois,7,7,512)

    pooling_regions = 7

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    
    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

############################################################
           #DEFINE ROI POOLING CONVOLUTIONAL LAYER
############################################################

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.common.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)
        

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    ### X => image ou feature map ? 
    def call(self, x, mask=None):

        assert(len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)
                

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 512)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

############################################################
                    #Calculate IoU
############################################################

def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union

def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h

def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)

def iiou(a,b):
    curr_iou = iou(a,b)
    curr_intersection = intersection(a,b)
    curr_union = union(a,b,curr_intersection)
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[2], b[2])
    y2 = max(a[3], b[3])

    min_area = abs((x2-x1)*(y2-y1))
    iiou = curr_iou - ((min_area - curr_union)/min_area)

    return iiou


############################################################
            #Calculate the the rpn for all images
############################################################
def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function, mode):
    """(Important part!) Calculate the rpn for all anchors 
        If feature map has shape 38x50=1900, there are 1900x9=17100 potential anchors

    Args:
        C: config
        img_data: augmented image data
        width: original image width (e.g. 600)
        height: original image height (e.g. 800)
        resized_width: resized image width according to C.im_size (e.g. 300)
        resized_height: resized image height according to C.im_size (e.g. 400)
        img_length_calc_function: function to calculate final layer's feature map (of base model) size according to input image size

    Returns:
        y_rpn_cls: list(num_bboxes, y_is_box_valid + y_rpn_overlap)
            y_is_box_valid: 0 or 1 (0 means the box is invalid, 1 means the box is valid)
            y_rpn_overlap: 0 or 1 (0 means the box is not an object, 1 means the box is an object)
        y_rpn_regr: list(num_bboxes, 4*y_rpn_overlap + y_rpn_regr)
            y_rpn_regr: x1,y1,x2,y2 bunding boxes coordinates
    """
    
    downscale = float(C.rpn_stride) 
    anchor_sizes = C.anchor_box_scales   # 128, 256, 512
    anchor_ratios = C.anchor_box_ratios  # 1:1, 1:2*sqrt(2), 2*sqrt(2):1
    num_anchors = len(anchor_sizes) * len(anchor_ratios) # 3x3=9

    # calculate the output map size based on the network architecture
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)    # 3

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth
    list_others_anchors = []
    list_all_others_anchors = []
    list_pig_anchors = []
    list_neg_anchors = []
    list_neutral_anchors = []

    cnt = 0
    for anchor_size_idx in range(len(anchor_sizes)):

        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
            
            for ix in range(output_width):					
                # x-coordinates of the current anchor box	
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
                
                # ignore boxes that go across image boundaries					
                if x1_anc < 0 or x2_anc > resized_width:
                    continue
                    
                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    # Initialize with 'negative'
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0                    
                    best_iou_with_others = 0.0
                    cnt = cnt + 1
                    for bbox_num in range(num_bboxes):    
                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])

                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc)/2.0
                            cya = (y1_anc + y2_anc)/2.0

                            # x,y are the center point of ground-truth bbox
                            # xa,ya are the center point of anchor bbox (xa=downscale * (ix + 0.5); ya=downscale * (iy+0.5))
                            # w,h are the width and height of ground-truth bbox
                            # wa,ha are the width and height of anchor bboxe
                            # tx = (x - xa) / wa
                            # ty = (y - ya) / ha
                            # tw = log(w / wa)
                            # th = log(h / ha)
                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                        
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)
                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'
                        
                            if img_data['bboxes'][bbox_num]['class'] == 'others':
                                if curr_iou > best_iou_with_others:
                                    best_iou_with_others = curr_iou

                    # turn on or off outputs depending on IOUs
                    if best_iou_with_others > 0:
                        if best_iou_with_others > C.rpn_max_overlap:
                            list_others_anchors.append((jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx, best_regr))
                            list_all_others_anchors.append((jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx))
                        else:
                            list_all_others_anchors.append((jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx))

                    elif bbox_type == 'pos':
                        list_pig_anchors.append((jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx, best_regr))
                    elif bbox_type == 'neutral':
                        list_neutral_anchors.append((jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx))
                    elif bbox_type == 'neg':
                        list_neg_anchors.append((jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx))
                    
    # we ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue

            elif img_data['bboxes'][idx]['class'] == 'pig':
                list_pig_anchors.append((best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3],best_dx_for_bbox[idx, :]))

            elif img_data['bboxes'][idx]['class'] == 'others':
                list_others_anchors.append((best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3],best_dx_for_bbox[idx, :]))
    
    random.shuffle(list_pig_anchors)
    random.shuffle(list_others_anchors)
    random.shuffle(list_all_others_anchors)
    random.shuffle(list_neg_anchors)
    pos = []
    neg = []

    if (mode == 'P'):
        # Fill positives with pig / negatives with background
        if len(list_pig_anchors) <= C.rpn_num_train_examples/2:
            pos = list_pig_anchors
        elif len(list_pig_anchors) > C.rpn_num_train_examples/2:
            pos = list_pig_anchors[:int((C.rpn_num_train_examples/2))]
        
        neg = list_neg_anchors[:(C.rpn_num_train_examples-len(pos))]

        # Fill positives with pig / negatives with background or others
    if (mode == 'PaO') or (mode == 'HOEM'):
        if len(list_pig_anchors) <= C.rpn_num_train_examples/2:
            pos = list_pig_anchors
        elif len(list_pig_anchors) > C.rpn_num_train_examples/2:
            pos = list_pig_anchors[:int((C.rpn_num_train_examples/2))]

        list_neg_anchors = list_neg_anchors + list_all_others_anchors
        random.shuffle(list_neg_anchors)
        neg = list_neg_anchors[:(C.rpn_num_train_examples-len(pos))]

    if (mode == '2CHA'):
        if len(list_pig_anchors) <= C.rpn_num_train_examples/3:
            pos = list_pig_anchors
        else:
            pos = list_pig_anchors[:int((C.rpn_num_train_examples/3))]
        
        if len(list_others_anchors) <= C.rpn_num_train_examples/3:
            neg = list_others_anchors
        else:
            neg = list_others_anchors[:int((C.rpn_num_train_examples/3))]
        
        nb_others = len(neg) 
        neg = neg + list_neg_anchors[:(C.rpn_num_train_examples-nb_others-len(pos))]

        # Fill positives with pig / negatives with background and others
    if (mode == 'PaHNO'):
        if len(list_pig_anchors) <= C.rpn_num_train_examples/3:
            pos = list_pig_anchors
        elif len(list_pig_anchors) > C.rpn_num_train_examples/3:
            pos = list_pig_anchors[:int((C.rpn_num_train_examples/3))]
        
        if len(list_others_anchors) <= C.rpn_num_train_examples/3:
            neg = list_others_anchors
        elif len(list_others_anchors) > C.rpn_num_train_examples/3:
            neg = list_others_anchors[:int((C.rpn_num_train_examples/3))]
        
        nb_others = len(neg) 
        neg = neg + list_neg_anchors[:(C.rpn_num_train_examples-nb_others-len(pos))]

    # Fill positives with pig and others / negatives with background
    if (mode == 'PaHPO' or mode == 'PaCO' ):
        if len(list_pig_anchors) <= C.rpn_num_train_examples/3:
            pos = list_pig_anchors
        elif len(list_pig_anchors) > C.rpn_num_train_examples/3:
            pos = list_pig_anchors[:int((C.rpn_num_train_examples/3))]
        
        if len(list_others_anchors) <= C.rpn_num_train_examples/3:
            pos = pos + list_others_anchors
        elif len(list_others_anchors) > C.rpn_num_train_examples/3:
            pos = pos + list_others_anchors[:int((C.rpn_num_train_examples/3))]
        
        neg = list_neg_anchors[:(C.rpn_num_train_examples-len(pos))]

    for i in range (len(pos)):
        y_is_box_valid[pos[i][0], pos[i][1], pos[i][2]] = 1
        y_rpn_overlap[pos[i][0], pos[i][1], pos[i][2]] = 1
        start = 4 * (pos[i][2])
        y_rpn_regr[pos[i][0], pos[i][1], start:start+4] = pos[i][3]

    for i in range (len(neg)):
        y_is_box_valid[neg[i][0], neg[i][1], neg[i][2]] = 1
        y_rpn_overlap[neg[i][0], neg[i][1], neg[i][2]] = 0
    
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])
    num_neg = len(neg_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
    
    # get list of box matching with gt others
    a = []
    for i in range(len(list_others_anchors)):
        list_others_anchors[i] = (list_others_anchors[i][0],list_others_anchors[i][1],list_others_anchors[i][2])
        box_x1 = list_others_anchors[i][1]*C.rpn_stride - (C.anchor_box_scales[list_others_anchors[i][2]//3]* C.anchor_box_ratios[list_others_anchors[i][2]%3][0])/2
        box_x2 = list_others_anchors[i][1]*C.rpn_stride + (C.anchor_box_scales[list_others_anchors[i][2]//3]* C.anchor_box_ratios[list_others_anchors[i][2]%3][0])/2
        box_y1 = list_others_anchors[i][0]*C.rpn_stride - (C.anchor_box_scales[list_others_anchors[i][2]//3]* C.anchor_box_ratios[list_others_anchors[i][2]%3][1])/2
        box_y2 = list_others_anchors[i][0]*C.rpn_stride + (C.anchor_box_scales[list_others_anchors[i][2]//3]* C.anchor_box_ratios[list_others_anchors[i][2]%3][1])/2
        if box_x1 < 0:
            box_x1 = 0
        if box_x2 > 1280:
            box_x2 = 1280
        if box_y1 < 0:
            box_y1 = 0
        if box_y2 > 720:
            box_y2 = 720
        a.append((int(box_x1),int(box_y1),int(box_x2),int(box_y2)))

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos, a

############################################################
            #Get new image size and augment the image
############################################################

def get_new_img_size(width, height, img_min_side=300):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height

def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img

############################################################
            #Generate the ground_truth anchors
############################################################

def get_anchor_gt(all_img_data, C, img_length_calc_function, mode='train'):
    """ Yield the ground-truth anchors as Y (labels)
        
    Args:
        all_img_data: list(filepath, width, height, list(bboxes))
        C: config
        img_length_calc_function: function to calculate final layer's feature map (of base model) size according to input image size
        mode: 'train' or 'test'; 'train' mode need augmentation

    Returns:
        x_img: image data after resized and scaling (smallest size = 300px)
        Y: [y_rpn_cls, y_rpn_regr]
        img_data_aug: augmented image data (original image with augmentation)
        debug_img: show image for debug
        num_pos: show number of positive anchors for debug
    """ 
    
    while True:
        for img_data in all_img_data:
            try:
                
                # read in image, and optionally add augmentation

                if mode == 'train':
                    img_data_aug, x_img = augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = augment(img_data, C, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                # resize the image so that smalles side is length = 300px
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                debug_img = x_img.copy()
                try:
                    y_rpn_cls, y_rpn_regr, num_pos, list_others = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function,mode)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                # Zero-center by mean pixel, and preprocess image

                x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

                x_img = np.transpose(x_img, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos, list_others

            except Exception as e:
                print(e)
                continue

############################################################
            #Define loss functions for all four outputs
############################################################

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors): 
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        # x is the difference between true value and predicted vaue
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred

        # absolute value of x
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num

def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):
            return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
    return rpn_loss_cls_fixed_num

def class_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
        targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+epsilon))/N
    return ce

def nms_boxes(boxes, probs=None, overlap_thresh=0.5, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    
    if len(boxes) == 0:
        return [],False

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # keep looping while some indexes still remain in the indexes
    # list
    if probs is not None:
        idxs = np.argsort(probs)
    else:
        idxs = np.arange(boxes.shape[0])

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")

    return boxes.tolist(), pick 

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    if len(boxes) == 0:
        return 0,0,False

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]

    return boxes, probs, True


def non_max_suppression_fast_iiou(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    if len(boxes) == 0:
        return 0,0,False

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection
        xx1_min_area = np.minimum(x1[i], x1[idxs[:last]])
        yy1_min_area = np.minimum(y1[i], y1[idxs[:last]])
        xx2_min_area = np.maximum(x2[i], x2[idxs[:last]])
        yy2_min_area= np.maximum(y2[i], y2[idxs[:last]])
        min_area = abs((xx2_min_area-xx1_min_area)*(yy2_min_area-yy1_min_area))

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        iou = area_int/(area_union + 1e-6)

        overlap = iou - ((min_area-area_union)/(min_area+ 1e-6))
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]

    return boxes, probs, True

def apply_regr_np(X, T):
    """Apply regression layer to all anchors in one feature map

    Args:
        X: shape=(4, 18, 25) the current anchor type for all points in the feature map
        T: regression layer shape=(4, 18, 25)

    Returns:
        X: regressed position and size for current anchor
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X
    
def apply_regr(x, y, w, h, tx, ty, tw, th):
    # Apply regression to x, y, w and h
    try:
        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h
def calc_iou1(R, img_data, C, class_mapping):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format

    Args:
        R: bboxes, probs
    """

    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        # gta[bbox_num, 0] = (40 * (600 / 800)) / 16 = int(round(1.875)) = 2 (x in feature map)
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    # R.shape[0]: number of bboxes (=300 from non_max_suppression)
    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        # Iterate through all the ground-truth bboxes to calculate the iou
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])

            # Find out the corresponding ground-truth bbox_num with larget iou
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
        if best_iou < C.classifier_min_overlap:
                continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot code for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    # corresponding labels and corresponding gt bboxes
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs

def calc_iou(R, img_data, C, class_mapping):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format

    Args:
        R: bboxes, probs
    """

    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))
    gtc = []

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        # gta[bbox_num, 0] = (40 * (600 / 800)) / 16 = int(round(1.875)) = 2 (x in feature map)
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))
        gtc.append(bbox['class'])

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only
    overlap_others =  []
    # R.shape[0]: number of bboxes (=300 from non_max_suppression)
    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        best_overlap_others = 0
        # Iterate through all the ground-truth bboxes to calculate the iou
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])

            # Find out the corresponding ground-truth bbox_num with larget iou
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
                
            if (curr_iou > best_overlap_others) and (gtc[bbox_num]=='others'):
                best_overlap_others = curr_iou
        
        overlap_others.append(best_overlap_others)

        if best_iou < C.classifier_min_overlap:
                continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot code for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    # corresponding labels and corresponding gt bboxes
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs, overlap_others

def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9):
    """Convert rpn layer to roi bboxes

    Args: (num_anchors = 9)
        rpn_layer: output layer for rpn classification 
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 18) if resized image is 400 width and 300
        regr_layer: output layer for rpn regression
            shape (1, feature_map.height, feature_map.width, num_anchors)
            Might be (1, 18, 25, 72) if resized image is 400 width and 300
        C: config
        use_regr: Wether to use bboxes regression in rpn
        max_boxes: max bboxes number for non-max-suppression (NMS)
        overlap_thresh: If iou in NMS is larger than this threshold, drop the box

    Returns:
        result: boxes from non-max-suppression (shape=(300, 4))
            boxes: coordinates for bboxes (on the feature map)
    """
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales   # (3 in here)
    anchor_ratios = C.anchor_box_ratios  # (3 in here)

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0

    # A.shape = (4, feature_map.height, feature_map.width, num_anchors) 
    # Might be (4, 18, 25, 18) if resized image is 400 width and 300
    # A is the coordinates for 9 anchors for every point in the feature map 
    # => all 18x25x9=4050 anchors cooridnates
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # anchor_x = (128 * 1) / 16 = 8  => width of current anchor
            # anchor_y = (128 * 2) / 16 = 16 => height of current anchor
            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
            
            # curr_layer: 0~8 (9 anchors)
            # the Kth anchor of all position in the feature map (9th in total)
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4] # shape => (18, 25, 4)
            regr = np.transpose(regr, (2, 0, 1)) # shape => (4, 18, 25)

            # Create 18x25 mesh grid
            # For every point in x, there are all the y points and vice versa
            # X.shape = (18, 25)
            # Y.shape = (18, 25)
            X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

            # Calculate anchor position and size for each feature map point
            A[0, :, :, curr_layer] = X - anchor_x/2 # Top left x coordinate
            A[1, :, :, curr_layer] = Y - anchor_y/2 # Top left y coordinate
            A[2, :, :, curr_layer] = anchor_x       # width of current anchor
            A[3, :, :, curr_layer] = anchor_y       # height of current anchor

            # Apply regression to x, y, w and h if there is rpn regression layer
            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # Avoid width and height exceeding 1
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            # x1, y1 is top left coordinate
            # x2, y2 is bottom right coordinate
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            # Avoid bboxes drawn outside the feature map
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))  # shape=(4050, 4)
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))                   # shape=(4050,)

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Find out the bboxes which is illegal and delete them from bboxes list
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    # Apply non_max_suppression
    # Only extract the bboxes. Don't need rpn probs in the later process
    result,probs, valid = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)

    return result,probs


def get_compo_batch(X2_train, Y1_train, Y2_train,sel_samples_HOEM):
    
    bg_samples = np.where(Y1_train[0, :, -1] == 1)
    
    if len(bg_samples) > 0:
        bg_samples = bg_samples[0].tolist()
    else:
        bg_samples = []
    # get others exemple
    others_samples = np.where(Y1_train[0, :, -2] == 1)
    if len(others_samples) > 0:
        others_samples = others_samples[0].tolist()
    else:
        others_samples = []
    # get pig sample
    pig_samples = np.where(Y1_train[0, :, -3] == 1)
    if len(pig_samples) > 0:
        pig_samples = pig_samples[0].tolist()
    else:
        pig_samples = []
    #print(len(pig_samples),len(others_samples),len(bg_samples))

def get_training_batch_classifier(C, X2_train, Y1_train, Y2_train, IouS_train, mode, overlap_others, hard_anchors_others,  sel_samples_HOEM):

    bg_samples = np.where(Y1_train[0, :, -1] == 1)
    
    if len(bg_samples) > 0:
        bg_samples = bg_samples[0].tolist()
    else:
        bg_samples = []
    # get others exemple
    others_samples = np.where(Y1_train[0, :, -2] == 1)
    if len(others_samples) > 0:
        others_samples = others_samples[0].tolist()
    else:
        others_samples = []
    # get pig sample
    pig_samples = np.where(Y1_train[0, :, -3] == 1)

    if len(pig_samples) > 0:
        pig_samples = pig_samples[0].tolist()
    else:
        pig_samples = []
    
    if (mode == 'HOEM'):
        sel_samples = sel_samples_HOEM
    if (mode == '2CHA'):

        X2_hard_ha = np.zeros((1,len(hard_anchors_others),4))
        Y1_hard_ha = np.zeros((1,len(hard_anchors_others),3))
        Y2_hard_ha = np.zeros((1,len(hard_anchors_others),16))
        
        # make X2, Y1 and Y2 for hard anchors
        for i in range(len(hard_anchors_others)):
            X2_hard_ha[0,i] = ( int (hard_anchors_others[i][0]/C.rpn_stride),
                                int(hard_anchors_others[i][1]/C.rpn_stride),
                                int(int(hard_anchors_others[i][2]/C.rpn_stride)-int (hard_anchors_others[i][0]/C.rpn_stride)),
                                int(int(hard_anchors_others[i][3]/C.rpn_stride)-int(hard_anchors_others[i][1]/C.rpn_stride)))
            Y1_hard_ha[0,i] = (0,1,0)
            Y2_hard_ha[0,i] = np.zeros((16))
        
        ha_samples = np.arange(start=Y1_train.shape[1],stop=Y1_train.shape[1]+len(hard_anchors_others)).tolist()
        # add them to original X2, Y1 and Y2
        
        X2_train = np.concatenate((X2_train,X2_hard_ha),axis=1)
        Y1_train = np.concatenate((Y1_train,Y1_hard_ha),axis=1)
        Y2_train = np.concatenate((Y2_train,Y2_hard_ha),axis=1)
        
        if len(pig_samples) >= C.num_rois/3:
            pig_samples = np.random.choice(pig_samples, int(C.num_rois/3), replace=False).tolist()
        if len(others_samples) >= C.num_rois/3:
            others_samples = np.random.choice(others_samples, int(C.num_rois/3), replace=False).tolist()
        if len(ha_samples) >= C.num_rois/3:
            ha_samples = np.random.choice(ha_samples, int(C.num_rois/3), replace=False).tolist()         
        
        bg_samples = np.random.choice(bg_samples, C.num_rois-len(pig_samples)-len(others_samples)-len(ha_samples), replace=True).tolist()
        sel_samples = pig_samples + others_samples + bg_samples + ha_samples
    
    if (mode == 'P'):
        # Remove examples wich overlap with a gt others if mode is 'P'
        for i in range (len(overlap_others)):
            if (overlap_others[i]>0):
                if i in pig_samples:
                    pig_samples.remove(i)
                if i in others_samples:
                    others_samples.remove(i)
                if i in bg_samples:
                    bg_samples.remove(i)

        # Fill pigs with pigs / background with background
        if len(pig_samples) >= C.num_rois/2:
            pig_samples = np.random.choice(pig_samples, int(C.num_rois/2), replace=False).tolist()
        
        bg_samples = np.random.choice(bg_samples, C.num_rois-len(pig_samples), replace=True).tolist()
        sel_samples = pig_samples + bg_samples

        # Fill pigs with pig / background with background or others
    elif (mode == 'PaO'):
        if len(pig_samples) >= C.num_rois/2:
            pig_samples = np.random.choice(pig_samples, int(C.num_rois/2), replace=False).tolist()

        bg_samples = bg_samples + others_samples
        bg_samples = np.random.choice(bg_samples, C.num_rois-len(pig_samples), replace=True).tolist()
        sel_samples = pig_samples + bg_samples

        # Fill pig with pig / background with background and others
    elif (mode == 'PaHNO') or (mode == 'PaHPO'):
        if len(pig_samples) >= C.num_rois/3:
            pig_samples = np.random.choice(pig_samples, int(C.num_rois/3), replace=False).tolist()
        
        if len(others_samples) >= C.num_rois/3:
            others_samples = np.random.choice(others_samples, int(C.num_rois/3), replace=False).tolist()
        
        bg_samples = np.random.choice(bg_samples, C.num_rois-len(pig_samples)-len(others_samples), replace=True).tolist()
        sel_samples = pig_samples + others_samples + bg_samples
    elif (mode == 'PaCO'):
        if len(pig_samples) >= C.num_rois/3:
            pig_samples = np.random.choice(pig_samples, int(C.num_rois/3), replace=False).tolist()
        
        if len(others_samples) >= C.num_rois/3:
            others_samples = np.random.choice(others_samples, int(C.num_rois/3), replace=False).tolist()
        
        bg_samples = np.random.choice(bg_samples, C.num_rois-len(pig_samples)-len(others_samples), replace=True).tolist()
        sel_samples = pig_samples + others_samples + bg_samples

    X2 = X2_train[:,sel_samples, :]

    if (mode != 'PaCO') :
        Y1_train = Y1_train[:,sel_samples,:]
        Y2_train = Y2_train[:,sel_samples,:]
        Y1 = np.zeros(Y1_train[:,:,0:2].shape, dtype=int)
        Y2 = np.zeros(Y2_train[:,:,0:8].shape)

        for i in range (Y1_train.shape[1]):
            if Y1_train[0,i,0] == 1:
                Y1[0,i,0] = 1
                Y1[0,i,1] = 0
                Y2[:,i,0:4] = Y2_train[:,i,0:4]
                Y2[:,i,4:8] = Y2_train[:,i,8:12]
            else:
                Y1[0,i,0] = 0
                Y1[0,i,1] = 1
                Y2[:,i,0:8] = np.zeros((8))
    else: 
        Y1 = Y1_train[:,sel_samples, :]
        Y2 = Y2_train[:,sel_samples, :]
    return X2, Y1, Y2, sel_samples

def get_testing_batch_classifier(C, X2_test, Y1_test, Y2_test, mode, k):
    
    sel_samples = []
    for i in range(C.num_rois):
        sel_samples.append(k*C.num_rois + i)
    X2 = X2_test[:,sel_samples, :]
    if (mode != 'PaCO') :
        X2_test = X2_test[:,sel_samples,:]
        Y1_test = Y1_test[:,sel_samples,:]
        Y2_test = Y2_test[:,sel_samples,:]
        Y1 = np.zeros(Y1_test[:,:,0:2].shape, dtype=int)
        Y2 = np.zeros(Y2_test[:,:,0:8].shape)

        for i in range (Y1_test.shape[1]):
            if Y1_test[0,i,0] == 1:
                Y1[0,i,0] = 1
                Y1[0,i,1] = 0
                Y2[:,i,0:4] = Y2_test[:,i,0:4]
                Y2[:,i,4:8] = Y2_test[:,i,8:12]
            else:
                Y1[0,i,0] = 0
                Y1[0,i,1] = 1
                Y2[:,i,0:8] = np.zeros((8))
    else: 
        Y1 = Y1_test[:,sel_samples, :]
        Y2 = Y2_test[:,sel_samples, :]
    return X2, Y1, Y2, sel_samples
############################################################
                    #Test all epochs
############################################################
def compare_detection_to_groundtruth(groundtruth_boxes, boxes, probs, iou_min=0.5, score_min=0):
    
    confusion_matrix = np.zeros((3,3))
    match = []
    detection_boxes = []
    for i in range(len(boxes)):
        if probs[i] >= score_min:
            detection_boxes.append(boxes[i])

    for idx in range(len(groundtruth_boxes)):
        groundtruth_boxe = [    groundtruth_boxes[idx]['x1'],
                                groundtruth_boxes[idx]['y1'],
                                groundtruth_boxes[idx]['x2'],
                                groundtruth_boxes[idx]['y2']]
        
        if groundtruth_boxes[idx]['class'] == 'pig':
            groundtruth_classe = 0
        if groundtruth_boxes[idx]['class'] == 'others':
            groundtruth_classe = 1
        if groundtruth_boxes[idx]['class'] == 'bg':
            groundtruth_classe = 2
        # Each detection boxe
        for j in range(len(detection_boxes)):
            detection_boxe = detection_boxes[j]
            detection_classe = 0
            curr_iou = iou(groundtruth_boxe,detection_boxe)
            if curr_iou > iou_min :
                match.append((  groundtruth_classe,
                                idx,
                                detection_classe,
                                j,
                                curr_iou))
    #Remove duplicate detection for groundtruth
    lt = len(match)
    cpt = 0
    while cpt < lt - 1:
        cpt2 = cpt + 1
        while cpt2 < lt:
            if match[cpt][0:2] == match[cpt2][0:2]:
                if match[cpt][4] > match[cpt2][4]:
                    del match[cpt2]
                else:
                    del match[cpt]
                lt = lt - 1
            else: 
                cpt2 = cpt2 + 1
        cpt = cpt + 1
    
    # Remove duplicate groundtruth for detection
    lt = len(match)
    cpt = 0
    while cpt < lt - 1:
        cpt2 = cpt + 1
        while cpt2 < lt:
            if match[cpt][2:4] == match[cpt2][2:4]:
                if match[cpt][4] > match[cpt2][4]:
                    del match[cpt2]
                else:
                    del match[cpt]
                lt = lt - 1
            else: 
                cpt2 = cpt2 + 1
        cpt = cpt + 1
        
    #Record good and missclassification    
    for i in range(len(match)):
        confusion_matrix[match[i][2],match[i][0]] = confusion_matrix[match[i][2],match[i][0]] + 1

    # Record false negative
    # Each groundtruth boxe
    for idx in range(len(groundtruth_boxes)):
        if groundtruth_boxes[idx]['class'] == 'pig':
            groundtruth_classe = 0
        if groundtruth_boxes[idx]['class'] == 'others':
            groundtruth_classe = 1
        if groundtruth_boxes[idx]['class'] == 'bg':
            groundtruth_classe = 2
        found_detection = False
        for j in range(len(match)):
            if idx == match[j][1]:
                found_detection = True
        if found_detection == False:
            confusion_matrix[2, groundtruth_classe] = confusion_matrix[2, groundtruth_classe] + 1
           
    # Each groundtruth boxe
    for i in range(len(detection_boxes)):
        found_detection = False
        #Each match
        for j in range(len(match)):
            if i == match[j][3]:
                found_detection = True
        if found_detection == False:
            confusion_matrix[0,2] = confusion_matrix[0,2] + 1
    return confusion_matrix

def compare_rpn_to_groundtruth(groundtruth_boxes, detection_boxes, num_boxes=300):
    confusion_matrix = np.zeros((2,3))

    for i in range(len(groundtruth_boxes)):
        detected = False
        groundtruth_boxe = [    groundtruth_boxes[i]['x1'],
                                groundtruth_boxes[i]['y1'],
                                groundtruth_boxes[i]['x2'],
                                groundtruth_boxes[i]['y2']]
        groundtruth_classe = groundtruth_boxes[i]['class'] 
        for j in range(len(detection_boxes)):
            curr_iou = iou(groundtruth_boxe,detection_boxes[j])
            if curr_iou >= 0.5:
                detected = True
                break
        if detected == True:
            if groundtruth_classe == 'pig':
                confusion_matrix[0,0] = confusion_matrix[0,0] + 1
            if groundtruth_classe == 'others':
                confusion_matrix[0,1] = confusion_matrix[0,1] + 1
        else:
            if groundtruth_classe == 'pig':
                confusion_matrix[1,0] = confusion_matrix[1,0] + 1
            if groundtruth_classe == 'others':
                confusion_matrix[1,1] = confusion_matrix[1,1] + 1
    
    confusion_matrix[0,2] = num_boxes - confusion_matrix[0,0] - confusion_matrix[0,1] - confusion_matrix[1,0] - confusion_matrix[1,1]
    return confusion_matrix  


def get_detections_boxes(Y, C, class_mapping):
    bboxes = []
    probs = []
    # class_mapping = {v: k for k, v in class_mapping.items()}
    for i in range(len(Y)):
        (x, y, w, h) = Y[i][0]
        cls_name =  np.argmax(Y[i][1])

        (tx, ty, tw, th) = Y[i][2][0:4]
        tx /= C.classifier_regr_std[0]
        ty /= C.classifier_regr_std[1]
        tw /= C.classifier_regr_std[2]
        th /= C.classifier_regr_std[3]
        x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
        x1 = C.rpn_stride*x
        y1 = C.rpn_stride*y
        x2 = C.rpn_stride*(x+w)
        y2 = C.rpn_stride*(y+h)

        if cls_name == 0:
            probs.append(Y[i][1][np.argmax(Y[i][1])])
            bboxes.append([x1,y1,x2,y2])
    
    bboxes = np.asarray(bboxes)
    probs = np.asarray(probs)

    return bboxes, probs

def get_average_precision(all_det,groundtruth_boxes,iou_min):
    
    gt_boxes = []
    
    for i in range(len(groundtruth_boxes)):
        if groundtruth_boxes[i]['class'] == 'pig':
            gt_boxes.append((   groundtruth_boxes[i]['x1'],
                                groundtruth_boxes[i]['y1'],
                                groundtruth_boxes[i]['x2'],
                                groundtruth_boxes[i]['y2']))
    all_det.sort(key=lambda tup: tup[2],reverse=True)
    Tp = 0
    cnt = 0
    Precision_inter = 0
    PR = np.zeros((len(all_det), 4))

    nb_gt = len(gt_boxes)

    for i in range(len(all_det)):
        cnt = cnt + 1
        for j in range(len(gt_boxes)):
            cur_iou = iou(gt_boxes[j],all_det[i][0])
            if cur_iou >= iou_min:
                Tp = Tp + 1
                PR[i][0] = 1
                Precision_inter = Tp / cnt
                del gt_boxes[j]
                break
        
        PR[i][1] = Tp / cnt
        PR[i][2] = Tp / (nb_gt + 1e-16)
        PR[i][3] = Precision_inter

    recall = 0
    tab = []
    for i in range(len(PR)):
        if PR[i][2] > recall:
            tab.append((PR[i][0],PR[i][1],PR[i][2],PR[i][3]))
            recall = PR[i][2]

    tab_final = np.zeros(11)
    max_p = 0

    seuil = 0
    for i in range(len(tab)):
        if (max_p < tab[i][3]):
            max_p = tab[i][3]
        max_tmp = max_p
        
        while(seuil/10 <= tab[i][2]):
            tab_final[seuil] = max_tmp
            max_p = 0
            seuil = seuil + 1

    ap = sum(tab_final)/len(tab_final)

    return ap


def show_rpn_on_image(img,g_boxes,d_boxes,num_boxes):
    img_copy = img.copy()
    for rect in d_boxes[0:num_boxes]:
        color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        p1 = (rect[0],rect[1])
        p2 = (rect[2],rect[3])
        cv2.rectangle(img,p1,p2, color, 2)
    cv2.imshow("detection",img)
    for rect in g_boxes:
        color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        p1 = (rect['x1'],rect['y1'])
        p2 = (rect['x2'],rect['y2'])
        cv2.rectangle(img_copy,p1,p2, color, 2)
    cv2.imshow("groundtruth",img_copy)
    while True:
        k = cv2.waitKey(0)
        if k == 27 or k == 113:         # wait for ESC or q key to exit
            cv2.destroyAllWindows()
            quit()
        if k == 110:
            break

### CLASSIFICATION

def get_groundtruth_rpn(detection_boxes,groundtruth_boxes):
    """ Return a list wich contain the real class of each output of the rpn"""
    classes = []
    for i in range(len(detection_boxes)):
        iou_max = 0
        classe = "neg"
        for j in range(len(groundtruth_boxes)):
            groundtruth_boxe = [    groundtruth_boxes[j]['x1'],
                                    groundtruth_boxes[j]['y1'],
                                    groundtruth_boxes[j]['x2'],
                                    groundtruth_boxes[j]['y2']]
            curr_iou = iou(groundtruth_boxe,detection_boxes[i])
            if curr_iou > iou_max :
                iou_max = curr_iou
                classe = groundtruth_boxes[j]['class']
        classes.append((classe,iou_max))
    return classes

def show_class_on_image(img,R_resized,class_mapping,R_classification,R_groundthruth,iou_min,score_min):
    iou_min = iou_min/100
    score_min = score_min/100

    fig, axs = plt.subplots(int(len(R_resized)/10),10)

    row = 0
    col=0
    for i in range(len(R_resized)):
        if (i % 10)==0 and i >0:
            row = row + 1
            col = 0
        if R_groundthruth[i][0] == "pig" and R_groundthruth[i][1] >= iou_min:
            groundtruth_classe = "pig"
        else:
            groundtruth_classe = "bg"
        if np.max(R_classification[i]) >= score_min and class_mapping[np.argmax(R_classification[i])] == "pig":
            classification_classe = "pig"
        else:
            classification_classe = "bg"
        
        legende = "{} / {}".format(    groundtruth_classe,
                                        classification_classe)
        sub_img = img[R_resized[i][1]:R_resized[i][3],R_resized[i][0]:R_resized[i][2]]
        
        if (groundtruth_classe==classification_classe):
            color='g'
        else:
            color='r'

        axs[row,col].set_title(legende,color=color)
        axs[row,col].set_yticklabels([])
        axs[row,col].set_xticklabels([])
        axs[row,col].imshow(cv2.cvtColor(sub_img,cv2.COLOR_BGR2RGB))
        col = col + 1
    plt.show()

def compare_classification_to_groundtruth(class_mapping, class_boxes, groundtruth_boxes, iou_min, score_min):
    # Groundtruth data : (classe,iou)
    # classification_boxes : [score_C1,score_C2,....,score_BG]
    iou_min = iou_min/100
    score_min = score_min/100
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(groundtruth_boxes)):
        groundtruth_classe = "bg"
        # Si la box a ete classifie pig avec un iou minimum la classe reelle  est "pig" 
        if groundtruth_boxes[i][0] == "pig" and groundtruth_boxes[i][1] >= iou_min:
            groundtruth_classe = "pig"
        # Si la box a ete classifie "pig" avec un score min 
        if np.max(class_boxes[i]) >= score_min and class_mapping[np.argmax(class_boxes[i])] == "pig":
            if groundtruth_classe == "pig":
                tp = tp + 1
            if groundtruth_classe == "bg":
                fp = fp + 1
        else:
            if groundtruth_classe == "pig":
                fn = fn + 1
            if groundtruth_classe == "bg":
                tn = tn + 1
    
    return tp, fp, fn, tn