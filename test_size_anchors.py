""" 
Input : CSV file with format "filename,width,height,class,x1,y1,x2,y2"
Output :- nb images 
            - nb pigs
            - nb _others
        - nb images with only pigs
            - nb pigs in image with only pigs
        - nb images with only others
            - nb others in image with only others
        - nb images with pigs and others
            - nb pigs in image with pigs and others
            - nb others in image with pigs and others 
"""

import argparse
import csv
import os
from utils import *
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Argument')
parser.add_argument('--label', type=str,
                    help='path to csv file')
parser.add_argument('--images_path', type=str,
			help='images folder')

args = parser.parse_args()
CSV_FILE=args.label
images_path = args.images_path


m_sizes = [100]
scales = [1.5,1.75,2]
# m_ratios = [1.25]
m_ratios = [1.25,1.5,1.75,2]

list_of_iou_pig = []
list_of_iou_others = []
list_of_iou_neg = []

output_width = 80
output_height = 45


scale = 16
train_imgs, classes_count_train, class_mapping = get_data(CSV_FILE,images_path)

for m in range(len(m_ratios)):
    for n in range(len(m_sizes)):
        for o in range(len(scales)):
            nb_pigs = 0
            nb_others = 0
            nb_neg = 0

            base_size = m_sizes[n]
            base_ratios = m_ratios[m]
            delta = scales[o]

            anchor_sizes = [base_size, base_size*delta, base_size*delta*2]
            anchor_ratios = [[1, 1], [1, base_ratios], [base_ratios, 1]]
            
            for i in range(len(train_imgs)):
                nb_iou_pigs = 0
                nb_iou_others = 0
                nb_iou_neg = 0
                
                for anchor_size_idx in range(len(anchor_sizes)):
                        for anchor_ratio_idx in range(len(anchor_ratios)):
                            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
                            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
                            
                            for ix in range(output_width):	
                                            
                                # x-coordinates of the current anchor box	
                                x1_anc = scale * ix - anchor_x / 2
                                x2_anc = scale * ix + anchor_x / 2
                                
                                # ignore boxes that go across image boundaries					
                                if x1_anc < 0 or x2_anc > (output_width*scale):
                                    continue
                                    
                                for jy in range(output_height):
                                        
                                    # y-coordinates of the current anchor box
                                    y1_anc = scale * jy - anchor_y / 2
                                    y2_anc = scale *jy + anchor_y / 2

                                    # ignore boxes that go across image boundaries
                                    if y1_anc < 0 or y2_anc > (output_height*scale):
                                        continue

                                    best_iou_for_anchor = 0
                                    best_class_for_anchor = 'none'
                                    
                                    for bbox_num in range (len(train_imgs[i]['bboxes'])):
                                        x1_box = train_imgs[i]['bboxes'][bbox_num]['x1']
                                        x2_box = train_imgs[i]['bboxes'][bbox_num]['x2']
                                        y1_box = train_imgs[i]['bboxes'][bbox_num]['y1']
                                        y2_box = train_imgs[i]['bboxes'][bbox_num]['y2']

                                        curr_iou = iou([x1_box, y1_box, x2_box, y2_box], [x1_anc, y1_anc, x2_anc, y2_anc])
            
                                        if  curr_iou > best_iou_for_anchor :
                                            best_iou_for_anchor = curr_iou
                                            best_class_for_anchor = train_imgs[i]['bboxes'][bbox_num]['class']
                                    
                                    if best_iou_for_anchor >= 0.7:
                                        if best_class_for_anchor == 'pig':
                                            nb_iou_pigs = nb_iou_pigs + 1
                                        if best_class_for_anchor == 'others':
                                            nb_iou_others = nb_iou_others + 1
                                    else:
                                        nb_iou_neg = nb_iou_neg + 1
                nb_pigs = nb_pigs + nb_iou_pigs
                nb_others = nb_others + nb_iou_others
                nb_neg = nb_neg + nb_iou_neg

                list_of_iou_pig.append(nb_iou_pigs)
                list_of_iou_others.append(nb_iou_others)
                list_of_iou_neg.append(nb_iou_neg)
            
            print(anchor_sizes,anchor_ratios,nb_pigs,nb_others,nb_neg)

# iou rpn
# print(list_of_iou_pig)
# print(list_of_iou_others)
# print(list_of_iou_neg)
# plt.hist(list_of_iou_pig,bins=100,color=('purple'))
# plt.show()