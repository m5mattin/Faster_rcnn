from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import argparse

def get_rpn_values(c):
    
    rpn_nms_range = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    max_boxes_range = [50,100,150,200,250,300,350,400,450,500,550,600]
    
    recall_per_nms = []
    precision_per_nms = []
    recall_per_boxes = []
    precision_per_boxes = []

    for i in range(len(rpn_nms_range)):
        j = i*len(max_boxes_range)+0
        TP = c[0,0,j]
        FN = c[1,0,j]
        FP = c[0,1,j] + c[0,2,j]
        recall_per_nms.append( TP / (TP+FN+1e-16))
        precision_per_nms.append( TP / (TP+FP+1e-16))

    for i in range(len(max_boxes_range)):
        j = 0 + i
        TP = c[0,0,j]
        FN = c[1,0,j]
        FP = c[0,1,j] + c[0,2,j]
        recall_per_boxes.append( TP / (TP+FN+1e-16))
        precision_per_boxes.append( TP / (TP+FP+1e-16))

    return recall_per_nms, precision_per_nms, recall_per_boxes, precision_per_boxes

def get_detect_values(c):
    
    detect_nms_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    detect_score_range = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    
    recall_per_nms = []
    precision_per_nms = []
    recall_per_score = []
    precision_per_score = []
    f1_per_nms = []
    f1_per_score = []

    for i in range(len(detect_nms_range)):
        j = (i*len(detect_score_range))+0
        
        TP = c[0,0,j]
        FN = c[1,0,j] + c[2,0,j]
        FP = c[0,1,j] + c[0,2,j]
        
        recall = TP / (TP+FN+1e-16)
        precision = TP / (TP+FP+1e-16)
        f1 = 2*recall*precision/(recall+precision+1e-16)

        recall_per_nms.append(recall )
        precision_per_nms.append( precision )
        f1_per_nms.append(f1)

    for i in range(len(detect_score_range)):
        j = (3*len(detect_score_range)) + i
        
        TP = c[0,0,j]
        FN = c[1,0,j] + c[2,0,j]
        FP = c[0,1,j] + c[0,2,j]

        recall = TP / (TP+FN)
        precision = TP / (TP+FP+1e-16)
        f1 = 2*recall*precision/(recall+precision+1e-16)

        recall_per_score.append(recall )
        precision_per_score.append( precision )
        f1_per_score.append(f1)

    return recall_per_nms, precision_per_nms, f1_per_nms, recall_per_score, precision_per_score, f1_per_score

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
parser = argparse.ArgumentParser(description='Argument')

parser.add_argument('-P','--P', action='store_true')
parser.add_argument('-PaO','--PaO', action='store_true')
parser.add_argument('-PaCO','--PaCO', action='store_true')
parser.add_argument('-HA','--HA', action='store_true')
parser.add_argument('-HOEM','--HOEM', action='store_true')


args = parser.parse_args()
show_P = args.P
show_2C = args.PaO
show_3C = args.PaCO
show_HA = args.HA
show_HOEM = args.HOEM

rpn_nms_range = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95] #10
max_boxes_range = [50,100,150,200,250,300,350,400,450,500,550,600] #12

detect_nms_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #9
detect_score_range = [0.5,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]


fig, axs = plt.subplots(1, 4)
axs[0].set_title('Rpn : NMS',color='black')
axs[1].set_title('Rpn : Max boxes',color='black')
axs[2].set_title('Detection : NMS',color='black')
axs[3].set_title('Detection : Score',color='black')

if show_P:
    rpn_P= np.load('../models/P/rpn.npy')
    detect_P = np.load('../models/P/detect.npy')
    rpn_recall_per_nms, rpn_precision_per_nms, rpn_recall_per_boxes, rpn_precision_per_boxes = get_rpn_values(rpn_P)
    d_recall_per_nms, d_precision_per_nms, d_f1_per_nms, d_recall_per_score, d_precision_per_score, d_f1_per_score = get_detect_values(detect_P)

    axs[0].plot(    rpn_nms_range,
                    rpn_recall_per_nms,
                    label='P recall',
                    color=colors['black'])
    axs[0].plot(    rpn_nms_range,
                    rpn_precision_per_nms,
                    label='P precision',
                    color=colors['gray'])
    axs[1].plot(    max_boxes_range,
                    rpn_recall_per_boxes,
                    label='P recall',
                    color=colors['black'])
    axs[1].plot(    max_boxes_range,
                    rpn_precision_per_boxes,
                    label='P precision',
                    color=colors['gray'])
    axs[2].plot(    detect_nms_range,
                    d_f1_per_nms,
                    label='P f1 score',
                    color=colors['black'])
    axs[3].plot(    detect_score_range,
                    d_f1_per_score,
                    label='P f1 score',
                    color=colors['black'])
if show_2C:
    rpn_2C= np.load('../models/PaO/rpn.npy')
    detect_2C = np.load('../models/PaO/detect.npy')
    rpn_recall_per_nms, rpn_precision_per_nms, rpn_recall_per_boxes, rpn_precision_per_boxes = get_rpn_values(rpn_2C)
    d_recall_per_nms, d_precision_per_nms, d_f1_per_nms, d_recall_per_score, d_precision_per_score, d_f1_per_score = get_detect_values(detect_2C)

    axs[0].plot(    rpn_nms_range,
                    rpn_recall_per_nms,
                    label='2C recall',
                    color=colors['darkred'])
    axs[0].plot(    rpn_nms_range,
                    rpn_precision_per_nms,
                    label='2C precision',
                    color=colors['red'])

    axs[1].plot(    max_boxes_range,
                    rpn_recall_per_boxes,
                    label='2C recall',
                    color=colors['darkred'])
    axs[1].plot(    max_boxes_range,
                    rpn_precision_per_boxes,
                    label='2C precision',
                    color=colors['red'])

    # axs[2].plot(    detect_nms_range,
    #                 d_recall_per_nms,
    #                 label='2C recall',
    #                 color=colors['darkred'])
    # axs[2].plot(    detect_nms_range,
    #                 d_precision_per_nms,
    #                 label='2C precision',
    #                 color=colors['red'])
    axs[2].plot(    detect_nms_range,
                    d_f1_per_nms,
                    label='2C f1 score',
                    color=colors['darkred'])
    
    # axs[3].plot(    detect_score_range,
    #                 d_recall_per_score,
    #                 label='2C recall',
    #                 color=colors['darkred'])
    # axs[3].plot(    detect_score_range,
    #                 d_precision_per_score,
    #                 label='2C precision',
    #                 color=colors['red'])
    axs[3].plot(    detect_score_range,
                    d_f1_per_score,
                    label='2C f1 score',
                    color=colors['darkred'])
if show_3C:
    rpn_3C= np.load('../models/PaCO/rpn.npy')
    detect_3C = np.load('../models/PaCO/detect.npy')
    rpn_recall_per_nms, rpn_precision_per_nms, rpn_recall_per_boxes, rpn_precision_per_boxes = get_rpn_values(rpn_3C)
    d_recall_per_nms, d_precision_per_nms, d_f1_per_nms, d_recall_per_score, d_precision_per_score, d_f1_per_score = get_detect_values(detect_3C)

    axs[0].plot(    rpn_nms_range,
                    rpn_recall_per_nms,
                    label='3C recall',
                    color=colors['darkblue'])
    axs[0].plot(    rpn_nms_range,
                    rpn_precision_per_nms,
                    label='3C precision',
                    color=colors['royalblue'])
    axs[1].plot(    max_boxes_range,
                    rpn_recall_per_boxes,
                    label='3C recall',
                    color=colors['darkblue'])
    axs[1].plot(    max_boxes_range,
                    rpn_precision_per_boxes,
                    label='3C precision',
                    color=colors['royalblue'])
    axs[2].plot(    detect_nms_range,
                    d_f1_per_nms,
                    label='3C f1 score',
                    color=colors['darkblue'])
    axs[3].plot(    detect_score_range,
                    d_f1_per_score,
                    label='3C f1 score',
                    color=colors['darkblue'])
if show_HOEM:

    rpn_HOEM= np.load('../models/HOEM/rpn.npy')
    detect_HOEM = np.load('../models/HOEM/detect.npy')
    rpn_recall_per_nms, rpn_precision_per_nms, rpn_recall_per_boxes, rpn_precision_per_boxes = get_rpn_values(rpn_HOEM)
    d_recall_per_nms, d_precision_per_nms, d_f1_per_nms, d_recall_per_score, d_precision_per_score, d_f1_per_score = get_detect_values(detect_HOEM)

    axs[0].plot(    rpn_nms_range,
                    rpn_recall_per_nms,
                    label='HOEM recall',
                    color=colors['darkorange'])
    axs[0].plot(    rpn_nms_range,
                    rpn_precision_per_nms,
                    label='HOEM precision',
                    color=colors['orange'])
    axs[1].plot(    max_boxes_range,
                    rpn_recall_per_boxes,
                    label='HOEM recall',
                    color=colors['darkorange'])
    axs[1].plot(    max_boxes_range,
                    rpn_precision_per_boxes,
                    label='HOEM precision',
                    color=colors['orange'])

    # axs[2].plot(    detect_nms_range,
    #                 d_recall_per_nms,
    #                 label='HOEM recall',
    #                 color=colors['darkorange'])
    # axs[2].plot(    detect_nms_range,
    #                 d_precision_per_nms,
    #                 label='HOEM precision',
    #                 color=colors['orange'])
    axs[2].plot(    detect_nms_range,
                    d_f1_per_nms,
                    label='HOEM f1 score',
                    color=colors['darkorange'])

    # axs[3].plot(    detect_score_range,
    #                 d_recall_per_score,
    #                 label='HOEM recall',
    #                 color=colors['darkorange'])
    # axs[3].plot(    detect_score_range,
    #                 d_precision_per_score,
    #                 label='HOEM precision',
    #             color=colors['orange'])
    axs[3].plot(    detect_score_range,
                    d_f1_per_score,
                    label='HOEM f1 score',
                    color=colors['darkorange'])
if show_HA:
    rpn_HA= np.load('../models/2CHA/rpn.npy')
    detect_HA = np.load('../models/2CHA/detect.npy')
    rpn_recall_per_nms, rpn_precision_per_nms, rpn_recall_per_boxes, rpn_precision_per_boxes = get_rpn_values(rpn_HA)
    d_recall_per_nms, d_precision_per_nms, d_f1_per_nms, d_recall_per_score, d_precision_per_score, d_f1_per_score = get_detect_values(detect_HA)

    axs[0].plot(    rpn_nms_range,
                    rpn_recall_per_nms,
                    label='HA recall',
                    color=colors['darkgreen'])
    axs[0].plot(    rpn_nms_range,
                    rpn_precision_per_nms,
                    label='HA precision',
                    color=colors['lime'])
    axs[1].plot(    max_boxes_range,
                    rpn_recall_per_boxes,
                    label='HA recall',
                    color=colors['darkgreen'])
    axs[1].plot(    max_boxes_range,
                    rpn_precision_per_boxes,
                    label='HA precision',
                    color=colors['lime'])

    # axs[2].plot(    detect_nms_range,
    #                 d_recall_per_nms,
    #                 label='HA recall',
    #                 color=colors['darkgreen'])
    # axs[2].plot(    detect_nms_range,
    #                 d_precision_per_nms,
    #                 label='HA precision',
    #                 color=colors['lime'])
    axs[2].plot(    detect_nms_range,
                    d_f1_per_nms,
                    label='HA f1 score',
                    color=colors['darkgreen'])

    # axs[3].plot(    detect_score_range,
    #                 d_recall_per_score,
    #                 label='HA recall',
    #                 color=colors['darkgreen'])
    # axs[3].plot(    detect_score_range,
    #                 d_precision_per_score,
    #                 label='HA precision',
    #                 color=colors['lime'])
    axs[3].plot(    detect_score_range,
                    d_f1_per_score,
                    label='HA f1 score',
                    color=colors['darkgreen'])

axs[0].legend(loc='best')
axs[1].legend(loc='best')
axs[2].legend(loc='best')
axs[3].legend(loc='best')
axs[0].set_ylim(0,1.05)
axs[1].set_ylim(0,1.05)
axs[2].set_ylim(0,1.05)
axs[3].set_ylim(0,1.05)

plt.show()