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
from new_utils import *
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Argument')
parser.add_argument('--label', type=str,
                    help='path to csv file')
parser.add_argument('--images_path', type=str,
			help='images folder')

args = parser.parse_args()
CSV_FILE=args.label
images_path = args.images_path
nb_total_images = 0
nb_total_pigs = 0
nb_total_others = 0

nb_images_with_only_pigs = 0
nb_pigs_in_images_with_only_pigs = 0

nb_images_with_only_others = 0
nb_others_in_images_with_only_others = 0

nb_images_with_both = 0
nb_pigs_in_images_with_both = 0
nb_others_in_images_with_both = 0

# with open(CSV_FILE, 'r') as csvFile:
#     reader = csv.reader(csvFile)
#     list_r = list(reader)
#     filename = "none"
#     data = []
#     i = 1
#     while (i < len(list_r)):
#         if list_r[i][0]!=filename:
#             data.append((list_r[i][0],[],[]))
#             filename = list_r[i][0]
#             nb_total_images = nb_total_images + 1
        
#         if list_r[i][3] == "pig":
#             data[len(data)-1][1].append(list_r[i][4:8])
#             nb_total_pigs = nb_total_pigs + 1 

#         elif list_r[i][3] == "others":
#             data[len(data)-1][2].append(list_r[i][4:8])
#             nb_total_others = nb_total_others + 1
        
#         i = i + 1
    
#     data.sort()

#     for i in range(len(data)):
#         if (len(data[i][1]) > 0) and (len(data[i][2]) == 0):
#             nb_images_with_only_pigs = nb_images_with_only_pigs + 1
#             nb_pigs_in_images_with_only_pigs = nb_pigs_in_images_with_only_pigs + len(data[i][1])


#         if (len(data[i][1]) == 0) and (len(data[i][2]) > 0):
#             nb_images_with_only_others = nb_images_with_only_others + 1
#             nb_others_in_images_with_only_others =  nb_others_in_images_with_only_others + len(data[i][2])

#         if (len(data[i][1]) > 0) and (len(data[i][2]) > 0):
#             nb_images_with_both = nb_images_with_both + 1
#             nb_pigs_in_images_with_both = nb_pigs_in_images_with_both + len(data[i][1])
#             nb_others_in_images_with_both =  nb_others_in_images_with_both + len(data[i][2])
#             # path = "/home/mattina/Desktop/Videos_environnement_porcin/abattoir/Video_Abattoir_24-09-2019-09-16-37/BDD/only_others/"+data[i][0][0:40]
#             # os.remove(path+".xml")
#             # os.remove(path+".jpg")


# print ("nb_total_images = ", nb_total_images)
# print ("    nb_total_pigs = ", nb_total_pigs)
# print ("    nb_total_others = ", nb_total_others)

# print ("nb_images_with_only_pigs = ", nb_images_with_only_pigs)
# print ("    nb_pigs_in_images_with_only_pigs = ", nb_pigs_in_images_with_only_pigs)

# print ("nb_images_with_only_others = ", nb_images_with_only_others)
# print ("    nb_others_in_images_with_only_others = ", nb_others_in_images_with_only_others)

# print ("nb_images_with_both = ", nb_images_with_both)
# print ("    nb_pigs_in_images_with_both = ", nb_pigs_in_images_with_both)
# print ("    nb_others_in_images_with_both = ", nb_others_in_images_with_both)



train_imgs, classes_count_train, class_mapping = get_data(CSV_FILE,images_path)


list_of_width = []
list_of_height = []
list_of_ratio = []
list_of_iou = []

# for i in range(len(train_imgs)):
#     for j in range (len(train_imgs[i]['bboxes'])):
#         if (train_imgs[i]['bboxes'][j]['class']=='pig'):
#             x1 = train_imgs[i]['bboxes'][j]['x1']
#             x2 = train_imgs[i]['bboxes'][j]['x2']
#             y1 = train_imgs[i]['bboxes'][j]['y1']
#             y2 = train_imgs[i]['bboxes'][j]['y2']
#             for k in range (j+1,len(train_imgs[i]['bboxes'])):
#                 if (train_imgs[i]['bboxes'][k]['class']=='pig'):
#                     x1_k = train_imgs[i]['bboxes'][k]['x1']
#                     x2_k = train_imgs[i]['bboxes'][k]['x2']
#                     y1_k = train_imgs[i]['bboxes'][k]['y1']
#                     y2_k = train_imgs[i]['bboxes'][k]['y2']
#                     curr_iou = iou([x1,y1,x2,y2],[x1_k,y1_k,x2_k,y2_k])
#                     if curr_iou > 0:
#                         list_of_iou.append(curr_iou)
#                     if curr_iou > 0.6:
#                         print(curr_iou,train_imgs[i]['filepath'])

#             width =  x2 - x1     
#             height = y2 - y1 
#             # if (width>500):
#             #     print(train_imgs[i]['filepath']) 
#             list_of_width.append(width)
#             list_of_height.append(height)
#             list_of_ratio.append(height/width)


# sizes
# plt.hist2d(list_of_width, list_of_height, bins=(80,45), cmap=plt.cm.BuPu)
# plt.colorbar()
# plt.ylabel("height")
# plt.xlabel("width")
# plt.show()

#ratios
# plt.hist(list_of_ratio,bins=20,color=('purple'))
# plt.xlabel("width")
# plt.show()

# #ious
# plt.hist(list_of_iou,bins=100,color=('purple'))
# plt.xlabel("ious")
# plt.show()

def nms_boxes(boxes):

    lt = len(boxes)
    cpt = 0
    while cpt < lt - 1:
        cpt2 = cpt + 1
        while cpt2 < lt:
            curr_iou = iou([boxes[cpt][0], boxes[cpt][1], boxes[cpt][2], boxes[cpt][3]], [boxes[cpt2][0], boxes[cpt2][1], boxes[cpt2][2], boxes[cpt2][3]])
            if curr_iou >= 0.7:
                del boxes[cpt2]
                lt = lt - 1
            else: 
                cpt2 = cpt2 + 1
        cpt = cpt + 1
    
    return len(boxes)

anchor_ratios = [[1, 1], [1, 2], [2, 1]]
anchor_sizes = [100, 175.0, 350.0]


list_of_iou_pig = []
list_of_iou_others = []
list_of_iou_neg = []

list_of_iou_pig_anms = []
list_of_iou_others_anms = []


output_width = 80
output_height = 45


scale = 16
cnt = 0
for i in range(len(train_imgs)):
    nb_iou_pigs = 0
    nb_iou_others = 0
    nb_iou_neg = 0
    list_of_boxs_pig = []
    list_of_boxs_others = []
    
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

                        best_iou_for_anchor = 0.0
                        best_class_for_anchor = 'none'
                        cnt = cnt + 1
                        for bbox_num in range (len(train_imgs[i]['bboxes'])):
                            x1_box = train_imgs[i]['bboxes'][bbox_num]['x1']
                            x2_box = train_imgs[i]['bboxes'][bbox_num]['x2']
                            y1_box = train_imgs[i]['bboxes'][bbox_num]['y1']
                            y2_box = train_imgs[i]['bboxes'][bbox_num]['y2']

                            curr_iou = iou([x1_box, y1_box, x2_box, y2_box], [x1_anc, y1_anc, x2_anc, y2_anc])

                            if  curr_iou > best_iou_for_anchor :
                                best_iou_for_anchor = curr_iou
                                best_class_for_anchor = train_imgs[i]['bboxes'][bbox_num]['class']
                        
                        if best_iou_for_anchor > 0.5:
                            if best_class_for_anchor == 'pig':
                                nb_iou_pigs = nb_iou_pigs + 1
                                list_of_boxs_pig.append((x1_anc,y1_anc,x2_anc,y2_anc))
                            if best_class_for_anchor == 'others':
                                nb_iou_others = nb_iou_others + 1
                                list_of_boxs_others.append((x1_anc,y1_anc,x2_anc,y2_anc))
                        else:
                            nb_iou_neg = nb_iou_neg + 1
                        

    if nb_iou_pigs > 0 or nb_iou_pigs == 0:
        print(i, train_imgs[i]['filepath'],nb_iou_pigs,nb_iou_others,nb_iou_neg)


    list_of_iou_pig_anms.append(nms_boxes(list_of_boxs_pig))
    list_of_iou_others_anms.append(nms_boxes(list_of_boxs_others))

    list_of_iou_pig.append(nb_iou_pigs)
    list_of_iou_others.append(nb_iou_others)
    list_of_iou_neg.append(nb_iou_neg)


# iou rpn
print("nb anchors : ",cnt)
print("mean pig : ",sum(list_of_iou_pig)/len(train_imgs))
print("mean others : ",sum(list_of_iou_others)/len(train_imgs))
print("mean neg : ",sum(list_of_iou_neg)/len(train_imgs))

print("mean pig anms: ",sum(list_of_iou_pig_anms)/len(train_imgs))
print("mean others anms : ",sum(list_of_iou_others_anms)/len(train_imgs))

fig, axs = plt.subplots(1, 3)

axs[0].set_title('Overlap Pig',color='black')
axs[1].set_title('Overlap Others',color='black')
axs[2].set_title('Overlap Negative',color='black')


axs[0].plot(    np.arange(0,len(train_imgs)),
                list_of_iou_pig,
                color='g')

axs[1].plot(   np.arange(0,len(train_imgs)),
                list_of_iou_others,
                color='y')

axs[2].plot(   np.arange(0,len(train_imgs)),
                list_of_iou_neg,
                color='r')

#plt.hist(list_of_iou_pig,bins=100,color=('purple'))
#plt.hist(list_of_iou_others,bins=100,color=('purple'))
plt.show()