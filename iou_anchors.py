import numpy as np 
import cv2
from utils import *

img = cv2.imread("../img3.jpg")

# for i in range(16,1280,16):
#     for j in range (16,720,16):
#         cv2.circle(img, (i,j), 3, (0,0,0),-1)
print(img.shape)

#scales = [150,200,250]
#scales = [64, 128 , 256]
scales = [128, 256 , 512]
ratios = [[1,1],[1,2],[2,1]]

# #img3
pigs = [[718,1,846,158],[482,14,640,334]]
#others = [[300,314,637,672],[536,288,601,467],[500,456,647,592],[291,488,399,675],[340,322,491,506],[370,469,569,629],[438,595,599,689]]
pigs = [[718,1,846,158],[482,14,640,334],[300,314,637,672],[536,288,601,467],[500,456,647,592],[291,488,399,675],[340,322,491,506],[370,469,569,629],[438,595,599,689]]
#img2
#pigs = [[325,249,514,720],[431,46,623,456]]
# others=[]

# for i in range(len(pigs)):
#     cv2.rectangle(img,(pigs[i][0],pigs[i][1]),(pigs[i][2],pigs[i][3]),(0,255,0),3)

# for i in range(len(others)):
#     cv2.rectangle(img,(others[i][0],others[i][1]),(others[i][2],others[i][3]),(255,255,0),3)

pos = []
neg = []
neutral = []
cpt = 0
for x in range(0,1280,16):
    for y in range (0,720,16):
        for k in range (len(scales)): 
            for l in range (len(ratios)):
                cpt = cpt + 1
                c = (x,y)
                x1 = int(c[0] - (scales[k]*ratios[l][0])/2)
                y1 = int(c[1] - (scales[k]*ratios[l][1])/2)
                x2 = int(c[0] + (scales[k]*ratios[l][0])/2)
                y2 = int(c[1] + (scales[k]*ratios[l][1])/2)
                
                overlay_pig = [0,0]

                for i in range (len(pigs)):
                    curr_iou_with_pig  = iou((x1,y1,x2,y2),(pigs[i][0],pigs[i][1],pigs[i][2],pigs[i][3]))
                    if  curr_iou_with_pig >= 0.7:
                       overlay_pig[0] = overlay_pig[0] + 1
                    elif curr_iou_with_pig <= 0.3:
                        overlay_pig[1] = overlay_pig[1] + 1

                if (overlay_pig[0] > 0):
                    pos.append([x1,y1,x2,y2])
                elif (overlay_pig[1] > 0):
                    neg.append([x1,y1,x2,y2])
                else:
                    print("mo")
                    neutral.append([x1,y1,x2,y2])

for i in range (len(pos)):
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.rectangle(img,(pos[i][0],pos[i][1]),(pos[i][2],pos[i][3]),color,3)
print("cpt : {}, pos : {}, neg : {}, neutral : {}".format(cpt,len(pos),len(neg),len(neutral)))

# for i in range (10):
#     ne = random.choice(neg)
#     color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
#     cv2.rectangle(img,(ne[0],ne[1]),(ne[2],ne[3]),color,3)

# for i in range (10):
#     n = random.choice(neutral)
#     color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
#     cv2.rectangle(img,(n[0],n[1]),(n[2],n[3]),color,3)


cv2.imshow("img",img)

while True:
    k = cv2.waitKey(0)
    if k == 27 or k == 113:         # wait for ESC or q key to exit
        cv2.destroyAllWindows()
        quit()
    if k == 110:
        break