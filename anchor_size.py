import numpy as np 
import cv2

img = cv2.imread("../img.jpg")

for i in range(16,1280,16):
    for j in range (16,720,16):
        cv2.circle(img, (i,j), 3, (255,0,0),-1)

scales = [100,200,400]
# scales = [64, 128 , 256]
#scales = [128, 256 , 512]
ratios = [[1,1],[1,1.25],[1.25,1]]

for k in range (len(scales)): 
    for l in range (len(ratios)):
        c = (640,360)
        x1 = int(c[0] - (scales[k]*ratios[l][0])/2)
        y1 = int(c[1] - (scales[k]*ratios[l][1])/2)
        x2 = int(c[0] + (scales[k]*ratios[l][0])/2)
        y2 = int(c[1] + (scales[k]*ratios[l][1])/2)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

cv2.circle(img, (640,360), 3, (0,255,0),-1)
cv2.imshow("img",img)

while True:
    k = cv2.waitKey(0)
    if k == 27 or k == 113:         # wait for ESC or q key to exit
        cv2.destroyAllWindows()
        quit()
    if k == 110:
        break