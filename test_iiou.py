
import numpy as np 
from new_utils import *

a = [0,0,2,4]
b = [0,3,2,4]

print(iou(a,b))
print(iiou(a,b))

a = [0,0,4,4]
b = [0,2,2,4]

print(iou(a,b))
print(iiou(a,b))

a = [0,0,2,2]
b = [1,1,3,3]

print(iou(a,b))
print(iiou(a,b))