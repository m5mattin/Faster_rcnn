

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from utils import *

# a = (390,182,702,457)
# b = (455,222,773,537)

a = (447,4,618,237)
b = (447,109,562,237)

iou = iou(a,b)
iiou = iiou(a,b)

print("iou : ",iou)
print("iiou : ",iiou)