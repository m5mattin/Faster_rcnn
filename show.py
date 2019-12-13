from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Load the records
record_train = pd.read_csv("../record_train.csv")
record_test = pd.read_csv("../record_test.csv")

r_epochs = len(record_train)

fig, axs = plt.subplots(1, 3)

axs[0].set_title('rpn',color='black')
axs[1].set_title('Classification',color='black')
axs[2].set_title('Total loss',color='black')

### RPN
axs[0].plot(    np.arange(0, r_epochs), 
                record_train['mean_overlapping_bboxes_pig']/100,
                label='train set: mean_overlapping_bboxes_pig/100',
                color=colors['darkblue'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['loss_rpn_cls'],
                label='train set: loss_rpn_cls',
                color=colors['darkgreen'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['loss_rpn_regr'],
                label='train set: loss_rpn_regr',
                color=colors['limegreen'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['mean_overlapping_bboxes_pig']/100,
                label='val set: mean_overlapping_bboxes_pigs/100',
                color=colors['darkred'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['loss_rpn_cls'],
                label='val set: loss_rpn_cls',
                color=colors['darkorange'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['loss_rpn_regr'],
                label='val set: loss_rpn_regr',
                color=colors['yellow'])


# Classification
axs[1].plot(    np.arange(0, r_epochs), 
                record_train['class_pig_acc'],
                label='train set: class_pig_acc',
                color=colors['darkblue'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_train['class_others_acc'],
                label='train set: class_others_acc',
                color=colors['black'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_train['loss_class_cls'],
                label='train set: loss_class_cls',
                color=colors['darkgreen'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_train['loss_class_regr'],
                label='train set: loss_class_regr',
                color=colors['limegreen'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['class_pig_acc']*1281/497,
                label='val set: class_pig_acc',
                color=colors['darkred'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['class_others_acc']*1281/497,
                label='val set: class_others_acc',
                color=colors['darkgray'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['loss_class_cls'],
                label='val set: loss_class_cls',
                color=colors['darkorange'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['loss_class_regr'],
                label='val set: loss_class_regr',
                color=colors['yellow'])


# TOTAL 
axs[2].plot(    np.arange(0, r_epochs), 
                record_train['curr_loss'],
                label='train set: curr_loss',
                color=colors['darkgreen'])

axs[2].plot(    np.arange(0, r_epochs), 
                record_test['curr_loss'],
                label='val set: curr_loss',
                color=colors['darkorange'])

for i in range(3):
    axs[i].legend(loc="best")

plt.show()
