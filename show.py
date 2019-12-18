from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

def get_rpn_recall(c):
    tp = c['rpn_00']
    fn = c['rpn_10']
    return tp / (tp+fn)

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Load the records
record_train = pd.read_csv("../record_train.csv")
record_test = pd.read_csv("../record_test.csv")

r_epochs = len(record_train)

fig, axs = plt.subplots(1, 3)

axs[0].set_title('Rpn',color='black')
axs[1].set_title('Classification',color='black')
axs[2].set_title('Total loss',color='black')


### RPN

tp_rpn_test = record_test['rpn_00']
fn_rpn_test = record_test['rpn_10']
recall_rpn_test = tp_rpn_test / (tp_rpn_test + fn_rpn_test)

tp_rpn_train = record_train['rpn_00']
fn_rpn_train = record_train['rpn_10']
recall_rpn_train = tp_rpn_train / (tp_rpn_train + fn_rpn_train)

# recall
axs[0].plot(    np.arange(0, r_epochs), 
                recall_rpn_train,
                label='train set: recall',
                color=colors['darkorange'])

axs[0].plot(    np.arange(0, r_epochs), 
                recall_rpn_test,
                label='eval set: recall',
                color=colors['darkred'])
# mean overlap

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['mean_overlapping_bboxes_pig']/100,
                label='train set: mean_overlapping_bboxes_pig/100',
                color=colors['indigo'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['mean_overlapping_bboxes_pig']/100,
                label='train set: mean_overlapping_bboxes_pig/100',
                color=colors['purple'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['mean_overlapping_bboxes_pig']/100,
                label='eval set: mean_overlapping_bboxes_pig/100',
                color=colors['black'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['mean_overlapping_bboxes_others']/100,
                label='eval set: mean_overlapping_bboxes_others/100',
                color=colors['dimgray'])


# Loss

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['loss_rpn_cls'],
                label='train set: loss_rpn_cls',
                color=colors['darkblue'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['loss_rpn_regr'],
                label='train set: loss_rpn_regr',
                color=colors['royalblue'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['loss_rpn_cls'],
                label='eval set: loss_rpn_cls',
                color=colors['darkgreen'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['loss_rpn_regr'],
                label='eval set: loss_rpn_regr',
                color=colors['lime'])

# # Classification

# axs[1].plot(    np.arange(0, r_epochs), 
#                 record_train['loss_class_cls'],
#                 label='train set: loss_class_cls',
#                 color=colors['darkgreen'])

# axs[1].plot(    np.arange(0, r_epochs), 
#                 record_train['loss_class_regr'],
#                 label='train set: loss_class_regr',
#                 color=colors['limegreen'])

# axs[1].plot(    np.arange(0, r_epochs), 
#                 record_test['class_pig_acc']*1281/497,
#                 label='val set: class_pig_acc',
#                 color=colors['darkred'])

# axs[1].plot(    np.arange(0, r_epochs), 
#                 record_test['class_others_acc']*1281/497,
#                 label='val set: class_others_acc',
#                 color=colors['darkgray'])

# axs[1].plot(    np.arange(0, r_epochs), 
#                 record_test['loss_class_cls'],
#                 label='val set: loss_class_cls',
#                 color=colors['darkorange'])

# axs[1].plot(    np.arange(0, r_epochs), 
#                 record_test['loss_class_regr'],
#                 label='val set: loss_class_regr',
#                 color=colors['yellow'])


# # TOTAL 
# axs[2].plot(    np.arange(0, r_epochs), 
#                 record_train['curr_loss'],
#                 label='train set: curr_loss',
#                 color=colors['darkgreen'])

# axs[2].plot(    np.arange(0, r_epochs), 
#                 record_test['curr_loss'],
#                 label='val set: curr_loss',
#                 color=colors['darkorange'])

for i in range(3):
    axs[i].legend(loc="best")

plt.show()
