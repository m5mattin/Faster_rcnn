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

fig, axs = plt.subplots(1, 2)

axs[0].set_title('Rpn',color='black')
axs[1].set_title('Classification',color='black')



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
                label='val set: recall',
                color=colors['darkred'])

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
                label='val set: loss_rpn_cls',
                color=colors['darkgreen'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['loss_rpn_regr'],
                label='val set: loss_rpn_regr',
                color=colors['lime'])

# Classification

tp_class_train = record_train['class_00']
fn_class_train = record_train['class_10'] + record_train['class_20']
fp_class_train = record_train['class_01'] + record_train['class_02']
tn_class_train = record_train['class_11'] + record_train['class_22'] + record_train['class_12'] + record_train['class_21']
acc_class_train = (tp_class_train+tn_class_train)/(tp_class_train+fn_class_train+fp_class_train+tn_class_train)

tp_class_test = record_test['class_00']
fn_class_test = record_test['class_10'] + record_test['class_20']
fp_class_test = record_test['class_01'] + record_test['class_02']
tn_class_test = record_test['class_11'] + record_test['class_22'] + record_test['class_12'] + record_test['class_21']
acc_class_test = (tp_class_test+tn_class_test)/(tp_class_test+fn_class_test+fp_class_test+tn_class_test)

# loss 
axs[1].plot(    np.arange(0, r_epochs), 
                record_train['loss_class_cls'],
                label='train set: loss_class_cls',
                color=colors['darkblue'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_train['loss_class_regr'],
                label='train set: loss_class_regr',
                color=colors['royalblue'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['loss_class_cls'],
                label='val set: loss_class_cls',
                color=colors['darkgreen'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['loss_class_regr'],
                label='val set: loss_class_regr',
                color=colors['lime'])


# accuracy

axs[1].plot(    np.arange(0, r_epochs), 
                acc_class_train,
                label='train set: class_pig_acc',
                color=colors['darkorange'])

axs[1].plot(    np.arange(0, r_epochs), 
                acc_class_test,
                label='val set: class_pig_acc',
                color=colors['darkred'])

for i in range(2):
    axs[i].legend(loc="best")

plt.show()
