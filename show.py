from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import argparse
def get_rpn_recall(c):
    tp = c['rpn_00']
    fn = c['rpn_10']
    return tp / (tp+fn)
parser = argparse.ArgumentParser(description='Argument')
parser.add_argument('-ap','--ap', action='store_true',
			help='show result rpn')

args = parser.parse_args()
show_ap = args.ap

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Load the records
record_train = pd.read_csv("../record_train.csv")
record_test = pd.read_csv("../record_test.csv")

r_epochs = len(record_train)

fig, axs = plt.subplots(1, 4)

axs[0].set_title('Rpn',color='black')
axs[1].set_title('Mean Overlap',color='black')
axs[2].set_title('Classification',color='black')
axs[3].set_title('Detection',color='black')

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
                color=colors['darkred'])

axs[0].plot(    np.arange(0, r_epochs), 
                recall_rpn_test,
                label='val set: recall',
                color=colors['darkred'],
                linestyle=":")

# Loss rpn

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['loss_rpn_cls'],
                label='train set: loss_rpn_cls',
                color=colors['darkblue'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['loss_rpn_regr'],
                label='train set: loss_rpn_regr',
                color=colors['darkgreen'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['loss_rpn_cls'],
                label='val set: loss_rpn_cls',
                color=colors['darkblue'],
                linestyle=":")

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['loss_rpn_regr'],
                label='val set: loss_rpn_regr',
                color=colors['darkgreen'],
                linestyle=":")


# mean overlap 

axs[1].plot(    np.arange(0, r_epochs), 
                record_train['rpn_mop'],
                label='train set: mean overlap pig',
                color=colors['darkgreen'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_train['rpn_moo'],
                label='train set: mean overlap others',
                color=colors['darkorange'])

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['rpn_mop'],
                label='val set: mean overlap pig',
                color=colors['darkgreen'],
                linestyle=":")

axs[1].plot(    np.arange(0, r_epochs), 
                record_test['rpn_moo'],
                label='val set: mean overlap others',
                color=colors['darkorange'],
                linestyle=":")

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
axs[2].plot(    np.arange(0, r_epochs), 
                record_train['loss_class_cls'],
                label='train set: loss_class_cls',
                color=colors['darkblue'])

axs[2].plot(    np.arange(0, r_epochs), 
                record_train['loss_class_regr'],
                label='train set: loss_class_regr',
                color=colors['darkgreen'])

axs[2].plot(    np.arange(0, r_epochs), 
                acc_class_train,
                label='train set: class_pig_acc',
                color=colors['darkred'])

axs[2].plot(    np.arange(0, r_epochs), 
                record_test['loss_class_cls'],
                label='val set: loss_class_cls',
                color=colors['royalblue'],
                linestyle=':')

axs[2].plot(    np.arange(0, r_epochs), 
                record_test['loss_class_regr'],
                label='val set: loss_class_regr',
                color=colors['darkgreen'],
                linestyle=':')

axs[2].plot(    np.arange(0, r_epochs), 
                acc_class_test,
                label='val set: class_pig_acc',
                color=colors['darkred'],
                linestyle=':')

#Detection

axs[3].plot(    np.arange(0, r_epochs), 
                record_train['curr_loss'],
                label='train set: curr_loss',
                color=colors['darkblue'])

axs[3].plot(    np.arange(0, r_epochs), 
                record_test['curr_loss'],
                label='test set: curr_loss',
                color=colors['darkblue'],
                linestyle=':')

axs[3].plot(    np.arange(0, r_epochs), 
                record_train['ap50'],
                label='train set: ap50',
                color=colors['darkgreen'])

axs[3].plot(    np.arange(0, r_epochs), 
                record_test['ap50'],
                label='test set: ap50',
                color=colors['darkgreen'],
                linestyle=':')

axs[3].plot(    np.arange(0, r_epochs), 
                record_train['ap75'],
                label='train set: ap75',
                color=colors['darkred'])

axs[3].plot(    np.arange(0, r_epochs), 
                record_test['ap75'],
                label='test set: ap75',
                color=colors['darkred'],
                linestyle=':')


for i in range(4):
    axs[i].legend(loc="best")

plt.show()
