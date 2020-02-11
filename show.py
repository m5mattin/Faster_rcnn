from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import argparse

def get_rpn_recall(c):
    tp = c['rpn_00']
    fn = c['rpn_10']
    return tp / (tp+fn)

def get_class_f1(c):
    tp = c['class_00']
    fn = c['class_10']  + c['class_10']
    fp = c['class_01']  + c['class_02']
    tn = c['class_11']  + c['class_12'] + c['class_21'] + c['class_22']
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2*(precision * recall)/(precision + recall)
    return recall, precision, f1


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
if show_ap:
    fig, axs = plt.subplots(1, 4)
    axs[3].set_title('Detection',color='black')

else: 
    fig, axs = plt.subplots(1, 4)

axs[0].set_title('Rpn',color='black')
axs[1].set_title('Mean Overlap',color='black')
axs[2].set_title('Classification',color='black')
axs[3].set_title('Detection',color='black')

### RPN

recall_rpn_test = get_rpn_recall(record_test)
recall_rpn_train = get_rpn_recall(record_train)

axs[0].plot(    np.arange(0, r_epochs), 
                record_train['ap_rpn50'],
                label='train set: ap50',
                color=colors['chocolate'])
axs[0].plot(    np.arange(0, r_epochs), 
                record_train['ap_rpn75'],
                label='train set: ap50',
                color=colors['darkred'])

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['ap_rpn50'],
                label='val set: ap50',
                color=colors['chocolate'],
                linestyle=":")

axs[0].plot(    np.arange(0, r_epochs), 
                record_test['ap_rpn50'],
                label='val set: ap75',
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

recall_class_test, precision_class_test, f1_class_test = get_class_f1(record_test)
recall_class_train, precision_class_train, f1_class_train = get_class_f1(record_train)

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
                f1_class_train,
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
                f1_class_test,
                label='val set: class_pig_acc',
                color=colors['darkred'],
                linestyle=':')

axs[2].plot(    np.arange(0, r_epochs), 
                precision_class_test,
                label='val set: precision_class_test',
                color=colors['darkviolet'],
                linestyle=':')

axs[2].plot(    np.arange(0, r_epochs), 
                recall_class_test,
                label='val set: recall_class_test',
                color=colors['pink'],
                linestyle=':')


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
                record_train['ap_detect50'],
                label='train set: ap50',
                color=colors['darkgreen'])

axs[3].plot(    np.arange(0, r_epochs), 
                record_test['ap_detect50'],
                label='test set: ap50',
                color=colors['darkgreen'],
                linestyle=':')

axs[3].plot(    np.arange(0, r_epochs), 
                record_train['ap_detect75'],
                label='train set: ap75',
                color=colors['darkred'])

axs[3].plot(    np.arange(0, r_epochs), 
                record_test['ap_detect75'],
                label='test set: ap75',
                color=colors['darkred'],
                linestyle=':')
axs[3].legend(loc="best")   


for i in range(4):
    axs[i].legend(loc="best")

plt.show()
