from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import argparse

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
parser = argparse.ArgumentParser(description='Argument')
parser.add_argument('--num_epoch', type=int,
			help='num_rois')

parser.add_argument('-P','--P', action='store_true')
parser.add_argument('-PaO','--PaO', action='store_true')
parser.add_argument('-PaCO','--PaCO', action='store_true')
parser.add_argument('-HA','--HA', action='store_true')
parser.add_argument('-HOEM','--HOEM', action='store_true')
parser.add_argument('-MO','--MO', action='store_true')

def get_rpn_recall(c):
    tp = c['rpn_00']
    fn = c['rpn_10']
    return tp / (tp+fn + 1e-16)

def get_class_f1(c):
    tp = c['class_00']
    fn = c['class_10']  + c['class_10']
    fp = c['class_01']  + c['class_02']
    tn = c['class_11']  + c['class_12'] + c['class_21'] + c['class_22']
    recall = tp / (tp + fn + 1e-16)
    precision = tp / (tp + fp + 1e-16)
    f1 = 2*(precision * recall)/(precision + recall+ + 1e-16)
    return recall, precision, f1


args = parser.parse_args()
nb_epochs = args.num_epoch
r_epochs = np.arange(0, nb_epochs)
show_P = args.P
show_2C = args.PaO
show_3C = args.PaCO
show_HA = args.HA
show_HOEM = args.HOEM
show_mean_overlap = args.MO

nb_method = 0
if show_P: 
    nb_method  = nb_method + 1
if show_2C: 
    nb_method  = nb_method + 1
if show_3C: 
    nb_method  = nb_method + 1
if show_HA: 
    nb_method  = nb_method + 1
if show_HOEM: 
    nb_method  = nb_method + 1

nb_method = 4
train_P = pd.read_csv("../models/P/record_train.csv")
test_P = pd.read_csv("../models/P/record_test.csv")
train_PaO = pd.read_csv("../models/PaO/record_train.csv")
test_PaO = pd.read_csv("../models/PaO/record_test.csv")
train_PaCO = pd.read_csv("../models/PaCO/record_train.csv")
test_PaCO = pd.read_csv("../models/PaCO/record_test.csv")
train_2CHA = pd.read_csv("../models/2CHA/record_train.csv")
test_2CHA = pd.read_csv("../models/2CHA/record_test.csv")
train_HOEM = pd.read_csv("../models/HOEM/record_train.csv")
test_HOEM = pd.read_csv("../models/HOEM/record_test.csv")

class_recall_P_train, class_precision_P_train, class_f1_P_train = get_class_f1(train_P)
class_recall_P_test, class_precision_P_test, class_f1_P_test = get_class_f1(test_P)
class_recall_PaO_train, class_precision_PaO_train, class_f1_PaO_train = get_class_f1(train_PaO)
class_recall_PaO_test, class_precision_PaO_test, class_f1_PaO_test = get_class_f1(test_PaO)
class_recall_PaCO_train, class_precision_PaCO_train, class_f1_PaCO_train = get_class_f1(train_PaCO)
class_recall_PaCO_test, class_precision_PaCO_test, class_f1_PaCO_test = get_class_f1(test_PaCO)
class_recall_2CHA_train, class_precision_2CHA_train, class_f1_2CHA_train = get_class_f1(train_2CHA)
class_recall_2CHA_test, class_precision_2CHA_test, class_f1_2CHA_test = get_class_f1(test_2CHA)
class_recall_HOEM_train, class_precision_HOEM_train, class_f1_HOEM_train = get_class_f1(train_HOEM)
class_recall_HOEM_test, class_precision_HOEM_test, class_f1_HOEM_test = get_class_f1(test_HOEM)




if show_mean_overlap:
    fig, axs = plt.subplots(1, 4)
    id_rpn = 0
    id_mo = 1
    id_class = 2
    id_detect = 3
    axs[0].set_title('Rpn',color='black')
    axs[1].set_title('Mean Overlap',color='black')
    axs[2].set_title('Classification',color='black')
    axs[3].set_title('Detection',color='black')
    axs[0].set_ylim(0,1.05)
    axs[2].set_ylim(0,1.05)
    axs[3].set_ylim(0,1.05)
else:
    fig, axs = plt.subplots(1, 3)
    id_rpn = 0
    id_class = 1
    id_detect = 2
    axs[0].set_title('Rpn',color='black')
    axs[1].set_title('Classification',color='black')
    axs[2].set_title('Detection',color='black')
    axs[0].set_ylim(0,1.05)
    axs[1].set_ylim(0,1.05)
    axs[2].set_ylim(0,1.05)



# axs[0].text(50, 1.01, 'recall', {'color': 'black', 'fontsize': 15})
# axs[0].text(50, 0.5, 'losses', {'color': 'black', 'fontsize': 15})

# axs[1].text(35, 59, 'overlap pigs', {'color': 'black', 'fontsize': 15})
# axs[1].text(35, 20, 'overlap others', {'color': 'black', 'fontsize': 15})

# axs[2].text(50, 1.01, 'f1_scores', {'color': 'black', 'fontsize': 15})
# axs[2].text(50, 0.5, 'losses', {'color': 'black', 'fontsize': 15})

# axs[3].text(50, 1.01, 'ap50', {'color': 'black', 'fontsize': 15})
# axs[3].set_ylim(0,1.05)

############# RPN
if show_P:
    axs[id_rpn].plot(    r_epochs, 
                    train_P['ap_rpn50'][0:nb_epochs],
                    label='P:ap50',
                    color=colors['black'])
    axs[0].plot(    r_epochs, 
                    train_P['loss_rpn_cls'][0:nb_epochs] + train_P['loss_rpn_regr'][0:nb_epochs],
                    label='P:loss',
                    color=colors['gray'])
if show_2C:
    axs[id_rpn].plot(    r_epochs, 
                    train_PaO['ap_rpn50'][0:nb_epochs],
                    label='2C:ap50',
                    color=colors['darkred'])
    axs[id_rpn].plot(    r_epochs, 
                    train_PaO['loss_rpn_cls'][0:nb_epochs] + train_PaO['loss_rpn_regr'][0:nb_epochs],
                    label='2C:loss',
                    color=colors['red'])
if show_3C:
    axs[id_rpn].plot(    r_epochs, 
                    train_PaCO['ap_rpn50'][0:nb_epochs],
                    label='3C:ap50',
                    color=colors['darkblue'])
    axs[id_rpn].plot(    r_epochs, 
                    train_PaCO['loss_rpn_cls'][0:nb_epochs] + train_PaCO['loss_rpn_regr'][0:nb_epochs],
                    label='3C:loss',
                    color=colors['royalblue'])
if show_HA:
    axs[id_rpn].plot(    r_epochs, 
                    train_2CHA['ap_rpn50'][0:nb_epochs],
                    label='HA:ap50',
                    color=colors['darkgreen'])
    axs[0].plot(    r_epochs, 
                    train_2CHA['loss_rpn_cls'][0:nb_epochs] + train_2CHA['loss_rpn_regr'][0:nb_epochs],
                    label='HA:loss',
                    color=colors['lime'])
if show_HOEM:
    axs[id_rpn].plot(    r_epochs, 
                        train_HOEM['ap_rpn50'][0:nb_epochs],
                        label='HOEM:ap50',
                        color=colors['darkorange'])
    axs[0].plot(    r_epochs, 
                    train_HOEM['loss_rpn_cls'][0:nb_epochs] + train_HOEM['loss_rpn_regr'][0:nb_epochs],
                    label='HOEM:loss',
                    color=colors['orange'])

axs[id_rpn].legend(  loc='upper center',
                bbox_to_anchor=(0.5,-0.025),
                ncol=nb_method,
                title='RPN',
                edgecolor = 'black',
                labelspacing=0.1,
                handlelength=1.3,
                handletextpad=0.1,
                columnspacing=1)
if show_P:
    axs[id_rpn].plot(    r_epochs, 
                    test_P['ap_rpn50'][0:nb_epochs],
                    label='P : ap50',
                    color=colors['black'],
                    linestyle=":")

    axs[id_rpn].plot(    r_epochs, 
                    test_P['loss_rpn_cls'][0:nb_epochs] + train_P['loss_rpn_regr'][0:nb_epochs],
                    label='P : loss',
                    color=colors['gray'],
                    linestyle=":")
if show_2C:
    axs[id_rpn].plot(    r_epochs, 
                    test_PaO['ap_rpn50'][0:nb_epochs],
                    label='2C : ap50',
                    color=colors['darkred'],
                    linestyle=":")

    axs[id_rpn].plot(    r_epochs, 
                    test_PaO['loss_rpn_cls'][0:nb_epochs] + train_PaO['loss_rpn_regr'][0:nb_epochs],
                    label='2C : loss',
                    color=colors['red'],
                    linestyle=":")
if show_3C:
    axs[id_rpn].plot(    r_epochs, 
                    test_PaCO['ap_rpn50'][0:nb_epochs],
                    label='3C : ap50',
                    color=colors['darkblue'],
                    linestyle=":")
    # axs[0].plot(    r_epochs, 
    #                 test_PaCO['ap_rpn75'][0:nb_epochs],
    #                 label='3C : ap75',
    #                 color=colors['blue'],
    #                 linestyle=":")
    axs[id_rpn].plot(    r_epochs, 
                    test_PaCO['loss_rpn_cls'][0:nb_epochs] + train_PaCO['loss_rpn_regr'][0:nb_epochs],
                    label='3C : loss',
                    color=colors['royalblue'],
                    linestyle=":")
if show_HA:
    axs[id_rpn].plot(    r_epochs, 
                    test_2CHA['ap_rpn50'][0:nb_epochs],
                    color=colors['darkgreen'],
                    linestyle=":")
    axs[id_rpn].plot(    r_epochs, 
                    test_2CHA['loss_rpn_cls'][0:nb_epochs] + test_2CHA['loss_rpn_regr'][0:nb_epochs],
                    color=colors['lime'],
                    linestyle=":")
if show_HOEM:
    axs[id_rpn].plot(    r_epochs, 
                        test_HOEM['ap_rpn50'][0:nb_epochs],
                        color=colors['darkorange'],
                        linestyle=":")

    axs[id_rpn].plot(    r_epochs, 
                        test_HOEM['loss_rpn_cls'][0:nb_epochs] + test_HOEM['loss_rpn_regr'][0:nb_epochs],
                        color=colors['orange'],
                        linestyle=":")

############# Overlap
if show_mean_overlap:
    if show_P:
        axs[id_mo].plot(    r_epochs, 
                        train_P['rpn_mop'][0:nb_epochs],
                        label='P:pigs',
                        color=colors['black'])
        axs[1].plot(    r_epochs, 
                        train_P['rpn_moo'][0:nb_epochs],
                        label='P:others',
                        color=colors['gray'])
    if show_2C:
        axs[id_mo].plot(    r_epochs, 
                        train_PaO['rpn_mop'][0:nb_epochs],
                        label='2C:pigs',
                        color=colors['darkred'])
        axs[id_mo].plot(    r_epochs, 
                        train_PaO['rpn_moo'][0:nb_epochs],
                        label='2C:others',
                        color=colors['red'])
    if show_3C:
        axs[id_mo].plot(    r_epochs, 
                        train_PaCO['rpn_mop'][0:nb_epochs],
                        label='3C:pigs',
                        color=colors['darkblue'])
        axs[id_mo].plot(    r_epochs, 
                        train_PaCO['rpn_moo'][0:nb_epochs],
                        label='3C:others',
                        color=colors['blue'])
    if show_HA:
        axs[id_mo].plot(    r_epochs, 
                        train_2CHA['rpn_mop'][0:nb_epochs],
                        label='HA:pigs',
                        color=colors['darkgreen'])
        axs[id_mo].plot(    r_epochs, 
                        train_2CHA['rpn_moo'][0:nb_epochs],
                        label='HA:others',
                        color=colors['forestgreen'])
    if show_HOEM:
        axs[id_mo].plot(    r_epochs, 
                        train_HOEM['rpn_mop'][0:nb_epochs],
                        label='HOEM:pigs',
                        color=colors['darkorange'])
        axs[id_mo].plot(    r_epochs, 
                        train_HOEM['rpn_moo'][0:nb_epochs],
                        label='HOEM:others',
                        color=colors['orange'])
                        
    axs[id_mo].legend(  loc='upper center',
                    bbox_to_anchor=(0.5,-0.025),
                    ncol=nb_method,
                    title='Mean Overlap RPN',
                    edgecolor = 'black',
                    labelspacing=0.1,
                    handlelength=1.3,
                    handletextpad=0.1,
                    columnspacing=1)
    if show_P:                
        axs[id_mo].plot(    r_epochs, 
                        test_P['rpn_mop'][0:nb_epochs],
                        color=colors['black'],
                        linestyle=":")
        axs[id_mo].plot(    r_epochs, 
                        test_P['rpn_moo'][0:nb_epochs],
                        color=colors['gray'],
                        linestyle=":")
    if show_2C:                
        axs[id_mo].plot(    r_epochs, 
                        test_PaO['rpn_mop'][0:nb_epochs],
                        color=colors['darkred'],
                        linestyle=":")
        axs[id_mo].plot(    r_epochs, 
                        test_PaO['rpn_moo'][0:nb_epochs],
                        color=colors['red'],
                        linestyle=":")
    if show_3C:
        axs[id_mo].plot(    r_epochs, 
                        test_PaCO['rpn_mop'][0:nb_epochs],
                        color=colors['darkblue'],
                        linestyle=":")
        axs[id_mo].plot(    r_epochs, 
                        test_PaCO['rpn_moo'][0:nb_epochs],
                        color=colors['blue'],
                        linestyle=":")
    if show_HA:
        axs[id_mo].plot(    r_epochs, 
                        test_2CHA['rpn_mop'][0:nb_epochs],
                        label='2CHA : pigs',
                        color=colors['darkgreen'],
                        linestyle=":")
        axs[id_mo].plot(    r_epochs, 
                        test_2CHA['rpn_moo'][0:nb_epochs],
                        label='2CHA : others',
                        color=colors['forestgreen'],
                        linestyle=":")
    if show_HOEM:
        axs[id_mo].plot(    r_epochs, 
                        test_HOEM['rpn_mop'][0:nb_epochs],
                        color=colors['darkorange'],
                        linestyle=":")
        axs[id_mo].plot(    r_epochs, 
                        test_HOEM['rpn_moo'][0:nb_epochs],
                        color=colors['orange'],
                        linestyle=":")

############# Classification

if show_P:
    axs[id_class].plot(    r_epochs, 
                    class_f1_P_train[0:nb_epochs],
                    label='P:F1',
                    color=colors['black'])

    axs[id_class].plot(    r_epochs, 
                    train_P['loss_class_cls'][0:nb_epochs] + train_P['loss_class_regr'][0:nb_epochs],
                    label='P:loss',
                    color=colors['gray'])
if show_2C:
    axs[id_class].plot(    r_epochs, 
                    class_f1_PaO_train[0:nb_epochs],
                    label='2C:F1',
                    color=colors['darkred'])

    axs[id_class].plot(    r_epochs, 
                    train_PaO['loss_class_cls'][0:nb_epochs] + train_PaO['loss_class_regr'][0:nb_epochs],
                    label='2C:loss',
                    color=colors['red'])
if show_3C:
    axs[id_class].plot(    r_epochs, 
                    class_f1_PaCO_train[0:nb_epochs],
                    label='3C : F1',
                    color=colors['darkblue'])

    axs[id_class].plot(    r_epochs, 
                    train_PaCO['loss_class_cls'][0:nb_epochs] + train_PaO['loss_class_regr'][0:nb_epochs],
                    label='3C:loss',
                    color=colors['royalblue'])
if show_HA:
    axs[id_class].plot(    r_epochs, 
                    class_f1_2CHA_train[0:nb_epochs],
                    label='HA:F1',
                    color=colors['darkgreen'])

    axs[id_class].plot(    r_epochs, 
                    train_2CHA['loss_class_cls'][0:nb_epochs] + train_2CHA['loss_class_regr'][0:nb_epochs],
                    label='HA:loss',
                    color=colors['lime'])
if show_HOEM:
    axs[id_class].plot(    r_epochs, 
                    class_f1_HOEM_train[0:nb_epochs],
                    label='HOEM:F1',
                    color=colors['darkorange'])

    axs[id_class].plot(    r_epochs, 
                    train_HOEM['loss_class_cls'][0:nb_epochs] + train_HOEM['loss_class_regr'][0:nb_epochs],
                    label='HOEM:loss',
                    color=colors['orange'])

axs[id_class].legend(  loc='upper center',
                bbox_to_anchor=(0.5,-0.025),
                ncol=nb_method,
                title='Classification',
                edgecolor = 'black',
                labelspacing=0.1,
                handlelength=1.3,
                handletextpad=0.1,
                columnspacing=1)
if show_P:
    axs[id_class].plot(    r_epochs, 
                    class_f1_P_test[0:nb_epochs],
                    color=colors['black'],
                    linestyle=":")

    axs[id_class].plot(    r_epochs, 
                    test_P['loss_class_cls'][0:nb_epochs] + train_P['loss_class_regr'][0:nb_epochs],
                    color=colors['gray'],
                    linestyle=":")
if show_2C:
    axs[id_class].plot(    r_epochs, 
                    class_f1_PaO_test[0:nb_epochs],
                    color=colors['darkred'],
                    linestyle=":")

    axs[id_class].plot(    r_epochs, 
                    test_PaO['loss_class_cls'][0:nb_epochs] + train_PaO['loss_class_regr'][0:nb_epochs],
                    color=colors['red'],
                    linestyle=":")
if show_3C:
    axs[id_class].plot(    r_epochs, 
                    class_f1_PaCO_test[0:nb_epochs],
                    color=colors['darkblue'],
                    linestyle=":")

    axs[id_class].plot(    r_epochs, 
                    test_PaCO['loss_class_cls'][0:nb_epochs] + train_PaO['loss_class_regr'][0:nb_epochs],
                    color=colors['royalblue'],
                    linestyle=":")
if show_HA:
    axs[id_class].plot(    r_epochs, 
                    class_f1_2CHA_test[0:nb_epochs],
                    color=colors['darkgreen'],
                    linestyle=":")

    axs[id_class].plot(    r_epochs, 
                    test_2CHA['loss_class_cls'][0:nb_epochs] + test_2CHA['loss_class_regr'][0:nb_epochs],
                    color=colors['lime'],
                    linestyle=":")
if show_HOEM:
    axs[id_class].plot(    r_epochs, 
                    class_f1_HOEM_test[0:nb_epochs],
                    color=colors['darkorange'],
                    linestyle=":")

    axs[id_class].plot(    r_epochs, 
                    test_HOEM['loss_class_cls'][0:nb_epochs] + test_HOEM['loss_class_regr'][0:nb_epochs],
                    color=colors['orange'],
                    linestyle=":")

############# Detection
if show_P:
    axs[id_detect].plot(    r_epochs, 
                    train_P['ap_detect50'][0:nb_epochs],
                    label='P:ap 50',
                    color=colors['black'])

    axs[id_detect].plot(    r_epochs, 
                    train_P['ap_detect75'][0:nb_epochs],
                    label='P:ap 75',
                    color=colors['gray'])
if show_2C:
    axs[id_detect].plot(    r_epochs, 
                    train_PaO['ap_detect50'][0:nb_epochs],
                    label='2C:ap 50',
                    color=colors['darkred'])

    axs[id_detect].plot(    r_epochs, 
                    train_PaO['ap_detect75'][0:nb_epochs],
                    label='2C:ap 75',
                    color=colors['red'])
if show_3C:
    axs[id_detect].plot(    r_epochs, 
                    train_PaCO['ap_detect50'][0:nb_epochs],
                    label='3C:ap 50',
                    color=colors['darkblue'])

    axs[id_detect].plot(    r_epochs, 
                    train_PaCO['ap_detect75'][0:nb_epochs],
                    label='3C:ap 75',
                    color=colors['blue'])
if show_HA:
    axs[id_detect].plot(    r_epochs, 
                    train_2CHA['ap_detect50'][0:nb_epochs],
                    label='HA:ap 50',
                    color=colors['darkgreen'])

    axs[id_detect].plot(    r_epochs, 
                    train_2CHA['ap_detect75'][0:nb_epochs],
                    label='HA:ap 75',
                    color=colors['forestgreen'])
if show_HOEM:
    axs[id_detect].plot(    r_epochs, 
                    train_HOEM['ap_detect50'][0:nb_epochs],
                    label='HOEM:ap 50',
                    color=colors['darkorange'])

    axs[id_detect].plot(    r_epochs, 
                    train_HOEM['ap_detect75'][0:nb_epochs],
                    label='HOEM:ap 75',
                    color=colors['orange'])

axs[id_detect].legend(  loc='upper center',
                bbox_to_anchor=(0.5,-0.025),
                ncol=nb_method,
                title='Detection',
                edgecolor = 'black',
                labelspacing=0.1,
                handlelength=1.3,
                handletextpad=0.1,
                columnspacing=1)
if show_P:
    axs[id_detect].plot(    r_epochs, 
                    test_P['ap_detect50'][0:nb_epochs],
                    label='P:ap 50',
                    color=colors['black'],
                    linestyle=":")

    axs[id_detect].plot(    r_epochs, 
                    test_P['ap_detect75'][0:nb_epochs],
                    label='P:ap 75',
                    color=colors['gray'],
                    linestyle=":")
if show_2C:
    axs[id_detect].plot(    r_epochs, 
                    test_PaO['ap_detect50'][0:nb_epochs],
                    label='2C:ap 50',
                    color=colors['darkred'],
                    linestyle=":")

    axs[id_detect].plot(    r_epochs, 
                    test_PaO['ap_detect75'][0:nb_epochs],
                    label='2C:ap 75',
                    color=colors['red'],
                    linestyle=":")

if show_3C:
    axs[id_detect].plot(    r_epochs, 
                    test_PaCO['ap_detect50'][0:nb_epochs],
                    label='3C:ap 50',
                    color=colors['darkblue'],
                    linestyle=":")

    axs[id_detect].plot(    r_epochs, 
                    test_PaCO['ap_detect75'][0:nb_epochs],
                    label='3C:ap 75',
                    color=colors['blue'],
                    linestyle=":")

if show_HA:
    axs[id_detect].plot(    r_epochs, 
                    test_2CHA['ap_detect50'][0:nb_epochs],
                    color=colors['darkgreen'],
                    linestyle=":")

    axs[id_detect].plot(    r_epochs, 
                    test_2CHA['ap_detect75'][0:nb_epochs],
                    color=colors['forestgreen'],
                    linestyle=":")
if show_HOEM:
    axs[id_detect].plot(    r_epochs, 
                    test_HOEM['ap_detect50'][0:nb_epochs],
                    color=colors['darkorange'],
                    linestyle=":")

    axs[id_detect].plot(    r_epochs, 
                    test_HOEM['ap_detect75'][0:nb_epochs],
                    color=colors['orange'],
                    linestyle=":")

plt.show()