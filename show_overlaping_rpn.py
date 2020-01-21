from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

train = np.load('../overlaping_rpn_train.npy')
test = np.load('../overlaping_rpn_test.npy')
epoch_num = 20
# fig, axs = plt.subplots(1, 3)

# axs[0].set_title('Overlap Pig',color='black')
# axs[1].set_title('Overlap Others',color='black')
# axs[2].set_title('Overlap Negative',color='black')

# cols = ['b', 'g', 'r', 'c', 'm', 'y'];

# 
# for i in range(1):

#     axs[0].plot(   np.arange(0,train.shape[0]),
#                     train[:,0,epoch_num],
#                     label='epoch {}'.format(epoch_num),
#                     color=cols[i])

#     axs[1].plot(   np.arange(0,train.shape[0]),
#                     train[:,1,epoch_num],
#                     label='epoch {}'.format(epoch_num),
#                     color=cols[i])
    
#     axs[2].plot(   np.arange(0,train.shape[0]),
#                     train[:,2,epoch_num],
#                     label='epoch {}'.format(epoch_num),
#                     color=cols[i])

# for i in range(3):
#     axs[i].legend(loc="best")
#     axs[i].set_xlabel("training image")
#     axs[i].set_ylabel("nb boxes overlaping")

# plt.show()

print(train.shape)
bar_width = 1
opacity = 1

rect3 = plt.bar(np.arange(0,train.shape[0]), train[:,2,epoch_num], bar_width,  color = "r", alpha=opacity)

rect1 = plt.bar(np.arange(0,train.shape[0]), train[:,1,epoch_num], bar_width, color = "y", alpha=opacity,bottom=train[:,2,epoch_num])

rect2 = plt.bar(np.arange(0,train.shape[0]), train[:,0,epoch_num], bar_width, color = "g", alpha=opacity,bottom=train[:,2,epoch_num]+train[:,1,epoch_num])


plt.tight_layout()
plt.show()