import numpy as np 

a = np.array([0,0,1])

print(a.shape)
b = np.where(a>0)
print(b) 

loss_class_test = []
for i in range (5):
    
    loss = np.array([i,i+i,i*i])
    print(loss)
    loss_class_test.append(loss)

loss_class_test = np.asarray(loss_class_test)
print(np.mean(loss_class_test[:,0]))
print(np.mean(loss_class_test[:,1]))
print(np.mean(loss_class_test[:,2]))
