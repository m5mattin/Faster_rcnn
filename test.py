import numpy as np 

gt = ["a","b","c","z"]
d = ["b","f","c","b","k","a","a"]

PR = np.zeros((len(d), 4))

nb_gt = len(gt)

Tp = 0
Fp = 0
Fn = 0

cnt = 0
Precision_inter = 0

for i in range (len(d)):
    cnt = cnt + 1
    for j in range(len(gt)):
        if d[i] == gt[j]:
            Tp = Tp + 1
            PR[i][0] = 1
            Precision_inter = Tp / cnt
            del gt[j]
            break
        
    PR[i][1] = Tp / cnt
    PR[i][2] = Tp / (nb_gt + 1e-16)
    PR[i][3] = Precision_inter

tab = PR

tab_final = np.zeros(11)
max_p = 0
seuil = 0
print(tab)
for i in range(len(tab)):
    if (max_p < tab[i][3]):
        max_p = tab[i][3]
    max_tmp = max_p
    print("max_tmp: ",max_tmp)
    
    while(seuil <= tab[i][2]):
        print("seuil:",seuil)
        tab_final[int(seuil*10)] = max_tmp
        max_p = 0
        seuil = seuil + 0.1





print(tab_final)