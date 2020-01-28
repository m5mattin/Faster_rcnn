import numpy as np 

# gt = ["a","b","c","z"]
# d = ["b","f","c","b","k","a","a"]

gt = ["a","b","c","z"]
d = ["b","f","c","b","k","a","a"]


gt = ["a","b","c"]
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


recall = 0
tab = []
for i in range(len(PR)):
    if PR[i][2] > recall:
        tab.append((PR[i][0],PR[i][1],PR[i][2],PR[i][3]))
        recall = PR[i][2]

tab_final = np.zeros(11)
max_p = 0

print(PR)
print("  ")
print(np.asarray(tab))


tab = PR
seuil = 0
for i in range(len(tab)):
    if (max_p < tab[i][3]):
        max_p = tab[i][3]
    max_tmp = max_p
    
    while(seuil/10 <= tab[i][2]):
        tab_final[seuil] = max_tmp
        max_p = 0
        seuil = seuil + 1


print("  ")
print(tab_final)
print(sum(tab_final)/len(tab_final))