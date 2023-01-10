import os

mapping = []
fit = []
with open("design_points.txt", "r") as f:
    i = 0
    while i < 1e7:
        d = f.readline()
        if d is None: break
        d = d.split(",")
        if len(d)==1: break
        i += 1
        #print(d[0])
        mapping.append([float(l) for l in d[1:]])
        fit.append(-float(d[0]))
    print(i)
    print(mapping[0][0:22])
    print(mapping[0][23:26])

top_fits = sorted(fit)[:int(len(fit)*0.8)]
mapping_selected = []
fit_selected = []
for i in range(len(fit)):
    if fit[i] <= top_fits[-1]: #fast design
        mapping_selected.append(mapping[i])
        fit_selected.append(fit[i])

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# 데이터셋 로드

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# class target 정보 제외
#train_df = df[['sepal length', 'sepal width', 'petal length', 'petal width']]

# 2차원 t-SNE 임베딩

'''
df = pd.DataFrame(data= np.asarray(mapping_selected))
print(df)
#tsne_np = TSNE(n_components = 2).fit_transform(df)
tsne_np = PCA(n_components = 1).fit_transform(df)
#tsne_np = [d for d in fit]
#tsne_np1, tsne_np2 = [l[0] for l in tsne_np], [l[1] for l in tsne_np]
'''

#df1 = pd.DataFrame(data= np.asarray([l[:22] for l in mapping_selected]))
#df1 = pd.DataFrame(data= np.asarray([[2**l[2]]+l[5:7]+l[8:12]+[2**l[14],2**l[15]]+l[16:20]+[l[22]] for l in mapping_selected]))
import math
#df1 = pd.DataFrame(data= np.asarray([[2**l[2]]+l[5:7]+l[8:12]+[2**l[14],2**l[15]]+[ll for ll in l[16:20]]+[l[22]] for l in mapping_selected]))
hw_map = [[2**l[2]]+l[5:7]+l[8:12]+[2**l[14],2**l[15]]+[ll for ll in l[16:20]]+[l[22]] for l in mapping_selected]
for j in range(len(hw_map[0])):
    maxi = max([hw_map[i][j] for i in range(len(hw_map))]) + 0.000001
    for i in range(len(hw_map)):
        hw_map[i][j] /= maxi
    
df1 = pd.DataFrame(data= np.asarray(hw_map))
print(df1)
df2 = pd.DataFrame(data= np.asarray([l[23:] for l in mapping_selected]))
print(df2)
#tsne_np1 = TSNE(n_components = 1).fit_transform(df1)
#tsne_np2 = TSNE(n_components = 1).fit_transform(df2)
tsne_np1 = PCA(n_components = 1).fit_transform(df1)
tsne_np2 = PCA(n_components = 1).fit_transform(df2)
#tsne_np = [d for d in fit]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# target 별 시각화
'''
plt.scatter(tsne_np, fit_selected, color = 'pink')#, label = 'setosa')
plt.xlabel('component hw')
plt.legend()
plt.show()
'''

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

col = [0 if e==min(fit_selected) else 1 for e in fit_selected]

ax1.scatter(tsne_np1, fit_selected, c=col, alpha=0.5)#, label = 'setosa')
#ax1.xlabel('component hw')
#plt.legend()
#plt.show()

ax2.scatter(tsne_np2, fit_selected, c=col,alpha=0.5)#, label = 'setosa')
#ax2.xlabel('component mapping')
#plt.legend()
plt.show()

'''
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_np1, tsne_np2, fit_selected, c=fit_selected, s=1, alpha=0.5)
plt.xlabel('component hw')
plt.ylabel('component mapping')
plt.show()
'''

'''
# numpy array -> DataFrame 변환
#tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])

import matplotlib.pyplot as plt

plt.hist(fit_selected, bins=500)

plt.show()
'''
