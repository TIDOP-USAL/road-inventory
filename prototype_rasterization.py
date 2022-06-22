
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:46:42 2022

@author: Tidop
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import time

path = r'D:\Road inventario\full_road'
file = r'straightened_full_road.txt' #'straightened_roadV5.txt' #'merged_3cm-005_classified.txt'


df = pd.read_csv(os.path.join(path,file) , delimiter=" ")
cloud = df.to_numpy() # shape : (n,2)

classes = cloud[:,(df.columns == "prediction").nonzero()[0][0]]
id_class = 1 #markings
idx = (classes == id_class).nonzero()[0]
cloud = cloud[idx]

#bool_arr = np.random.choice(len(cloud), size=20000)
#cloud = cloud[bool_arr]



def pts_to_image(points, step=0.1):
    start_time = time.time()
    density = 1 /step
    rounded = np.round(points * density)
    uniques, counts = np.unique(rounded[:,0:2], axis=0, return_counts=True)
    
    deltaX = rounded[:,0].max() - rounded[:,0].min() + 1
    deltaY = rounded[:,1].max() - rounded[:,1].min() + 1
    img = np.zeros((int(deltaX), int(deltaY)))
    
    uniques = uniques - [rounded[:,0].min(), rounded[:,1].min()]
    
    #import pdb; pdb.set_trace()
    print(len(uniques))
    
    big_samples = counts > 2#(0.05 * (step / 0.03) ** 2)
    uniques, counts = uniques[big_samples], counts[big_samples]
    print(len(uniques))
    
    pixel_coords = (rounded - [rounded[:,0].min(), rounded[:,1].min()]).astype(np.int)
    for k in range(len(uniques)):
        if k % 10000 == 0:
            print(k)
            print("time : ", time.time() - start_time)
        x,y = uniques[k]
        count = counts[k]
        img[ int(x), int(y) ] = 1#count
        #idx_to_keep.extend( np.where( (pixel_coords[:,0] == x) & (pixel_coords[:,1] == y) )[0].tolist())    
        
        #idx_to_keep.extend( np.where((normalized_rounded == [x,y]).all(axis=1))[0] )
        #idx_to_keep.extend( (normalized_rounded == [x,y]).all(axis=1).nonzero()[0] )
        #idx_to_keep = np.append(idx_to_keep, np.where( (rounded[:,0] - rounded[:,0].min() == x) & (rounded[:,1] - rounded[:,1].min() == y) ))
    end_time = time.time()
    print("time : ", end_time-start_time)
    return np.round((img / img.max()) * 255), pixel_coords

def image_to_pts(points, bool_img, pixel_coords, step=0.1):
    
    pts_to_be_kept = bool_img[pixel_coords[:,0], pixel_coords[:,1]]
    return points[pts_to_be_kept.nonzero()[0]]
    """
    x,y = np.where(bool_img)
    to_be_kept = np.stack((x,y),axis=1)
    import pdb; pdb.set_trace()
    
    density = 1 /step
    rounded = np.round(points * density)
    rounded = rounded[:,0:2] - [rounded[:,0].min(), rounded[:,1].min()]

    #sub_points = points[ np.isin(rounded, to_be_kept).sum(axis=1) == 2]
    bool_arr = np.full((len(points), 1), True)
    liste = to_be_kept.tolist()
    print(points.shape[0])
    for i,row in enumerate(points):
        if i % 10000 ==0:
            print(i)
        bool_arr[i] = row[:2].tolist() in liste
    
    
    return points[bool_arr]
    """

img, pixel_coords = pts_to_image(cloud[:,0:2], step=0.1)
plt.figure()
plt.imshow(img.T, cmap='gray')


from scipy import ndimage
dx = ndimage.sobel(img,1)
dx_high = np.abs(np.copy(dx))
dx_high[dx_high < 760] = 0

plt.figure()
plt.imshow(dx_high.T, cmap='gray')


sub_pts = image_to_pts(cloud, dx_high > 10,pixel_coords, step=2)
plt.figure()
bool_arr = np.random.choice(len(sub_pts), size=20000)
plt.scatter(sub_pts[bool_arr,0], sub_pts[bool_arr,1], s=1)

plt.figure()
bool_arr = np.random.choice(len(cloud), size=20000)
plt.scatter(cloud[bool_arr,0], cloud[bool_arr,1], s=1)

np.savetxt(r'rasterization.txt', sub_pts, fmt='%1.3f')