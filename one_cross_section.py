# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:02:26 2022

@author: Tidop
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\Road inventario\rasterization.txt', delimiter=" ")#filtered_markings.txt #straightened_roadV5.txt', delimiter=" ")
cloud = df.to_numpy() # shape : (n,2)

"""
def fit_line_V1(points):
    best = 0
    best_points = None, None
    best_inliers = []
    dist_treshold = 0.2
    #generate points along Y axis with step of 0.1m
    for i in np.linspace(points[:,1].min(), points[:,1].max(), int(abs(points[:,1].max() - points[:,1].min()) / 0.1)):
        p1, p2 = np.array([points[:,0].min(),i]), np.array([points[:,0].max(), i])
        
        d=np.cross(p2-p1,points-p1)/np.linalg.norm(p2-p1)
        
        nb_inliers = (np.abs(d) < dist_treshold).sum()
        
        if nb_inliers > best:
            best = nb_inliers
            best_points = p1, p2
            best_inliers = (np.abs(d) < dist_treshold).nonzero()[0]

    return best_inliers, best_points

def fit_line_V2(points):
    best = 0
    best_points = None, None
    best_inliers = []
    dist_treshold = 0.35
    #generate points along Y axis with step of 0.1m
    for i in np.linspace(points[:,1].min(), points[:,1].max(), int(abs(points[:,1].max() - points[:,1].min()) / 0.1)):
        p1, p2 = np.array([points[:,0].min(),i]), np.array([points[:,0].max(), i])
        
        d=np.cross(p2-p1,points-p1)/np.linalg.norm(p2-p1)
        
        is_near = (np.abs(d) < dist_treshold)
        idx_near = is_near.nonzero()[0]
        
        # Under the hypothesis of a road along X-axis, instead of 
        # considering the raw number of inliers under a certain distance treshold
        # we group them by interval and count the number of interval
        # this allows to avoid very dense concentration of points influencing the line
        interval = 0.3
        nb_grouped_inliers = len(np.unique(points[idx_near, 0] // interval))

        if nb_grouped_inliers > best:
            best = nb_grouped_inliers
            best_points = p1, p2
            best_inliers = idx_near

    return best_inliers, best_points
"""
def fit_line_V3(points):
    best = 0
    best_points = None, None
    best_inliers = []
    dist_treshold = 0.2
    #generate points along Y axis with step of 0.1m
    #TODO : take care, the minimum and maximum of points is moving as we remove points from the dataset each time we encounter a line. Oh well whatever in fact, the step is still 0.1m
    step = int(abs(points[:,1].max() - points[:,1].min()) / 0.1)
    for i in np.linspace(points[:,1].min(), points[:,1].max(), num=step):
        for iteration in range(10):
            
            p1, p2 = np.array([points[:,0].min(),i]), np.array([points[:,0].max(), i])
            
            #modify p1 position vertically in order to allow a small angle for ransac line
            p1[1] = p1[1] - 1 + iteration * 0.2
            
        
            d=np.cross(p2-p1,points-p1)/np.linalg.norm(p2-p1)
            
            is_near = (np.abs(d) < dist_treshold)
            idx_near = is_near.nonzero()[0]
            
            # Under the hypothesis of a road along X-axis, instead of 
            # considering the raw number of inliers under a certain distance treshold
            # we group them by interval and count the number of interval
            # this allows to avoid very dense concentration of points influencing the line
            interval = 0.3
            nb_grouped_inliers = len(np.unique(points[idx_near, 0] // interval))
    
            if nb_grouped_inliers > best:
                best = nb_grouped_inliers
                best_points = p1, p2
                best_inliers = idx_near
    
    return best_inliers, best_points

"""
def ransac2D_V1(points):
    #plt.figure()
    rest = points
    class_idx = np.zeros(points.shape[0])
    list_tuples = [] #list that will contain 2 points of each line
    pop_treshold = 30000
    idx_in_original_array = np.array(list(range(len(class_idx))))
    for i in range(10):
        print("one")
        if rest.shape[0] < 2:
            continue
        
        inliers, best_points = fit_line_V1(rest)

        if len(inliers) > pop_treshold:
            class_idx[idx_in_original_array[inliers]] = i 
            #plt.scatter(rest[inliers][:,0], rest[inliers][:,1], s=2)
            #plt.plot(rest[idx,0], rest[idx,1], lw=5, c='c')
            list_tuples.append( best_points )
        rest = np.delete(rest, inliers, axis=0)
        idx_in_original_array = np.delete(idx_in_original_array, inliers)
        
    #plt.scatter(rest[:,0], rest[:,1])
    
    return class_idx, list_tuples
"""
def ransac2D_V2(points):
    #plt.figure()
    rest = points
    class_idx = -1 * np.ones(points.shape[0])
    list_tuples = [] #list that will contain 2 points of each line
    pop_treshold = 300
    idx_in_original_array = np.array(list(range(len(class_idx))))
    for i in range(7):
        print("iter")
        if rest.shape[0] < 2: #only try if there is at least 2 points
            break
        
        inliers, best_points = fit_line_V3(rest)

        #if there is enough intervals
        print(len(np.unique(rest[inliers, 0] // 0.3)))
        if len(np.unique(rest[inliers, 0] // 0.3)) > pop_treshold:
            
            # before adding a tuple, we will verify if the new tuple is not too close to one already defining a line
            # especially on the Y-axis as the road is oriented along X-axis            
            if len(list_tuples) != 0:
                diff = np.array(list_tuples) - np.array(best_points)
            import pdb; pdb.set_trace()
            if len(list_tuples) == 0 or np.abs(diff[:,:,1]).min() > 0.7:
            
                #assign class
                class_idx[idx_in_original_array[inliers]] = i 
                #add to tuples list
                list_tuples.append( best_points )
                
        rest = np.delete(rest, inliers, axis=0)
        idx_in_original_array = np.delete(idx_in_original_array, inliers)
        
    #plt.scatter(rest[:,0], rest[:,1])
    
    return class_idx, list_tuples

#get the index of the pts classified as road markings 
classes = cloud[:,(df.columns == "prediction").nonzero()[0][0]]
class_markings = 1 #Value corresponding to the class "road markings". Is dependant on how was classified the dataset
idx_markings = (classes == class_markings).nonzero()[0]



class_idx, list_tuples = ransac2D_V2(cloud[idx_markings,0:2])

#realign the translation elements

cloud_index = cloud[:,(df.columns == "Original_cloud_index").nonzero()[0][0]]
realigned_cloud = np.copy(cloud)
for identifiant in np.unique(cloud_index):
    if identifiant % 0 == 0:
        print(identifiant)
    
    sub_cloud = cloud[ ( np.logical_and(cloud_index == identifiant, classes == class_markings)).nonzero()[0]]
    p1,p2 = list_tuples[0]
    slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
    origin = p1[1] - slope * p1[0]
    
    
    d=np.cross(p2-p1,sub_cloud[:,0:2]-p1)/np.linalg.norm(p2-p1)
    is_near = (np.abs(d) < 0.6)
    sub_cloud = sub_cloud[is_near]
    idx_near = is_near.nonzero()[0]
                
    
    Ty = np.linalg.lstsq( -1 * np.ones((is_near.sum(),1)) ,sub_cloud[:,1] - sub_cloud[:,0] * slope - origin)[0]
    realigned_cloud[(cloud_index == identifiant).nonzero()[0],1] = realigned_cloud[(cloud_index == identifiant).nonzero()[0],1] + Ty

    #plt.figure()
    #plt.scatter(sub_cloud[:,0], sub_cloud[:,1], s=1)
    #plt.scatter(sub_cloud[:,0], sub_cloud[:,1] + Ty, s=1)
    
# plt.figure()
# for identifiant in np.unique(cloud_index):
#     try:
#         sub_cloud = cloud[ (np.logical_and(cloud_index == identifiant, classes == class_markings)).nonzero()[0]]
#         bool_arr = np.random.choice(len(sub_cloud), size=100)
#         plt.scatter(sub_cloud[bool_arr,0], sub_cloud[bool_arr,1], s=1)
#     except:
#         continue
    
plt.figure()
for classe in np.unique(class_idx):
    bool_arr = (classe == class_idx).nonzero()[0]
    bool_arr = np.random.choice(bool_arr, size=20000)
    plt.scatter(realigned_cloud[idx_markings,0][bool_arr], realigned_cloud[idx_markings,1][bool_arr], s=1)
  

#road markings class

plt.figure()
for classe in np.unique(class_idx):
    bool_arr = (classe == class_idx).nonzero()[0]
    bool_arr = np.random.choice(bool_arr, size=20000)
    plt.scatter(cloud[idx_markings,0][bool_arr], cloud[idx_markings,1][bool_arr], s=1)

    
""" 
class_road = 0
idx_road = (classes == class_road).nonzero()[0]
plt.figure()
plt.hist(cloud[idx_road,1], bins=50)

plt.figure()
bool_arr = np.random.choice(len(cloud[idx_road]), size=min(20000,len(cloud[idx_road])) )
plt.scatter(cloud[idx_road][bool_arr,0], cloud[idx_road][bool_arr,1], s=1)


 
#barrerras class

plt.figure()
for classe in np.unique(class_idx):
    bool_arr = (classe == class_idx).nonzero()[0]
    bool_arr = np.random.choice(bool_arr, size=20000)
    plt.scatter(cloud[idx_markings,0][bool_arr], cloud[idx_markings,1][bool_arr], s=1)
"""
    
"""    
class_barriers = 4
idx_barriers = (classes == class_barriers).nonzero()[0]
plt.figure()
plt.hist(cloud[idx_barriers,1], bins=50)

plt.figure()
bool_arr = np.random.choice(len(cloud[idx_barriers]), size=min(len(idx_barriers),20000))
plt.scatter(cloud[idx_barriers][bool_arr,0], cloud[idx_barriers][bool_arr,1], s=1)

"""