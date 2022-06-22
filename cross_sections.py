# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:42:57 2022

@author: Tidop
"""

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


alignement = pd.read_csv(r'D:\Road inventario\alignment.txt', delimiter=",").to_numpy() # shape : (n,2)

shift = (-355300,-4502400)
alignement += shift

for step in range(1,2):

    vector = alignement[step:] - alignement[:-step]

    norm = np.linalg.norm(vector, axis=1)
    angles = []
    for i in range(0, len(vector)-step, step):
        u,v = vector[i], vector[i+step]
        angles.append( np.arccos( np.clip(  u @ v / (norm[i] * norm[i+step]) , -1, 1) ) )
        
    print(np.array(angles).mean())


normal_vectors = vector#np.stack( (vector[:,1], - vector[:,0]), axis=1) #shape : (n,2) because the normal vectors are in the XY plane
vector_origin = alignement[:-step]

#plane equation : a(x - x1) + b(y - y1) + c(z - z1) + d = 0

#distance from point to plane : d = | a.px + b.py + c.pz + ( -ax1 -by1 - cz1 )| / np.sqrt( a² + b² + c²)

def distance_to_plane2D(point, normal_vector, point_plane):
    x,y = point[:,0], point[:,1]
    a,b = normal_vector #vector defined only in XY plane
    x0, y0 = point_plane
    d = abs(a * x  + b * y + 0 + (- a * x0 - b * y0)) / np.sqrt( a ** 2 + b ** 2)
    return d

df =  pd.read_csv(r'D:\Road inventario\merged_3cm-005_classifiedV5.txt', delimiter=" ")
cloud = df.to_numpy() # shape : (n,2)

"""
for i in range(0, len(normal_vectors)):
    if i % 10 == 0:
        print(i)
    distance_to_origin = np.linalg.norm(cloud[:,0:2] - vector_origin[i], axis=1)
    sub_cloud = cloud[distance_to_origin < 20]
    distances_to_plane = distance_to_plane2D(sub_cloud[:,0:2], normal_vectors[i], vector_origin[i])
    if len(distances_to_plane) !=0:
        section = sub_cloud[distances_to_plane < 5] # width of demi cross section. 1m gives 2meters large sections
        if len(section) != 0 and i % 5 == 0:    #save sections every 5 points
            #np.savetxt(r'D:\Road inventario\sections_10m\section{}.txt'.format(i), section, fmt="%1.3f", delimiter=' ',comments='', header=' '.join(df.columns))
            np.savetxt(r'D:\Road inventario\sections_10m_vectors\origin{}.txt'.format(i), vector_origin[i], fmt="%1.3f", delimiter=' ',comments='')
            np.savetxt(r'D:\Road inventario\sections_10m_vectors\vector{}.txt'.format(i), normal_vectors[i], fmt="%1.3f", delimiter=' ',comments='')
"""

"""
total_translation = np.array([0,0])
for i in range(1, 800):#len(normal_vectors) -1): #angle is not defined for the last vector
    if i % 10 == 0 : 
        print(i)
    try:
        df =  pd.read_csv(r'D:\Road inventario\sections\section{}.txt'.format(i), delimiter=" ")
    except:
        continue
    cloud = df.to_numpy() 
    #a = angles[i]
    u,v = vector[0], vector[i]
    a =  -np.arccos( np.clip(  u @ v / (norm[0] * norm[i]) , -1, 1) ) 
        
    rotation_matrix = np.array([[ np.cos(a), -np.sin(a)],
                                [np.sin(a), np.cos(a)]])
    rotated_cloud = (rotation_matrix @ cloud[:,0:2].T).T
    rotated_origin = rotation_matrix @ vector_origin[i]
    translation = np.stack(rotated_origin - vector_origin[i])
    total_translation = total_translation + translation
    #translation = rotated_origin -  pos
    #translation += vector_origin[i] - vector_origin[i-1]
    
    translated_cloud = rotated_cloud - total_translation
        
    np.savetxt(r'D:\Road inventario\sections_rotated\section{}.txt'.format(i), np.hstack([translated_cloud, cloud[:,2:]]), fmt="%1.3f", delimiter=' ',comments='', header=' '.join(df.columns))
"""

def fit_lineV2(points, dist_treshold=0.2):
    best = 0
    best_index = -1
    best_inliers = []
    for i in range(1000):
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        
        d=np.cross(p2-p1,points-p1)/np.linalg.norm(p2-p1)
        
        nb_inliers = (np.abs(d) < dist_treshold).sum()
        
        if nb_inliers > best:
            best = nb_inliers
            best_index = idx
            best_inliers = (np.abs(d) < dist_treshold).nonzero()[0]

    return best_inliers, best_index

def fit_line(points, dist_treshold=0.2):
    best = 0
    best_points = None, None
    best_inliers = []
    
    step = int(abs(points[:,1].max() - points[:,1].min()) / 0.05)    
    for i in np.linspace(points[:,1].min(), points[:,1].max() ):
        delta = points[:,0].max() - points[:,0].min()
        step = 0.1
        max_angle = 0.2
        for iteration in range(int(2*(max_angle * delta) // step)):
            
            p1, p2 = np.array([points[:,0].min(),i]), np.array([points[:,0].max(), i])
            #modify p1 position vertically in order to allow a small angle for ransac line
            #we tolerate a line difference of 1 meter every 5 meters
            p1[1] = p1[1] - ((max_angle * delta)) + iteration * step
        
            d=np.cross(p2-p1,points-p1)/np.linalg.norm(p2-p1)
            
            nb_inliers = (np.abs(d) < dist_treshold).sum()
            
            if nb_inliers > best:
                best = nb_inliers
                best_points = [p1,p2]
                best_inliers = (np.abs(d) < dist_treshold).nonzero()[0]

    return best_inliers, best_points

def ransac2D_V1(points, pop_treshold=200, dist_treshold=0.2):
    #plt.figure()
    rest = points
    class_idx = np.zeros(points.shape[0])
    list_tuples = [] #list that will contain 2 points of each line
    idx_in_original_array = np.array(list(range(len(class_idx))))
    for i in range(10):
        if rest.shape[0] < 2:
            continue
        
        inliers, idx = fit_line(rest, dist_treshold=dist_treshold)
        
        if len(inliers) > pop_treshold:
            class_idx[idx_in_original_array[inliers]] = i 
            #plt.scatter(rest[inliers][:,0], rest[inliers][:,1], s=2)
            #plt.plot(rest[idx,0], rest[idx,1], lw=5, c='c')
            list_tuples.append( (rest[idx][0], rest[idx][1]) )
        rest = np.delete(rest, inliers, axis=0)
        idx_in_original_array = np.delete(idx_in_original_array, inliers)
        
    #plt.scatter(rest[:,0], rest[:,1])
    
    return class_idx, list_tuples

def ransac2D_V1(points, pop_treshold=200, dist_treshold=0.2):
    #plt.figure()
    rest = points
    class_idx = np.zeros(points.shape[0])
    list_tuples = [] #list that will contain 2 points of each line
    idx_in_original_array = np.array(list(range(len(class_idx))))
    for i in range(10):
        if rest.shape[0] < 2:
            continue
        
        inliers, best_points = fit_line(rest, dist_treshold=dist_treshold)
        
        if len(inliers) > pop_treshold:
            class_idx[idx_in_original_array[inliers]] = i 
            #plt.scatter(rest[inliers][:,0], rest[inliers][:,1], s=2)
            #plt.plot(rest[idx,0], rest[idx,1], lw=5, c='c')
            list_tuples.append( best_points )
        rest = np.delete(rest, inliers, axis=0)
        idx_in_original_array = np.delete(idx_in_original_array, inliers)
        
    #plt.scatter(rest[:,0], rest[:,1])
    
    return class_idx, list_tuples

def ransac2D(points,pop_treshold=200, dist_treshold=0.2):
    #plt.figure()
    rest = points
    class_idx = -1 * np.ones(points.shape[0])
    list_tuples = [] #list that will contain 2 points of each line
    idx_in_original_array = np.array(list(range(len(class_idx))))
    for i in range(7):
        if rest.shape[0] < 2: #only try if there is at least 2 points
            break
        
        inliers, best_points = fit_line(rest, dist_treshold=dist_treshold)

        #if there is enough intervals
        if len(inliers) > pop_treshold:
            
            # before adding a tuple, we will verify if the new tuple is not too close to one already defining a line
            # especially on the Y-axis as the road is oriented along X-axis            
            if len(list_tuples) != 0:
                diff = np.array(list_tuples) - np.array(best_points)

            #need at least 0.7m between lines
            if len(list_tuples) == 0 or np.abs(diff[:,:,1]).min() > 1:
            
                #assign class
                class_idx[idx_in_original_array[inliers]] = i 
                #add to tuples list
                list_tuples.append( best_points )
                
        rest = np.delete(rest, inliers, axis=0)
        idx_in_original_array = np.delete(idx_in_original_array, inliers)
        
    #plt.scatter(rest[:,0], rest[:,1])
    
    return class_idx, list_tuples

def distance_perp(points, a,b,c,d):
    length_squared = np.sqrt(a**2 + b**2 + c**2)
    x,y,z = points[:,0], points[:,1], points[:,2]
    distance = ((a * x + b * y + c * z + d) / length_squared)
    return distance

def ransac_plane(points, treshold = 0.08):
    best = 0
    best_index = -1
    best_inliers = []
    for iteration in range(100):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]
        
        #compute normal. Its coordinate
        a,b,c = np.cross((p2 - p1) , (p3 - p1))
        
        #orient normal towards z positive
        if c < 0:
            a,b,c = -a,-b,-c
        
        #calculate last coeficient thanks to 1 point belonging to the plane
        d = -a * p1[0] - b * p1[1] -c * p1[2]
        
        distance = distance_perp(points, a, b, c, d)
        
        nb_inliers = (np.abs(distance) < treshold).sum()

        if nb_inliers > best:
            best = nb_inliers
            best_index = idx
            best_inliers = (np.abs(distance) < treshold).nonzero()[0]

    return best_inliers, best_index, (a,b,c,d)

def fit_plane_lstsq(points):
    c = 1
    A = np.stack([ points[:,0], points[:,1], np.ones(len(points)) ], axis=1)
    B = points[:,2]
    a,b,d = - np.linalg.lstsq(A,B)[0]
    return a,b,c,d

def remove_outliers(points, rotate=False, rotation_matrix=None):
    
    index = np.arange(len(points))
    #remove outliers in altitude
    mean, std = points[:,2].mean(), points[:,2].std()
    inliers_z = np.logical_and( points[:,2] < mean + 2*std, points[:,2] > mean - 2*std)
    points = points[inliers_z]
    index = index[inliers_z]
    
    if rotate:
        rotated_cloud = (rotation_matrix @ points[:,:2].T).T 
        mean, std = rotated_cloud[:,1].mean(), rotated_cloud[:,1].std()
        inliers_y = np.logical_and( rotated_cloud[:,1] < mean + 2*std, rotated_cloud[:,1] > mean - 2*std)
        points = points[inliers_y]
        index = index[inliers_y]
    
    return points, index
        
def pts_in_between(points, y1, y2, treshold=2):
    """
    This function supposes that points are rotated such as the road axis is X (the road appears horizontal)

    Parameters
    ----------
    points : np.array
        shape (n,3 + c)
    y1 : int
        DESCRIPTION.
    y2 : int
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    y_min = min(y1,y2)
    y_max = max(y1,y2)
    idx = np.logical_and(y_min - treshold < points[:,1], points[:,1] < y_max + treshold)
    
    return points[idx], idx
    
    
    
def largest_component(points):
    pass
    
path = r'D:\Road inventario\sections_10mV5'
path_vectors = r'D:\Road inventario\sections_10m_vectors'
sections = os.listdir(path)
idi = [float(s.split('.')[0].split('tion')[1]) for s in sections]
orde = np.array(idi).argsort()
sections = np.array(sections)[np.array(idi).argsort()]


df = pd.read_csv(r'D:\Road inventario\parameters_10m.txt', delimiter=", ", names=['col' + str(x) for x in range(30)])
global_param = df.to_numpy()[:,0:7]
ordered = np.array(global_param)[:,1].argsort()
global_param = global_param[ordered]
plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,4])
plt.title('road width (m)')
plt.grid()

plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,5])
plt.title('right shoulder width (m)')
plt.grid()

plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,6])
plt.title('left shoulder width (m)')
plt.grid()

plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,2])
plt.title('lanes number')
plt.grid()
  

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    ax.quiver()
    

parameters = []
barriers_height = []
barriers_distance_road = []
barriers_cloud = np.array([])
display_visual = False
for i,s in enumerate(sections):
    #if int(s.split('.')[0].split('n')[1]) < 1360:# or int(s.split('.')[0].split('n')[1]) > 800:
    #    continue
    if s == 'section285.txt':
        continue
    
    if i % 10 == 0:
        print(i)
    df = pd.read_csv(os.path.join(path,s), delimiter=" ")
    cloud = df.to_numpy()
    prediction_column_index = (df.columns == "prediction").nonzero()[0][0]
    classes = cloud[:,prediction_column_index]

    
        
    
    
    class_idx_road_markings = 1
    road_markings_idx = (cloud[:,prediction_column_index] == class_idx_road_markings).nonzero()[0]
    
    
    

    
    nb = int(s.split('section')[1].split('.')[0])
    vector_normal = pd.read_csv(os.path.join(path_vectors,r'vector{}.txt'.format(nb)), delimiter=" ",header= None).to_numpy().flatten()
    vector_origin = pd.read_csv(os.path.join(path_vectors,r'origin{}.txt'.format(nb)), delimiter=" ",header= None).to_numpy().flatten()

    #purely visual
    if display_visual:
        arrowprops=dict(arrowstyle='->',
                            linewidth=2,
                            shrinkA=0, shrinkB=0)
        plt.figure()
        for p in np.unique(classes):
            plt.scatter(*cloud[classes == p, 0:2].T, s=2)
        ax = plt.gca()
        ax.annotate('', cloud[classes == p, 0:2].mean(axis=0) + vector_normal, cloud[classes == p, 0:2].mean(axis=0), arrowprops=arrowprops)
        plt.draw()
            
    u,v = np.array([1, 0]), vector_normal
    unit_u, unit_v = u, v / np.linalg.norm(v)
    #dot = unit_u @ unit_v
    #a =  -np.arccos( dot  ) 
    
    a = - np.arccos(v[0] / np.linalg.norm(v)) #valid only because we are working with the X axis vector (1,0)
    
    if v[1] < 0: #if the direction vector is oriented in the lower part of the trigonometric circle, the angle should be reversed
        a = -a
        
    rotation_matrix = np.array([[ np.cos(a), -np.sin(a)],
                                [np.sin(a), np.cos(a)]])
    
    
    rotated_2Dcloud = (rotation_matrix @ cloud[:,0:2].T).T
    

    
    
    
    #compute road width
    class_idx_asphalt = 0
    asphalt_idx = (cloud[:,prediction_column_index] == class_idx_asphalt).nonzero()[0]
    clean_asphalt, index_clean = remove_outliers(cloud[asphalt_idx], rotate=True,rotation_matrix=rotation_matrix)
    rotated_2D_asphalt = (rotation_matrix @ clean_asphalt[:,0:2].T).T 
    
    
    if display_visual:
        #purely visual
        plt.figure()
        plt.hist(rotated_2D_asphalt[:,1], bins=40)
        plt.figure()
        for p in np.unique(classes):
            plt.scatter(*rotated_2Dcloud[classes == p].T, s=2)
        
        
        plt.figure()
        plt.title("clean asphalt")
        plt.scatter(clean_asphalt[:,0], clean_asphalt[:,1], s=2)
    #Discard road points that are considered as "outliers" by using the 2*std as criteria
    #mean, std = rotated_2D_asphalt[:,1].mean(), rotated_2D_asphalt[:,1].std()
    #inliers = np.logical_and( rotated_2D_asphalt[:,1] < mean + 2*std, rotated_2D_asphalt[:,1] > mean - 2*std)
    
    y1 = np.quantile(rotated_2D_asphalt[:,1], 0.01)
    y2 = np.quantile(rotated_2D_asphalt[:,1], 0.99)
    
    
    road_width = abs(y2 - y1)
    
    #compute road markings
    if len(rotated_2Dcloud[road_markings_idx,1]) ==0:
        import pdb; pdb.set_trace()
    #mean_markings, std_markings = rotated_2Dcloud[road_markings_idx,1].mean(), rotated_2Dcloud[road_markings_idx,1].std()
    #markings_inliers = np.logical_and( rotated_2Dcloud[road_markings_idx,1] < mean_markings + 2*std_markings, rotated_2Dcloud[road_markings_idx,1] > mean_markings - 2*std_markings)
    _ , markings_inliers = remove_outliers(cloud[road_markings_idx], rotate=True, rotation_matrix=rotation_matrix)
    
    road_markings_idx = road_markings_idx[markings_inliers]
    
    cluster_idx, list_tuples = ransac2D(rotated_2Dcloud[road_markings_idx,0:2])
    rotated_road_markings = rotated_2Dcloud[road_markings_idx,0:2]
    
    if len(list_tuples) <= 1:
        print('section {} was discarded'.format(s))
        continue
    
    if display_visual:
        plt.figure()
        for k,idx in enumerate(np.unique(cluster_idx)):
            plt.scatter(rotated_road_markings[cluster_idx == idx,0], rotated_road_markings[cluster_idx == idx,1])
            if idx != -1:
                rotated_line = np.array(list_tuples[k-1])#np.stack( [rotation_matrix @ list_tuples[k][0], rotation_matrix @ list_tuples[k][1]])
                plt.plot( rotated_line[:,0] ,  rotated_line[:,1], c='c')
        plt.show()
    
    list_pts_projected = []
    for (pointA, pointB) in list_tuples:
        
        ptA_projected = rotation_matrix @ pointA
        ptB_projected = rotation_matrix @ pointB
        
        slope = ( ptB_projected[1] - ptA_projected[1] ) / ( ptB_projected[0] - ptA_projected[0] )
        origin = ptA_projected[1] - slope * ptA_projected[0]
        
        if abs(slope) > 1:  #discard if one line is too vertical
            continue
        
        list_pts_projected.append(( min(rotated_road_markings[:,0]), slope * min(rotated_road_markings[:,0]) + origin))
    list_pts_projected.sort()
    list_pts_projected = np.array(list_pts_projected)
    distances = list_pts_projected[1:] - list_pts_projected[:-1]
    distances = distances[:,1]
    
    right_shoulder = abs(y1 - list_pts_projected[0,1])
    left_shoulder = abs(y2 - list_pts_projected[-1,1])
    
    
    #compute road superelevation
    side1 = np.array([rotated_2D_asphalt[:,0].mean(),y1])
    side2 = np.array([rotated_2D_asphalt[:,0].mean(),y2])
    
    closest1 = asphalt_idx[np.linalg.norm( rotated_2D_asphalt[:] - side1, axis=1 ).argmin()]
    closest2 = asphalt_idx[np.linalg.norm( rotated_2D_asphalt[:] - side2, axis=1 ).argmin()]
    
    import pdb; pdb.set_trace()
    elevation = (cloud[closest1, 2] - cloud[closest2,2]) / road_width # delta_z / road_width
    
    
    
    #barriers
    class_idx_barriers = 4
    barriers_idx = (cloud[:,prediction_column_index] == class_idx_barriers).nonzero()[0]
    
    _, idx = pts_in_between(rotated_2Dcloud[barriers_idx], y1,y2)
    barriers_idx = barriers_idx[idx]
    
    clean_barriers, index_clean_barriers = remove_outliers(cloud[barriers_idx], rotate=True, rotation_matrix=rotation_matrix)
    barriers_idx = barriers_idx[index_clean_barriers]
    
    #clean_barriers, index_clean_barriers = remove_outliers()
    cluster_idx, list_tuples = ransac2D(rotated_2Dcloud[barriers_idx,0:2])#cloud[barriers_idx,:2])
    rotated_barriers = rotated_2Dcloud[barriers_idx,0:2]
    
    
    
    if display_visual and len(rotated_barriers) != 0 and len(list_tuples) != 0:
        plt.figure()
        for k,idx in enumerate(np.unique(cluster_idx)):
            plt.scatter(rotated_barriers[cluster_idx == idx,0], rotated_barriers[cluster_idx == idx,1])
            if idx != -1:
                rotated_line = np.array(list_tuples[k-1])#np.stack( [rotation_matrix @ list_tuples[k][0], rotation_matrix @ list_tuples[k][1]])
                plt.plot( rotated_line[:,0] ,  rotated_line[:,1], c='c')
        plt.title("barriers")
        plt.show()
    
    
    #fit plane to road section
    #inliers, index, (a,b,c,d) = ransac_plane(cloud[asphalt_idx,0:3])
    
    a,b,c, d = fit_plane_lstsq(clean_asphalt)
                  
    
    #compute barriers height
    distance_barriers_from_ground = distance_perp(cloud[barriers_idx,0:3], a,b,c,d)
    barriers_altitude = []
    for k,idx in enumerate(np.unique(cluster_idx)):
        if idx != -1:
            dist = distance_perp(cloud[barriers_idx][cluster_idx == idx,0:3], a,b,c,d)
            dist = np.clip(dist, -1,2)
            barriers_altitude.append(np.quantile(dist, 0.95))
            
    #save barriers point to a point cloud
    
    idx_save = (cloud[:,prediction_column_index] == class_idx_barriers).nonzero()[0]
    #import pdb; pdb.set_trace()
    #temp_cloud = np.hstack([cloud[idx], np.nan * np.ones((len(idx),1)), -1 * np.ones((len(idx),1))])
    temp_cloud = np.hstack([cloud, np.nan * np.ones((len(cloud),1)), -1 * np.ones((len(cloud),1))])
    for k,idx in enumerate(np.unique(cluster_idx)):
        if idx != -1:
            temp_cloud[barriers_idx[cluster_idx == idx], -1] = idx
            dist = distance_perp(cloud[barriers_idx][cluster_idx == idx,0:3], a,b,c,d)
            dist = np.clip(dist, -1,2)
            temp_cloud[barriers_idx[cluster_idx == idx], -2] = np.quantile(dist, 0.95)
    if len(barriers_cloud) == 0:
        barriers_cloud = temp_cloud[idx_save]
    else:
        barriers_cloud = np.vstack([barriers_cloud, temp_cloud[idx_save]])
        
    
    #compute barriers distance to the road
    distance_to_road = []
    for k,idx in enumerate(np.unique(cluster_idx)):
        if idx != -1:
            x_mean = rotated_barriers[cluster_idx == idx,0].mean()
            (p1,p2) = list_tuples[k -1]
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            origin = p1[1] - p1[0] * slope 
            y_barrier = x_mean * slope + origin
            distance_to_road.append( min( abs(y1 - y_barrier), abs(y2 - y_barrier)) ) 
    barriers_distance_road.append(distance_to_road)
                
    barriers_height.append([*barriers_altitude])
    parameters.append([i, nb, len(distances), elevation, road_width, right_shoulder, left_shoulder, *distances])
    
    fix_parameters = np.array([p[0:7] for p in parameters])
    ordered = np.array(fix_parameters)[:,1].argsort()
    fix_parameters = fix_parameters[ordered]
    variable_parameters = np.array([p[7:] for p in parameters])
    variable_parameters = variable_parameters[ordered]
    
    #import pdb; pdb.set_trace()
    #if display_visual:
    #    plt.figure()
    #    plt.plot( np.array(fix_parameters)[:,1], np.array(fix_parameters)[:,4])
    #    plt.grid()
    
    #to read back
    #df = pd.read_csv(r'D:\Road inventario\parameters_10m.txt', delimiter=", ", names=['col' + str(x) for x in range(30)])
    
#with open(r'D:\Road inventario\parameters_10m.txt', 'w') as f :
#    for p in parameters:
#        f.write(str(p)[1:-1] + '\n')

global_params = fix_parameters
plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,4])
plt.title('road width (m)')
plt.grid()

plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,5])
plt.title('right shoulder width (m)')
plt.grid()

plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,6])
plt.title('left shoulder width (m)')
plt.grid()

plt.figure()
plt.plot( np.array(global_param)[:,1], np.array(global_param)[:,2])
plt.title('lanes number')
plt.grid()


longueur = [len(h) for h in barriers_height]
first = []
second = []
for i in range(len(longueur)):
    first.append( barriers_height[i][0] if longueur[i] > 0 else 0)
    second.append(barriers_height[i][1] if longueur[i] > 1 else 0)
plt.figure()
plt.plot(list(range(290,1700,5)), first)
plt.plot(list(range(290,1700,5)), second)


def smooth_gaussian(points, width=3):
    smoothed = points[:width//2].tolist() 
    from scipy import signal 
    window = signal.windows.gaussian(width, std=1) 
    window = window / window.sum()
    for i in range(width // 2, len(points) - width // 2):
        smoothed.append( (window * points[ i - width //2 : i + width // 2 + 1]).sum() )
    
    smoothed.extend( points[-width//2 + 1:].tolist())
    return smoothed

sfirst = smooth_gaussian(np.array(first))
ssecond = smooth_gaussian(np.array(second))

plt.figure()
plt.plot(list(range(290,1700,5)), sfirst)
plt.plot(list(range(290,1700,5)), ssecond)



longueur = [len(h) for h in barriers_distance_road]
first = []
second = []
for i in range(len(longueur)):
    first.append( barriers_distance_road[i][0] if longueur[i] > 0 else 0)
    second.append( barriers_distance_road[i][1] if longueur[i] > 1 else 0)

plt.figure()
plt.plot(list(range(290,1700,5)), first)
plt.plot(list(range(290,1700,5)), second)

#TODO IMPROVEMENTS :
#   filters to discard obvious errors (height filters, for cars for instance)
#   use the road asphalt to compute a vector or even the trajectory (the one from Mario is good but what if people don't have a trajectory centerline ?)
#   use the trajectory of the vehicle directly ?
#   compute the statistics on the cross section of the vector, not on min, max etc... (exemple of rotated markings)
#   In ransac, give a higher reward when fitting a line with 2 points distants (because the longer the line, the more reliable the axis is in theory, but beware of curb roads)
#   In ransac, use the mean of the points to compute the distances, otherwise a distance can vary according to from which side of a road marking we begin the measure
#   check for other means to compute the edge of the road, 0.01 and 0.99 quantiles are quite arbitrary treshold...
#    

# errors seen :
#   right_shoulder : values up to 2000 meters (!!!). Example at section 795, 200 meters. Error from the abs computation ?
#                   also errors on section 320, 8m for instance while this should be a simple case. Investigate. 1565 aussi
#   road_width : the 99 quantile poses problem on the section 1040 for instance. Moreover, tvery first sections give width a bit larger (12m) than should be (10m)
#                    other exampel with section 320 where outliers in road points give false results
#
#
#
# errors fixed:
#   problem on the rotation, the angle had to be separated in two cases, with vector in the upper part and vector in the lower part of the trigo circle
#   I was doing cloud @ rotation.T au lieu (rotation @ cloud.T).T
#   discard outliers from asphalt with a 2*std criteria before using quantile
#   
#   errors sources :
#       places with joining lanes
#       too much markings on the ground. Especially zebra, gives vertical results sometimes...
#
#
#
#
#
#
  






    
    

