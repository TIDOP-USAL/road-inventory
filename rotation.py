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



#-----COMPUTE VECTORS FROM ALIGNEMENT
alignement = pd.read_csv(r'D:\Road inventario\alignment.txt', delimiter=",").to_numpy() # shape : (n,2)

shift = (-355300,-4502400)
alignement += shift

for step in range(1,2):

    vector = alignement[step:] - alignement[:-step]

    norm = np.linalg.norm(vector, axis=1)
    angles = []
    for i in range(0, len(vector)-step, step):
        u,v = vector[i], vector[i+step]
        #angles.append( np.arccos( np.clip(  u @ v / (norm[i] * norm[i+step]) , -1, 1) ) )
        
    print(np.array(angles).mean())


normal_vectors = vector#np.stack( (vector[:,1], - vector[:,0]), axis=1) #shape : (n,2) because the normal vectors are in the XY plane
vector_origin = alignement[:-step]

def distance_to_plane2D(point, normal_vector, point_plane):
    x,y = point[:,0], point[:,1]
    a,b = normal_vector #vector defined only in XY plane
    x0, y0 = point_plane
    d = abs(a * x  + b * y + 0 + (- a * x0 - b * y0)) / np.sqrt( a ** 2 + b ** 2)
    return d


#---------------------CREATE SECTIONS

df =  pd.read_csv(r'D:\InRoads\Avila\classified\all_merged_light.txt', delimiter=' ')
cloud = df.to_numpy() # shape : (n,2)

path = r'D:\Road inventario\full_road'


for i in range(0, len(normal_vectors)):
    if i % 10 == 0:
        print(i)
    distance_to_origin = np.linalg.norm(cloud[:,0:2] - vector_origin[i], axis=1)
    sub_cloud = cloud[distance_to_origin < 20]
    distances_to_plane = distance_to_plane2D(sub_cloud[:,0:2], normal_vectors[i], vector_origin[i])
    if len(distances_to_plane) !=0:
        section = sub_cloud[distances_to_plane < 5] # width of demi cross section. 1m gives 2meters large sections
        if len(section) != 0 and i % 5 == 0:    #save sections every 5 points
            np.savetxt(os.path.join(path,r'sections_10mV5\section{}.txt'.format(i)), section, fmt="%1.3f", delimiter=' ',comments='', header=' '.join(df.columns))
            # already done :
            np.savetxt(os.path.join(path,r'sections_10m_vectors\origin{}.txt'.format(i)), vector_origin[i], fmt="%1.3f", delimiter=' ',comments='')
            np.savetxt(os.path.join(path,r'sections_10m_vectors\vector{}.txt'.format(i)), normal_vectors[i], fmt="%1.3f", delimiter=' ',comments='')
            pass

#df =  pd.read_csv(r'D:\Road inventario\merged_3cm-005_classifiedV5.txt', delimiter=" ")
#cloud = df.to_numpy() # shape : (n,2)




#-------------------TRANSLATE AND ROTATE THESE SECTIONS
path_vectors = os.path.join(path, r'sections_10m_vectors')
path_sections = os.path.join(path,r'sections_10mV5')
sections = os.listdir(path_sections)

ordered = [int(s.split('n')[1].split('.')[0]) for s in sections]
ordered = np.array(ordered).argsort()
sections = np.array(sections)[ordered]

parameters = []
display_visual = True

resulting_vector_origin = []
for i,s in enumerate(sections):
    if i % 10 == 0:
        print(i)
    df = pd.read_csv(os.path.join(path_sections,s), delimiter=" ")
    cloud = df.to_numpy()

    nb = int(s.split('section')[1].split('.')[0])
    vector_normal = pd.read_csv(os.path.join(path_vectors,r'vector{}.txt'.format(nb)), delimiter=" ",header= None).to_numpy().flatten()
    vector_origin = pd.read_csv(os.path.join(path_vectors,r'origin{}.txt'.format(nb)), delimiter=" ",header= None).to_numpy().flatten()
            
    u,v = np.array([1, 0]), vector_normal
    unit_u, unit_v = u, v / np.linalg.norm(v)
    #dot = unit_u @ unit_v
    #a =  -np.arccos( dot  ) 
    
    a = - np.arccos(v[0] / np.linalg.norm(v)) #valid only because we are working with the X axis vector (1,0)
    a += np.pi
    if v[1] < 0: #if the direction vector is oriented in the lower part of the trigonometric circle, the angle should be reversed
        a = -a
        
    rotation_matrix = np.array([[ np.cos(a), -np.sin(a)],
                                [np.sin(a), np.cos(a)]])
    
    
    backup_cloud = np.copy(cloud)
    cloud = cloud[:,0:2]
    if s != 'section0.txt':
        
        
        
        precedent_vector_origin = pd.read_csv(os.path.join(path_vectors,r'origin{}.txt'.format(nb-5)), delimiter=" ",header= None).to_numpy().flatten()
        #precedent_vector_normal = pd.read_csv(os.path.join(path_vectors,r'vector{}.txt'.format(nb-5)), delimiter=" ",header= None).to_numpy().flatten()
        
        distance = np.linalg.norm(precedent_vector_origin - vector_origin)
        if vector_origin[0] - precedent_vector_origin[0] <0:
            distance = -distance
        
        expected_pos = np.array([resulting_vector_origin[-1][0] + distance, 0])
        translation_vector = expected_pos - vector_origin
        
        resulting_vector_origin.append(expected_pos)
        
        cloud = cloud + translation_vector
        

        rotated_2Dcloud = (rotation_matrix @ cloud[:,0:2].T).T
        rotated_vector_origin = rotation_matrix @ expected_pos
        
        delta = expected_pos - rotated_vector_origin
        
        
        result = rotated_2Dcloud + delta
        print(result[:,1].mean())
        
        
        #Avoid overlapping
        idx = np.abs((rotated_2Dcloud - rotated_vector_origin )[:,0]) < 2.5
        
        result = result[idx]
        backup_cloud = backup_cloud[idx]
    else:
        #resulting_vector_origin.append(rotation_matrix @ vector_origin)
        resulting_vector_origin.append( np.array([vector_origin[0], 0]) ) 
        cloud = cloud - np.array([0, vector_origin[1]])
        result = cloud
        
        
    np.savetxt(os.path.join(path,r'sections_rotatedV5\{}'.format(s)), np.hstack([result, backup_cloud[:,2:]]), fmt="%1.3f", delimiter=' ',comments='', header=' '.join(df.columns))

    
  






    
    

