from pyoctree import pyoctree as ot
import trimesh
import numpy as np
#Point coordinates is an Nx3 array representing the coordinates of the points
#connectivity is an Nx3 integer array describing which points are connected
mesh = trimesh.load_mesh('transectShape.ply')

f = np.array(mesh.faces).astype(np.int32)
v = np.array(mesh.vertices)


minX = np.min(v[:,0])
minY = np.min(v[:,1])
v[:,0] += -minX
v[:,1] += -minY

pointCoords = v
connectivity = f

tree = ot.PyOctree(pointCoords, connectivity)


# Choose some starting points, for example the positions of the cameras
import pandas as pd

pose = pd.read_csv('C:/Users/haavasl/PycharmProjects/Testing/venv/Code/nav_file_for_tautra_new.txt')

X = pose['X'].values.reshape((-1,1))
Y = pose['Y'].values.reshape((-1,1))
Z = pose['Z'].values.reshape((-1,1))

xyz_start = np.zeros((X.shape[0], 3), dtype=np.float32)
xyz_end = np.zeros((X.shape[0], 3), dtype=np.float32)

xyz_start[:,0] = X[:,0] - minX
xyz_end[:,0] = X[:,0] - minX

xyz_start[:,1] = Y[:,0] - minY
xyz_end[:,1] = Y[:,0] - minY

xyz_start[:,2] = Z[:,0] + 5
xyz_end[:,2] = Z[:,0] - 5
pos_arr = np.zeros((X.shape[0], 3))
#rayList = np.array([xyz_start, xyz_end], dtype=np.float32)

#intersectionFound = tree.rayIntersection(rayList[:,1000:1100,:])

import time
t1 = time.time()
for i in range(X.shape[0]):
    rayList = np.array([xyz_start[i,:], xyz_end[i,:]], dtype=np.float32)
    intersectionFound = tree.rayIntersection(rayList)
    if len(intersectionFound) != 0:
        pos = intersectionFound[0].p
        pos_arr[i,:] = pos
t2 = time.time()
#
tot = t2-t1
import matplotlib.pyplot as plt
plt.plot(pos_arr[pos_arr[:,1] > 0, 0], pos_arr[pos_arr[:,1] > 0, 1])
plt.show()
arr = pos_arr[pos_arr[:,1] > 0, 0]
print()
#np.save(RayList)