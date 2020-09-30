import vtk
import numpy as np
## Code from https://blog.kitware.com/ray-casting-ray-tracing-with-vtk/
def isHit(obbTree, pSource, pTarget):
  code = obbTree.IntersectWithLine(pSource, pTarget, None, None)
  if code==0:
    return False
  return True

def GetIntersect(obbTree, pSource, pTarget):
  points = vtk.vtkPoints()
  cellIds = vtk.vtkIdList()
  # Perform intersection test
  code = obbTree.IntersectWithLine(pSource, pTarget, points, cellIds)

  pointData = points.GetData()
  noPoints = pointData.GetNumberOfTuples()
  noIds = cellIds.GetNumberOfIds()

  pointsInter = []
  cellIdsInter = []
  for idx in range(noPoints):
    pointsInter.append(pointData.GetTuple3(idx))
    cellIdsInter.append(cellIds.GetId(idx))

  return pointsInter, cellIdsInter

## From https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/#more-91
def loadSTL(filenameSTL):
    readerSTL = vtk.vtkSTLReader()
    readerSTL.SetFileName(filenameSTL)
    # 'update' the reader i.e. read the .stl file
    readerSTL.Update()

    polydata = readerSTL.GetOutput()

    # If there are no points in 'vtkPolyData' something went wrong
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError(
            "No point data could be loaded from '" + filenameSTL)
        return None

    return polydata
##

# Choose some starting points, for example the positions of the cameras
import pandas as pd

pose = pd.read_csv('C:/Users/haavasl/PycharmProjects/Testing/venv/Code/nav_file_for_tautra_new.txt')

X = pose['X'].values.reshape((-1,1))
Y = pose['Y'].values.reshape((-1,1))
Z = pose['Z'].values.reshape((-1,1))

xyz_start = np.zeros((X.shape[0], 3), dtype=np.float64)
xyz_end = np.zeros((X.shape[0], 3), dtype=np.float64)

offX = 575200
offY = 7052700
offZ = -90

xyz_start[:,0] = X[:,0] - offX
xyz_end[:,0] = X[:,0] - offX

xyz_start[:,1] = Y[:,0] - offY
xyz_end[:,1] = Y[:,0] - offY

xyz_start[:,2] = Z[:,0] - offZ
xyz_end[:,2] = Z[:,0] - 5 - offZ



pos_arr = np.zeros((X.shape[0], 3))

mesh = loadSTL('transectShape.stl')

obbTree = vtk.vtkOBBTree()
obbTree.SetDataSet(mesh)
obbTree.BuildLocator()


import time
t1 = time.time()
for i in range(X.shape[0]):
    points = vtk.vtkPoints()
    #cellIds = vtk.vtkIdList()
    code = obbTree.IntersectWithLine((xyz_start[i,0], xyz_start[i,1], xyz_start[i,2]),
                                 (xyz_end[i,0], xyz_end[i,1], xyz_end[i,2]), points, None)
    pointsVTKIntersectionData = points.GetData()

    noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

    if noPointsVTKIntersection > 0:
        _tup = pointsVTKIntersectionData.GetTuple3(0)
        pos_arr[i,:] = np.array([_tup])



t2 = time.time()
#
tot = t2-t1
print(points)