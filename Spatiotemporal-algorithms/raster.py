import sys
import os

from osgeo import gdal, gdalconst
from osgeo.gdalconst import *
import rasterio
import numpy as np


fn = 'geo_uhi_20200111_103423_uhi.tif'
#data, geodata=load_data(fn, gdal_driver='GTiff')

#import matplotlib.pyplot as plt
#plt.imshow



driver = gdal.GetDriverByName('GTiff')
driver.Register()

ds = gdal.Open(fn, GA_ReadOnly)

# Print basic raster info
print(ds.RasterXSize, ds.RasterYSize)

print(ds.GetProjection())

print(ds.GetGeoTransform())

print('Number of Raster Bands ', ds.RasterCount)

# Print raster band info
band1 = ds.GetRasterBand(1)
print('No data value ',band1.GetNoDataValue())
print('MinVal Band 1 ', band1.GetMinimum())
print('MaxVal Band 1 ', band1.GetMaximum())
print('Data type ', band1.GetUnitType())

# Work with bands as arrays
band2 = ds.GetRasterBand(1).ReadAsArray()
band3 = ds.GetRasterBand(2).ReadAsArray()
band4 = ds.GetRasterBand(3).ReadAsArray()

print(band2.shape)

