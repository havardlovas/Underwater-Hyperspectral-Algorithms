import gdal
import osr
import rasterio
from rasterio import Affine, crs
import h5py
import pandas as pd
import datetime
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

class Img:
    def __init__(self, filename, is_radiance=True):
        self.f = h5py.File(filename, 'r', libver='latest')
        self.datacube = self.f['rawdata/hyperspectral/dataCube'].value
        #self.t_exp = self.f['uhi/parameters'].value[0, 0] / 1000
        #self.radiometricFrame = np.load('radiometricFrame.npy') / 1000
        #self.darkFrame = np.load('darkFrame.npy')
        #self.datacube_corrected = np.zeros(self.datacube.shape)
        #self.band2wlen = self.f['uhi/calib'].value
        #self.dataCubeTimeStamps = self.f['uhi/parameters'].value[:, 2]
        #self.imgTimeStamps = self.f['rgb/parameters'].value[:, 2]
        self.geometric_calib = self.f['rawdata/hyperspectral/calibration/geometric/fieldOfView'].value
        self.imgs = self.f['rawdata/rgb/rgbFrames'].value

filename = 'C:/Users/haavasl/calibFiles/calib_exp30ms_wack_20200923_122655_1.h5'
image_dir_write= 'C:/Users/haavasl/calibFiles/'
img = Img(filename)
n = img.imgs.shape[0]

#plt.plot(np.arange(960), np.tan(img.geometric_calib*np.pi/180))

#df = pd.DataFrame(np.tan(img.geometric_calib*np.pi/180))

#df.to_csv('spectral_rays.csv')
#plt.show()
grey_scale = np.mean(img.datacube, axis = 2)
#grey_scale = gaussian_filter(grey_scale, (2,2))
# Correct for lateral effects
max_tot = np.max(grey_scale)
max_lat = np.max(grey_scale, axis = 0)

#grey_scale *= max_tot/max_lat
grey_scale *= (grey_scale-np.min(grey_scale))/(np.max(grey_scale) - np.min(grey_scale))


matplotlib.pyplot.plot(np.arange(grey_scale[1000,:].shape[0]), grey_scale[400,:])
matplotlib.pyplot.show()

matplotlib.image.imsave(image_dir_write + 'img3.png', grey_scale, cmap = 'gray')
matplotlib.pyplot.imshow(grey_scale, cmap = 'gray')
matplotlib.pyplot.show()
#wavelens = img.band2wlen

#for i in range(n):
#
#    img_x = np.flip(np.asarray(img.imgs[i])[:, :, 0:3], axis=2)
#    if i%2 == 0:
#        matplotlib.image.imsave(image_dir_write + #"'img_varied_att' + str(i) + '.png', img_x)