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
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

class Hyperspectral:
    def __init__(self, filename, is_radiance=True):
        self.f = h5py.File(filename, 'r', libver='latest')
        #self.datacube = self.f['uhi/pixels'].value[:, 0:866]
        #self.t_exp = self.f['uhi/parameters'].value[0, 0] / 1000
        #self.radiometricFrame = np.load('radiometricFrame.npy') / 1000
        #self.darkFrame = np.load('darkFrame.npy')
        #self.datacube_corrected = np.zeros(self.datacube.shape)
        #self.band2wlen = self.f['uhi/calib'].value
        #self.dataCubeTimeStamps = self.f['uhi/parameters'].value[:, 2]
        #self.imgTimeStamps = self.f['rgb/parameters'].value[:, 2]
        self.imgs = self.f['rawdata/rgb/rgbFrames'].value
        #self.n_imgs = self.imgTimeStamps.shape[0]
    def reflectance_conversion(self, dist):
        # Radiance conversion
        for i in range(self.datacube.shape[0]):
                self.datacube_corrected[i, :, :] = (self.datacube[i, :, :] - self.darkFrame[0:866]) / (
                            self.radiometricFrame[0:866] * self.t_exp)

        # Apparent reflectance
        #light_spectra = pd.read_excel('Multi_SeaLite_spectrum.xlsx')
        #wavelengths = light_spectra.values[:,0]



fn = "C:/Users/haavasl/Downloads/USV_20180322_125319_3.h5"

hyp = Hyperspectral(fn)

plt.imshow(np.mean(hyp.imgs[:,:,:,:], axis = 0)/ np.max(np.mean(hyp.imgs[:,:,:,:], axis = 0)))
plt.show()