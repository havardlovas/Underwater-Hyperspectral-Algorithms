
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

def Rot_BODY2NED(x, y, z, roll, pitch, yaw):
    phi = roll
    theta = pitch
    psi = yaw
    # For making it readable:
    Cphi = np.cos(phi)
    Sphi = np.sin(phi)
    Ctheta = np.cos(theta)
    Stheta = np.sin(theta)
    Cpsi = np.cos(psi)
    Spsi = np.sin(psi)


    R = np.array([[Cpsi*Ctheta,-Spsi*Cphi + Cpsi*Stheta*Sphi,Spsi*Sphi + Cpsi*Cphi*Stheta],
                  [Spsi*Ctheta, Cpsi*Cphi + Sphi*Stheta*Spsi, -Cpsi*Sphi + Stheta*Spsi*Cphi],
                  [-Stheta,Ctheta*Sphi,Ctheta*Cphi]])
    p_BODY = np.array([x, y, z])
    p_NED = np.matmul(R, p_BODY)

    return p_NED

def interpolate_to(x_old,y_old, x_new):
    a = 0
    b = 0
    index_start = 0
    index_stop = x_new.shape[0]
    f_interp = interp1d(x_old, y_old)
    y_new = np.ones(x_new.shape[0]) * y_old[0]
    for i in range(x_new.shape[0]):
        if (x_new[i] > x_old[0] and a == 0):
            index_start = i
            a = 1
        elif (x_new[i] > x_old[-1] and b == 0):
            index_stop = i
            b = 1

        if (x_new[i] > x_old[0] and a == 0):
            index_start = i
            a = 1
        elif (x_new[i] > x_old[-1] and b == 0):
            index_stop = i
            b = 1

    y_new[index_start:index_stop] = f_interp(x_new[index_start:index_stop])
    y_new[index_stop:] = y_old[len(y_old)-1]
    return y_new

def convertTimeToEpoch(date, time):
    date_str = date.split('.')
    time_str = time.split(':')
    # Convert to a date time
    time_splitted = date_str + time_str
    time_dt = datetime.datetime(int(time_splitted[2]), int(time_splitted[1]), int(time_splitted[0]),
                                int(time_splitted[3]), int(time_splitted[4]), int(np.floor(float(time_splitted[5]))),
                                int((float(time_splitted[5]) - np.floor(float(time_splitted[5]))) * 1000000))

    # Convert to epoch time
    time_epoch = datetime.datetime.timestamp(time_dt)
    return time_epoch

def get_timestamps(date, timestamps):
    numeric_timestamps = np.zeros(len(date))
    for i in range(len(date)):
        numeric_timestamps[i] = convertTimeToEpoch(date[i], timestamps[i])
    return numeric_timestamps

def append_wavelength_to_header(filename, wavelengths):
    # Function for appending necessary fields for aan ENVI header
    file_obj = open(filename, 'a+')
    n_wavelens = len(wavelengths)
    # Write 6 entries per line
    count = 1
    file_obj.write('wavelength = {\n')
    for i in range(n_wavelens):
        if i < n_wavelens-1:
            file_obj.write(str(wavelengths[i]) + ', ')
            if count  == 6:
                file_obj.write(str(wavelengths[i]) + ',\n')
        else:
            file_obj.write(str(wavelengths[i]) + '}\n')


    # Add unit for wavelength to make readable
    file_obj.write('wavelength units = Nanometers')
    
    file_obj.close()






class Navigation:
    def __init__(self, nav_file_name):
        col_names = ['Date','Timestamp','Easting','Northing','Roll','Pitch','Yaw', 'Depth', 'Altitude']
        self.file = pd.read_csv(nav_file_name, sep  = '\t',header = None, names = col_names)
        self.date = self.file['Date'].values
        self.timestamp = self.file['Timestamp'].values
        self.easting = self.file['Easting'].values
        self.northing = self.file['Northing'].values
        self.roll = self.file['Roll'].values
        self.pitch= self.file['Pitch'].values
        self.yaw = self.file['Yaw'].values
        self.depth = self.file['Depth'].values
        self.altitude = self.file['Altitude'].values
        self.timestamp_zero = convertTimeToEpoch(self.date[0], self.timestamp[0])
        self.timestamp_numerical = get_timestamps(self.date, self.timestamp)
        self.altitude = self.smooth(self.altitude, 3)
    def smooth(self, array, sigma):
        return gaussian_filter1d(array, sigma)
    def interpolate_to_scan_times(self, dataCubeTime):
        northing = self.northing
        easting = self.easting
        pitch = self.pitch
        roll = self.roll
        yaw = self.yaw

        altitude = self.altitude
        t = self.timestamp_numerical + 3600 # Compensate for time zone
        self.t_new = dataCubeTime

        self.northing_cam = interpolate_to(t, northing, self.t_new)
        self.easting_cam = interpolate_to(t, easting, self.t_new)
        self.pitch_cam = interpolate_to(t, pitch, self.t_new)
        self.roll_cam = interpolate_to(t, roll, self.t_new)
        self.yaw_cam = interpolate_to(t, yaw, self.t_new)
        self.altitude_cam = interpolate_to(t, altitude, self.t_new)
        self.n_scan_lines = self.t_new.shape[0]
        print('')


    def transform_coordinates_to_ned(self, field_of_view):
        self.n_fov = field_of_view.shape[0]
        # First, based on flat seabed assumption, the true altitude can be estimated from the altitude
        self.true_altitude = np.multiply(self.altitude_cam, np.cos(self.roll_cam*np.pi/180))
        # Then we iterate through the camera positions (captures) and pixels (look-angles)
        n_pos, n_angles = self.northing_cam.shape[0], field_of_view.shape[0]
        self.north_Non_Gridded = np.zeros((n_pos, n_angles))
        self.east_Non_Gridded = np.zeros((n_pos, n_angles))
        #self.z_NED_Non_Gridded = np.zeros((n_pos, n_angles))
        unit_pos_body = np.array([np.zeros(field_of_view.shape), np.sin(field_of_view),
                             np.cos(field_of_view)])
        for j in range(n_pos):
                # The unit_pos_body is multiplied by som radial distance and the z component should then be true alt
                rel_pos_NED_unit = Rot_BODY2NED(unit_pos_body[0, :], unit_pos_body[1, :], unit_pos_body[2, :], self.roll_cam[j] * np.pi / 180,
                               self.pitch_cam[j] * np.pi / 180,
                               self.yaw_cam[j] * np.pi / 180)
                # Scaling is estimated by setting z components equal to "true altitude" 
                scaling = np.multiply(self.true_altitude[j], 1/rel_pos_NED_unit[2,:])
                # Calculate the north-east positions
                self.north_Non_Gridded[j, :] = np.multiply(scaling, rel_pos_NED_unit[0,:]) + self.northing_cam[j]
                self.east_Non_Gridded[j, :] = np.multiply(scaling, rel_pos_NED_unit[1, :]) + self.easting_cam[j]
    def grid_to_raster_format(self, resolution = 0.01):
        self.resolution = resolution
        # First thing is to establish the highest y value and lowest y value and x value
        self.x_lim = [np.min(self.north_Non_Gridded), np.max(self.north_Non_Gridded)]
        self.y_lim = [np.min(self.east_Non_Gridded), np.max(self.east_Non_Gridded)]


        x = np.arange(self.x_lim[0], self.x_lim[1], resolution)
        y = np.arange(self.y_lim[0], self.y_lim[1], resolution)
        Y, X = np.meshgrid(y, x)

        z = np.linspace(1, 1 + self.n_scan_lines * self.n_fov, self.n_scan_lines * self.n_fov)
        #z = datacube[:, :, 100].reshape((-1, 1))
        x = self.north_Non_Gridded.reshape((-1,1))
        y = self.east_Non_Gridded.reshape((-1,1))
        x = x[:,0]
        y = y[:, 0]
        #z = z[:, 0]

        samples = self.north_Non_Gridded.shape[1]
        lines = self.north_Non_Gridded.shape[0]
        z[:samples] = 0
        z[-samples:] = 0
        z[0:samples * lines:samples] = 0
        z[samples - 1:samples * lines:samples] = 0
        self.map = np.zeros([X.shape[0], X.shape[1]])
        self.griddedband = griddata((x, y), z, (X, Y), method='nearest')

        self.map[:, :][np.nonzero(self.griddedband)] = self.griddedband[
            np.nonzero(self.griddedband)]
        self.map = self.map.astype('int32')

    def iterative_referencing(self, datacube, resolution = 0.02):

        self.resolution = resolution
        # First thing is to establish the highest y value and lowest y value and x value
        self.x_lim = [np.floor(np.min(self.north_Non_Gridded)*100), np.ceil(np.max(self.north_Non_Gridded)*100)]
        self.y_lim = [np.floor(np.min(self.east_Non_Gridded)*100), np.ceil(np.max(self.east_Non_Gridded)*100)]
        # Iterate through scan line by scan line and georeference full datacube at once through average
        # First we define a count matrix counting number of measurements per cell
        x = np.arange(self.x_lim[0], self.x_lim[1], resolution)
        y = np.arange(self.y_lim[0], self.y_lim[1], resolution)

        Y, X = np.meshgrid(y, x)

        hyperspec_geocorr = np.zeros((X.shape[0], X.shape[1], datacube.shape[2]))
        hyperspec_count = np.zeros(X.shape)
        # Hundred scan lines
        for i in range(100):
            for j in range(968):
                # Round to closest
                print('Success')


    def set_geo_transform(self):
        self.x_min = self.x_lim[0]
        self.x_max = self.x_lim[1]
        self.y_min = self.y_lim[0]
        self.y_max = self.y_lim[1]


        self.geotransform = Affine.translation(self.y_min, self.x_min) * Affine.scale(self.resolution, self.resolution)




class Hyperspectral:
    def __init__(self, filename, is_radiance = True):

        self.f = h5py.File(filename, 'r', libver='latest')
        self.datacube = self.f['processed/radiance/dataCube'].value
        self.dataCubeTimeStamps = self.f['processed/radiance/timestamp'].value
        self.fov = self.f['processed/radiance/calibration/geometric/fieldOfView'].value * np.pi / 180
        self.band2wlen = self.f['processed/radiance/calibration/spectral/band2Wavelength'].value

# Goal is to create small raster with pixel size 1 cm with say 50 bands
is_gdal = 1

file_base = 'C:/Users/haavasl/NyAlesundUHI/Svalbard_2020/Newdata/'

radiance_base = 'rad_uhi_20200111_103423_1_1_'


# Start by establishing the full grid:
file_name_hyperspectral = file_base + radiance_base

fn1 = file_name_hyperspectral + '1' + ".h5"

hyp1 = Hyperspectral(fn1)
hyp2 = Hyperspectral(file_name_hyperspectral + '2' + ".h5")
hyp3 = Hyperspectral(file_name_hyperspectral + '3' + ".h5")

hyp_arr = [hyp1,hyp2,hyp3]

time_stamps_total = np.concatenate((hyp1.dataCubeTimeStamps,hyp2.dataCubeTimeStamps,hyp3.dataCubeTimeStamps))

n_arr = np.array([0,hyp1.dataCubeTimeStamps.shape[0], hyp2.dataCubeTimeStamps.shape[0], hyp3.dataCubeTimeStamps.shape[0]])*968
n_arr = np.cumsum(n_arr)
# The navigation file corresponds to
nav = Navigation('NavLong2.nav')

# The Hyperspectral data file


nav.interpolate_to_scan_times(time_stamps_total)

# Next step is to place the data geometrically
# First to estimate the altitude the pitch is all that is needed
nav.transform_coordinates_to_ned(hyp1.fov)
# Create the grid for
nav.grid_to_raster_format(resolution=0.015)
nav.set_geo_transform()

col_count = 0
nrows_tot, ncols_tot = nav.map.shape

n_max_spectra = 250000
# The number of chunks can be decided through division. 250000 spectra at a time
n_chunks = int(np.ceil((nrows_tot*ncols_tot)/(n_max_spectra)))

for j in range(n_chunks):
    ncols = int(np.min((ncols_tot-col_count, np.ceil(n_max_spectra/nrows_tot))))
    nrows = nrows_tot
    # Current map
    curr_map = nav.map[:, col_count : ncols+col_count]
    # Find the maximal index of the chunk and minimal
    max_index = np.max(curr_map)
    min_index = np.min(curr_map)
    file_name = 'svalbard_rasters' + str(j + 1)
    # Identify which file-segments they include
    a = 0
    for i in range(len(n_arr)-1):
        if (n_arr[i] <= min_index and n_arr[i+1] > min_index) or a == 1:

            dcube = hyp_arr[i].datacube
            if a == 0:
                dcube_tot = dcube
                min_index_gim = n_arr[i]
                a = 1
            else:
                dcube_tot = np.concatenate((dcube_tot,dcube),axis=0)
            # If then the next index is larger than max we break
            if n_arr[i + 1] > max_index:
                a = 0




    bands = hyp_arr[0].band2wlen
    gim = np.zeros((nrows_tot, ncols), dtype = 'float32')



    light_spectra = pd.read_excel('C:/Users/haavasl/PycharmProjects/GeoLoc/venv/light_source.xlsx', header=None)



    # Interpolation of light and attenuation
    f = interp1d(light_spectra.values[:,0], light_spectra.values[:,1])

    light_spectrum = f(bands)

    dcube_tot /= light_spectrum

    if is_gdal == 0:
        new_dataset = rasterio.open(
            'new_raster_tiff_3.tif',
            'w',
            driver='GTiff',
            height=gim.shape[0],
            width=gim.shape[1],
            count=3,
            dtype=gim.dtype,
            crs=crs.CRS.from_epsg(32632),
            transform=nav.geotransform,
            nodata=0,
        )

        for i in range(3):
            im = dcube_tot[:,:,bands[i]].reshape(-1)
            gim = im[nav.map.astype('int32').reshape(-1) - 1]
            gim[nav.map.astype('int32').reshape(-1) == 0] = 0
            gim = gim.reshape(136, 2446)

            new_dataset.write(gim, i+1)

        new_dataset.close()

    elif is_gdal == 1:
        driver = gdal.GetDriverByName('ENVI')
        new_dataset = driver.Create(file_name + '.bip', ncols, nrows, len(bands), gdal.GDT_Float32, options=['INTERLEAVE=BIP'])
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromEPSG(32632)
        sr_wkt = spatialRef.ExportToWkt()
        new_dataset.SetProjection(sr_wkt)
        #7036650.77,
        new_dataset.SetGeoTransform([
            569021.00 + col_count*nav.resolution,
            nav.resolution,
            0,
            7036650.77,
            0,
            -nav.resolution,
        ])

        for i in range(len(bands)):

            im = dcube_tot[:, :, i].reshape(-1)
            gim = im[curr_map.reshape(-1) - 1 - min_index_gim]
            gim[curr_map.astype('int32').reshape(-1) == 0] = 0
            gim = gim.reshape(nrows, ncols)

            rasterband = new_dataset.GetRasterBand(i+1)
            rasterband.WriteArray(gim)
            print(100*i/len(bands))
            del rasterband

    del new_dataset

    append_wavelength_to_header(file_name + '.hdr', bands)


    col_count += ncols

print('success')

