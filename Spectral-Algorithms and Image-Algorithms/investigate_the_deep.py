import h5py
import matplotlib.pyplot as plt
import spectral as sp
import numpy as np
from scipy import ndimage
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

def attenuate_spectra(spectra, attenuation_spectrum, alt, field_of_view, roll_ang):
    # Spectra.shape = inf,968,208 attenuation_spectrum.shape = 1,208, alt.shape = -1, and field_of_view.shape = 968
    # Calculate the distance based on geometry
    OOI_angle = np.multiply(np.ones(alt.shape).reshape((-1,1)),field_of_view.reshape((1,-1))) + roll_ang.reshape((-1,1))
    optical_distances = np.multiply(alt.reshape((-1,1)), 1/np.cos(OOI_angle*np.pi/180)) * 2
    optical_ln_attenuation = np.multiply(optical_distances.reshape((optical_distances.shape[0], optical_distances.shape[1],1
                                                                 )), attenuation_spectrum.reshape((1,1, attenuation_spectrum.shape[0])))

    optical_spread = np.power(optical_distances.reshape((optical_distances.shape[0], optical_distances.shape[1],1
                                                                 )), 2)

    # The spread

    return spectra * np.exp(optical_ln_attenuation)

def normalize_spectra(spectra, wave_idx):
    # spectra.shape = inf,968,208, normalizing_arr.shape = inf, 968
    normalizing_arr = spectra[:,:,wave_idx].reshape((spectra.shape[0],spectra.shape[1],1))
    spectra = np.multiply(spectra, 1/normalizing_arr)
    return spectra
def image_stitcher(rgb_frames, pixels_per_frame):
    photomosaic = np.zeros((int(rgb_frames.shape[0]*pixels_per_frame),rgb_frames.shape[2],3))
    for i in range(rgb_frames.shape[0]):
        miniframe = rgb_frames[i, 0:pixels_per_frame]
        photomosaic[8*i:8*i + 8] = miniframe
    return photomosaic
def interpolate_to(x_old,y_old, x_new):
    a = 0
    b = 0
    index_start = 0
    index_stop = -1
    f_interp = interp1d(x_old,y_old)
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
    y_new[index_stop:] = y_old[-1]
    return y_new


def vegetation_idx(spectra,start_idx, end_idx):
    veg_col = np.multiply(spectra[:,end_idx], 1/spectra[:,end_idx - start_idx])
    return veg_col


def filter_alt(altitude):
    for i in range(altitude.shape[0]):
        if altitude[i] > 2.5:
            altitude[i] = altitude[i-1]
    return altitude


file_base = 'C:/Users/haavasl/NyAlesundUHI/Svalbard_2020/Newdata/'



radiance_base = 'rad_uhi_20200111_134622_rad'

file_name1 = file_base + radiance_base + ".h5"

f1 = h5py.File(file_name1, 'r',libver='latest')


## Extract all datacube data
dataCube1 = f1['processed/radiance/dataCube']
dataCubeTimeStamps1 = f1['processed/radiance/timestamp']
##

total_timestamps = dataCubeTimeStamps1.value

## Extract altitude data and timestamps
altitude = f1['rawdata/navigation/altitude/Altitude']
altitudeTimeStamp = f1['rawdata/navigation/altitude/TimestampMeasured']
# Need to smooth this data
smoothed_altitude = filter_alt(altitude.value)


file_base = "C:/Users/haavasl/PycharmProjects/GeoLoc/venv/"

light_spectra = pd.read_excel(file_base + 'light_source.xlsx', header=None)

wavelens = pd.read_csv(file_base + 'wavelens.csv', header = None)
wavelens = np.array(wavelens.values)
wavelens = wavelens.reshape((1,-1))

# Extract geometric shape of UHI fan to use for altitude correction:
field_of_view = f1['processed/radiance/calibration/geometric/fieldOfView'].value
# Extract Roll angle to use for altitude correction

roll1 = f1['/rawdata/navigation/imu/Roll']
time_stamp_roll1 = f1['/rawdata/navigation/imu/TimestampMeasured']

roll_tot = roll1.value
time_roll_tot = time_stamp_roll1.value






lib_spectra = pd.read_excel(file_base + 'attenuation_David.xlsx', header=None)
lib_spectra2 = pd.read_excel(file_base + 'attenuation_David.xlsx', header=None, sheet_name = 1)
att_arr = np.array(lib_spectra.values)
att_arr2 = np.array(lib_spectra2.values)
f = interp1d(att_arr2[:,0],att_arr2[:,1])
att_comp = f(att_arr[:,0])
att_water = att_arr[:,3]
att_tot = att_comp + att_water

f2 = interp1d(att_arr[:,0], att_tot)
attenuation_spectra = f2(wavelens[0])

# Interpolation of light and attenuation
f = interp1d(light_spectra.values[:,0], light_spectra.values[:,1])

light_spectrum = f(wavelens)

light_spectrum_UHI = np.load('light_source_UHI_measurement.npy', allow_pickle=True)




# Interpolate altitude
f_alt = interp1d(altitudeTimeStamp.value, smoothed_altitude)
altitude_total = interpolate_to(altitudeTimeStamp.value,smoothed_altitude,total_timestamps)
# Interpolate Roll
f_roll = interp1d(time_roll_tot, roll_tot)
roll_total = interpolate_to(time_roll_tot, roll_tot, total_timestamps)


compression = 8
img = dataCube1.value[:,0:968,:]
start_frame = 1000
#end_frame = 2300
end_frame = -1
new_img = img[start_frame:2100,0:968,:]
new_alt = altitude_total[start_frame:2100]
new_roll = roll_total[start_frame:end_frame:2100]



new_img_reshaped = new_img.reshape((new_img.shape[0]*968, -1))

# Average spectrum for the plaque
ref_plaq = np.load('ref_plaque.npy', allow_pickle=True)
ref_plaq_med = np.mean(ref_plaq.reshape((ref_plaq.shape[0]*ref_plaq.shape[1],ref_plaq.shape[2])), axis = 0)
a = np.load('beam_attenuation_slope.npy', allow_pickle=True)
b = np.load('beam_attenuation_bias.npy', allow_pickle=True)

# Apply division on light source -> Apparent reflectance
new_img_reshaped /= light_spectrum

#max_val = np.amax(new_img_reshaped[:,10:181], axis = 1).reshape((-1,1))
#min_val = np.amin(new_img_reshaped[:,10:181], axis = 1)
#new_img_reshaped /= max_val

new_img_reshaped = new_img.reshape((new_img.shape[0],968, -1))
#sp.imshow(np.log(new_img_reshaped[:,::8,:] + 1), [180,81,50])
#n = new_img.shape[0]*compression
#m = new_img.shape[1]
#k = new_img.shape[2]
#
#new_dim_img = np.zeros((n, m, k))
##
#for i in range(compression):
#    new_dim_img[i::8,0:968,:] = np.concatenate((np.zeros((1,968,208)), new_img_reshaped[1:,:,:]),axis=0)*(i/7) + \
#                                np.concatenate((new_img_reshaped[0:-1,:,:],np.zeros((1,968,208))),axis=0)*((7-i)/7)
#new_img_reshaped *= np.exp(b)

# Investigate the depth some spectra at different depths:

#for i in range(121):
#    if i > 0:
#        plt.plot(wavelens[0], ndimage.gaussian_filter1d(new_img_reshaped[250,i*8,:], sigma=2))
#    else:
#        plt.plot(wavelens[0], ndimage.gaussian_filter1d(new_img_reshaped[250,i*8,:], sigma=2), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[250], decimals=2)) + ' m')

#plt.plot(wavelens[0], np.mean(ndimage.gaussian_filter1d(new_img_reshaped[10,470:498,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[10], decimals=2)) + ' m')
##plt.plot(wavelens[0], np.mean(ndimage.gaussian_filter1d(new_img_reshaped[50,470:498,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[50], decimals=2)) + ' m')
#plt.plot(wavelens[0], np.mean(ndimage.gaussian_filter1d(new_img_reshaped[765,470:498,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[765], decimals=2)) + ' m')
#plt.plot(wavelens[0], np.mean(ndimage.gaussian_filter1d(new_img_reshaped[1000,470:498,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[1000], decimals=2)) + ' m')
##plt.plot(wavelens[0], np.median(ndimage.gaussian_filter1d(new_img_reshaped[1200,:,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[1200], decimals=2)) + ' m')
#plt.plot(wavelens[0], np.mean(ndimage.gaussian_filter1d(new_img_reshaped[1400,470:498,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[1400], decimals=2)) + ' m')
#plt.plot(wavelens[0], np.mean(ndimage.gaussian_filter1d(new_img_reshaped[1966,470:498,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[1966], decimals=2)) + ' m')
#plt.plot(wavelens[0], np.median(ndimage.gaussian_filter1d(new_img_reshaped[2050,:,:], sigma=7), axis=0), label = '$\\rho_{w}$ at altitude ' + str(np.round(altitude_total[2050], decimals=2)) + ' m')

#for i in range(new_img_reshaped.shape[0]):
#    plt.plot(wavelens[0], np.median(ndimage.gaussian_filter1d(new_img_reshaped[i, 470:498, :], sigma=7), axis=0))
#plt.legend()
#plt.grid()
#plt.show()

#wavelenx = 82
#waveleny = 160
#norm_wlen = 110
#x = np.log10(new_img_reshaped[:,:,wavelenx] + 0.00001).reshape(-1) + 2
#x_norm = np.log10(new_img_reshaped[:,:,norm_wlen] + 0.00001).reshape(-1) + 2
#y = np.log10(new_img_reshaped[:,:,waveleny] + 0.00001).reshape(-1) + 2
#y_norm = np.log10(new_img_reshaped[:,:,norm_wlen] + 0.00001).reshape(-1) + 2
#plt.hist2d(x-x_norm, y-y_norm, bins = (100,100), range = ((-1,1),(-1,1)))
#plt.ylabel('Log-Reflectance at ' + str(np.round(wavelens[0, waveleny])) + ' nm')
#plt.xlabel('Log-Reflectance at ' + str(np.round(wavelens[0, wavelenx])) + ' nm')
#plt.show()


# Apply Calibration Correction Bias
# new_img_reshaped *= np.exp(b)
# Replace attenuation spectra
attenuation_spectra = a

new_img_reshaped = new_img_reshaped.reshape((new_img.shape[0],968,-1))
# Allocate memory for backscatter function
backscatter_arr = np.zeros(wavelens.shape[1])
# Look iterate through all wavelengths:

backscatter_arr = np.median(np.mean(new_img_reshaped[952:968,82*8:98*8,:], axis = 1), axis= 0)

time = np.arange(new_img_reshaped.shape[0])/25
backscatter_arr = ndimage.gaussian_filter1d(backscatter_arr, sigma=3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(wavelens[0, 10:181].reshape((-1,1)), backscatter_arr[10:181].reshape((-1,1)))
a = regressor.coef_
b = regressor.intercept_

bs_params = np.zeros((2,1))
bs_params[0] = a
bs_params[1] = b
np.save('backscatter_parameters', bs_params)

np.save('backscatter_array', backscatter_arr)
r = np.arange(300) + 400
y = r*a + b

#plt.plot(time, ndimage.gaussian_filter1d(np.mean(new_img_reshaped[:,0:8,110]/new_img_reshaped[:,960:968,110], axis = 1), sigma = 10), label = 'Remote sensing reflectance [sr$^{-1}$] at 700 nm in middle')
plt.plot(wavelens[0, 10:181], backscatter_arr[10:181], label = 'Apparent backscatter [sr$^{-1}$]')
plt.plot(r, y[0], label = 'Linear trend fit [sr$^{-1}$]')
#plt.plot(time, ndimage.gaussian_filter1d(filter_alt(new_alt), sigma = 3*25), label = 'Altitude [m]')
#plt.scatter(filter_alt(new_alt), np.mean(new_img_reshaped[:,960:968,190], axis = 1), label = 'Altitude [m]')
plt.xlabel('Wavelength [nm]')
plt.ylim(([0,0.01]))
plt.xlim(([400, 700]))
plt.legend()
plt.grid()
plt.show()
# Estimate through using water attenuation
#new_img_reshaped_with_attenuation = attenuate_spectra(new_img_reshaped, attenuation_spectra, new_alt, field_of_view, new_roll)

sp.imshow(new_img_reshaped[:,::8,:], [180,81,50])
print()

