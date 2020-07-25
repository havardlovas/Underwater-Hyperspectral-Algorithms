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
def filter_alt(altitude):
    for i in range(altitude.shape[0]):
        if altitude[i] > 3:
            altitude[i] = altitude[i-1]
    return altitude






new_img_reshaped = new_img.reshape((new_img.shape[0]*968, -1))

# Average spectrum for the plaque
ref_plaq = np.load('ref_plaque.npy', allow_pickle=True)
ref_plaq_med = np.mean(ref_plaq.reshape((ref_plaq.shape[0]*ref_plaq.shape[1],ref_plaq.shape[2])), axis = 0)
a = np.load('beam_attenuation_slope.npy', allow_pickle=True)
b = np.load('beam_attenuation_bias.npy', allow_pickle=True)

#light_spectrum = ref_plaq_med
# Apply division on light source -> Apparent reflectance
new_img_reshaped /= light_spectrum

new_img_reshaped = new_img_reshaped.reshape((new_img.shape[0],968,-1))

n = new_img.shape[0]*compression
m = new_img.shape[1]
k = new_img.shape[2]
#
new_dim_img = np.zeros((n, m, k))
#
for i in range(compression):
    new_dim_img[i::8,0:968,:] = new_img_reshaped

sp.imshow(new_dim_img,[150,80,50])
print()
## Apply Calibration Correction Bias
##new_img_reshaped *= np.exp(b)
## Replace attenuation spectra
#attenuation_spectra = a
#new_img_reshaped = new_img_reshaped.reshape((new_img.shape[0],968,-1))
#wavelens = wavelens[0,:]
#
#bs_params = np.load('backscatter_parameters.npy', allow_pickle=True)
#bs_arr_real = np.load('backscatter_array.npy', allow_pickle=True)
##
#bs_array = wavelens[10:190]*bs_params[0] + bs_params[1]
#
#testimg = np.zeros((15,15,bs_array.shape[0]))
#for i in range(15):
#    for j in range(15):
#        idx_j = 8*j + 80
#        idx_i = 2500 + 310 + i
#        y = ndimage.gaussian_filter1d(np.mean(new_img_reshaped[idx_i, idx_j:idx_j + 8, 10:190], axis = 0), sigma = 1) - bs_arr_real[10:190]
#        y_norm = y/y[110-10]
#        #plt.plot(wavelens[10:190], y)
#        testimg[i,j,:] = y_norm
#
##sp.imshow(testimg,[170,170,170])
##for i in range(10):
##    for j in range(15):
##        idx_j = 8*j + 80
##        idx_i = 640 + i
##        y = ndimage.gaussian_filter1d(np.mean(new_img_reshaped[idx_i, idx_j:idx_j + 8, 0:190], axis = 0), sigma = 3)
##        y_norm = y/y[110]
##        plt.plot(wavelens[0:190], y_norm, 'b')
##
##for i in range(10):
##    for j in range(15):
##        idx_j = 8*j + 80
##        idx_i = 394 + i
##        y = ndimage.gaussian_filter1d(np.mean(new_img_reshaped[idx_i, idx_j:idx_j + 8, 0:190], axis = 0), sigma = 3)
##        y_norm = y/y[110]
##        plt.plot(wavelens[0:190], y_norm, 'g')
#lib_spectra = pd.read_excel('C:/Users/haavasl/PycharmProjects/GeoLoc/venv/Lib_spectra_All.xlsx')
### 66:468
#lib_spectra_arr = np.array(lib_spectra.values[66:468,:])
#spectra = lib_spectra_arr[:,1:51].transpose()
##
##plt.plot(lib_spectra_arr[:,0], spectra[10,:]/spectra[10,223], label = 'Old 1')
##plt.plot(lib_spectra_arr[:,0], spectra[11,:]/spectra[11,223], label = 'New 1')
##plt.plot(lib_spectra_arr[:,0], spectra[12,:]/spectra[12,223], label = 'Old 2')
##plt.plot(lib_spectra_arr[:,0], spectra[13,:]/spectra[13,223], label = 'New 2')
##plt.plot(lib_spectra_arr[:,0], spectra[14,:]/spectra[14,223], label = 'Old 3')
##plt.plot(lib_spectra_arr[:,0], spectra[15,:]/spectra[15,223], label = 'New 3')
#from sklearn.metrics.pairwise import cosine_similarity
#
#from sklearn.linear_model import LinearRegression
#
#
#
#
#color_arr = ['r', 'g', 'b', 'c', 'm', 'k']
#label_arr = ['Old 1', 'New 1', 'Old 2', 'New 2', 'Old 3 ', 'New 3']
#delta_start_idx = 0
#delta_end_idx = 1
## Approach 1: Linear regression
#
#a = a[delta_start_idx+10:190-delta_end_idx].reshape((1,-1))
#regressor = LinearRegression()
#
#binner = 1
#k = 4
#y_true = spectra[10+k, :]
#y_true = interpolate_to(lib_spectra_arr[:, 0], y_true, wavelens)
#y_true = y_true[delta_start_idx+10:190-delta_end_idx]
#for i in range(int(16 / binner)):
#    for j in range(int(16 / binner)):
#        factor = int(binner)
#        idx_j = factor * 8 * j + 80
#        idx_i = 2500 + 310 + factor * i
#        y = ndimage.gaussian_filter1d(np.mean(np.mean(
#            new_img_reshaped[idx_i:idx_i + factor, idx_j:idx_j + 8 * factor, delta_start_idx + 10:190 - delta_end_idx],
#            axis=1), axis=0),
#                                      sigma=3) - bs_arr_real[delta_start_idx + 10:190 - delta_end_idx]
#        for r in range(20):
#            dist = 0 + 1*((r+1)/10)
#            y_att = np.multiply(y_true, np.exp(-a*dist))
#            sim = cosine_similarity(y.reshape((1,-1)), y_att.reshape((1,-1)))
#            angle = np.arccos(sim)
#            plt.scatter(dist, angle[0,0], c  = color_arr[k])
#            if i  == 0:
#                plt.scatter(dist, angle[0,0], c=color_arr[k])
#
#plt.ylabel('Spectral angle [rad]')
#plt.xlabel('Attenuation length [m]')
#plt.legend()
#plt.grid()
#plt.show()
#
#for dist in range(10):
#    for i in range(int(16/binner)):
#        for j in range(int(16/binner)):
#            factor = int(binner)
#            idx_j = factor*8 * j + 80
#            idx_i = 2500 + 310 + factor*i
#            y = ndimage.gaussian_filter1d(np.mean(np.mean(new_img_reshaped[idx_i:idx_i + factor, idx_j:idx_j + 8*factor, delta_start_idx+10:190-delta_end_idx], axis=1),axis=0),
#                                          sigma=3) - bs_arr_real[delta_start_idx+10:190-delta_end_idx]
#
#            y[y <= 0] = 0.0001
#            y_true = spectra[10+k, :]
#            y_true = interpolate_to(lib_spectra_arr[:, 0], y_true, wavelens)
#            y_true = y_true[delta_start_idx+10:190-delta_end_idx]
#
#            x1 = a.reshape((-1, 1))
#            x2 = np.log(y_true/y).reshape((-1, 1))
#            regressor.fit(x1, x2)
#            r = regressor.coef_
#            bias = regressor.intercept_
#
#            #plt.scatter(k, r, c=color_arr[k])
#
#            y_att = np.multiply(y_true, np.exp(-a * r))
#            sim = cosine_similarity(y.reshape((1, -1)), y_att.reshape((1, -1)))
#            angle = np.arccos(sim)
#            plt.scatter(r, angle[0, 0], c=color_arr[k])
#
#    #plt.scatter(k, r, c=color_arr[k], label = label_arr[k])
#
#
#plt.ylabel('Spectral angle [rad]')
#plt.xlabel('Attenuation length [m]')
#plt.legend()
#plt.grid()
#plt.show()
#
#
#
#
#for k in range(6):
#    y_true = spectra[10+k, :]
#    y_true = interpolate_to(lib_spectra_arr[:, 0], y_true, wavelens)
#    y_true = y_true[delta_start_idx+10:190-delta_end_idx]
#    for i in range(100):
#        dist = 3*(i/100)
#        y_att = np.multiply(y_true, np.exp(-a[delta_start_idx+10:190-delta_end_idx].reshape((1,-1))*dist))
#        sim = cosine_similarity(y[delta_start_idx:-delta_end_idx].reshape((1,-1)), y_att.reshape((1,-1)))
#        angle = np.arccos(sim)
#        plt.scatter(dist, angle[0,0], c  = color_arr[k])
#        if i  == 0:
#            plt.scatter(dist, angle[0,0], c=color_arr[k], label = label_arr[k])
#plt.ylabel('Spectral angle [rad]')
#plt.xlabel('Attenuation length [m]')
#plt.legend()
#plt.grid()
#plt.show()
#
#
#plt.show()
#
#
#



# Estimate through using water attenuation
new_img_reshaped_with_attenuation = attenuate_spectra(new_img_reshaped, attenuation_spectra, new_alt, field_of_view, new_roll)
np.save('testing_image_time', new_time)
#np.save('testing_image', new_img_reshaped)
np.save('altitude', altitude_total)
## Convert to digital counts
f1_reflectance = h5py.File(file_base_reflectance +'uhi_20200111_with_spread_103423_1.h5', 'r+' ,libver='latest')
f2_reflectance = h5py.File(file_base_reflectance + 'uhi_20200111_with_spread_103423_2.h5', 'r+',libver='latest')

dataset_1 = f1_reflectance['/rawdata/hyperspectral/dataCube']
dataset_2 = f2_reflectance['/rawdata/hyperspectral/dataCube']

dark_frame = f1_reflectance['rawdata/hyperspectral/calibration/radiometric/darkFrame']
radiometric_frame = f1_reflectance['rawdata/hyperspectral/calibration/radiometric/radiometricFrame']
exposure_time = 39.983 / 1000

length_1 = dataset_1.shape[0]
length_2 = dataset_2.shape[0]



# Life hack to remove noisy measurements
filter_vec = np.ones(new_img_reshaped_with_attenuation.shape[2])
filter_vec[0:10] = np.zeros(10)
filter_vec[190:] = np.zeros(len(filter_vec[190:]))

#new_img_reshaped_with_attenuation *= filter_vec
# Smoothing
new_img_reshaped_with_attenuation = ndimage.gaussian_filter(new_img_reshaped_with_attenuation, sigma = (1,3,3))
# Find the set of maximum values
new_img_reshaped_with_attenuation = new_img_reshaped_with_attenuation.reshape((-1,208))
max_val = np.amax(new_img_reshaped_with_attenuation[:,10:190], axis = 1).reshape((-1,1))
min_val = np.amin(new_img_reshaped_with_attenuation[:,10:190], axis = 1)
new_img_reshaped_with_attenuation /= max_val
new_img_reshaped_with_attenuation *= filter_vec
new_img_reshaped_with_attenuation = new_img_reshaped_with_attenuation.reshape((-1,968,208))

## Plot some spectra

#for i in range(20):
#
#    if i  == 0:
#        plt.plot(wavelens[0, 10:190], ndimage.gaussian_filter1d(new_img_reshaped_with_attenuation[100, 500 + i, 10:190], sigma=3),
#                 label='Some stone-spectra at distance ' + str(np.round(altitude_total[100]*2, decimals= 2)) + ' m', color = 'r')
#        plt.plot(wavelens[0, 10:190], ndimage.gaussian_filter1d(new_img_reshaped_with_attenuation[4000, 500 + i, 10:190], sigma=3),
#                 label='Some stone-spectra at distance ' + str(np.round(altitude_total[4000]*2, decimals= 2)) + ' m', color='g')
#    else:
#        plt.plot(wavelens[0, 10:190], ndimage.gaussian_filter1d(new_img_reshaped_with_attenuation[100, 500 + i, 10:190], sigma=3), color='r')
#        plt.plot(wavelens[0, 10:190], ndimage.gaussian_filter1d(new_img_reshaped_with_attenuation[4000, 500 + i, 10:190], sigma=3), color='g')
#
#
##plt.ylim([0.35, 1.85])
#plt.legend()
#plt.grid()
#plt.show()
#




dataset_1[:,:,:] = np.zeros((length_1, 968, 208), dtype='uint16')
dataset_2[:,:,:] = np.zeros((length_2, 968, 208), dtype='uint16')

for i in range(length_1):
    dataset_1[i,:,:] = 1000 * new_img_reshaped_with_attenuation[i,:,:] * radiometric_frame * exposure_time + dark_frame
    print(i/(length_1+length_2) *100)

for i in range(length_1, length_1 + length_2):
    dataset_2[i-length_1,:,:] = 1000 * new_img_reshaped_with_attenuation[i,:,:] * radiometric_frame * exposure_time + dark_frame
    print(i / (length_1+length_2) * 100)

f1_reflectance.close()
f2_reflectance.close()





print()






## Calculate altitude for each frame
n = new_img.shape[0]*compression
m = new_img.shape[1]
k = new_img.shape[2]
#
new_dim_img = np.zeros((n, m, k))
#
##wavelength_index = 170
##for i in range(new_img_reshaped.shape[0]):
##    for j in range(new_img_reshaped.shape[1]):
##        if new_img_reshaped[i, j, wavelength_index] != 0:
##            num = float(new_img_reshaped[i, j, wavelength_index])
##            new_img_reshaped[i, j, :] = (new_img_reshaped[i, j, :])/num
##            #new_img_reshaped[i, j, :] = -np.log(new_img_reshaped[i, j, :])
#
for i in range(compression):
    new_dim_img[i::8,0:968,:] = new_img_reshaped_with_attenuation
#
#
#
#
#arr1 = new_img_reshaped_with_attenuation[:,484,:].reshape((-1,208))
#
#arr2 = new_alt.reshape((-1,1))
#new_img_with_alt = np.concatenate((arr1,arr2), axis = 1)
#
#column_list = []
#for i in range(0, 208):
#    column_list.append(str(wavelens[0,i]))
#
#
#column_list.append('alt')
#column_list.append('vegidx')
#
#
#
## Function that takes in spectra and produces a boolean of vegetation_idxetation Veg_index = R(180)/R(160)
#veg_column = vegetation_idx(arr1, 50, 185)
#
##idx = 180
##plt.scatter(gaussian_filter(np.log(1/np.median(new_img_reshaped[:,432:496, 190], axis=1)),sigma=1), np.log(1/np.median(new_img_reshaped[:,472:496, 81])),  c = gaussian_filter(new_alt, sigma=1))
##plt.ylabel('Absorption at '+ str(int(wavelens[0,81])) + ' nm')
##plt.xlabel('Absorption at '+ str(int(wavelens[0,190])) + ' nm')
##plt.colorbar()
#####
##plt.legend()
##plt.show()
##
#arr3 = veg_column.reshape((-1,1))
#
#new_img_with_alt = np.concatenate((new_img_with_alt, arr3), axis = 1)
#
#spectra_df = pd.DataFrame(data=new_img_with_alt, columns=column_list)
#
##export_csv = spectra_df.to_csv (r'apparent_reflectance_set.csv', index = None, header=True)
#
#np.save('spectra_mid_line', new_img_with_alt)
#np.save('attenuation_david', attenuation_spectra)
#np.save('wavelengths', wavelens)
##plt.hist(veg_column, bins=120)
##plt.show()
#
#
#
#
##plt.plot(wavelens[0], light_spectrum[0], label = 'Light spectrum')
##plt.ylabel('Irradiance [W/cm^2]')
##plt.show()
##print('')
#sp.imshow(new_dim_img[:,:,:], [190, 190, 190])
##sp.imshow(new_dim_img[:,:,:], [180,100,50])
## 637 (145), 523 (81), 472 (52) nm for red, green and blue
sp.imshow(new_dim_img[:,:,:], [81,81,81])
##plt.plot(np.arange(new_alt.shape[0])*8, gaussian_filter(new_alt, sigma = 25), label = 'Filtered Altitude')
##
##model_alt = 2*gaussian_filter(new_alt, sigma = 75)
##c = attenuation_spectra[180]
##a = 1/1.13
##a = 1/3
##est_light = model_alt*(c + a) + 0.42
##plt.plot(np.arange(new_alt.shape[0])*8, est_light, label = 'Attenuation estimated from Altitude at ')
##
##plt.ylabel('Altitude [m]')
##plt.legend()
##plt.plot(wavelens[0], attenuation_spectra)
##plt.ylabel('Attenuation [1/m]')
##plt.show()
#print()