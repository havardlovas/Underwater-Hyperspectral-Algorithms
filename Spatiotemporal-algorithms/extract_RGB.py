import h5py
import numpy as np
import scipy.misc
import png
import matplotlib.image
import os

## Script that extracts all RGB files from UHI storage to PNG images

# The directory of the *.hdf-files. Files should be sorted chronologically.
directory_read = 'D:/TautraUHI/ROV_UHI/'
# Some directory for storing the red-green-blue images
directory_write = 'D:/TautraUHI/RGB_imgs_full/'
# Image index
img_idx = 0
# File index
file_idx = 0
# The number of files in the read directory.
num_files = 55
# Loop iterating through all UHI files in the directory of the *.hdf-files.
for filename in os.listdir(directory_read):
    # Only files ending with hdf.
    if filename.endswith(".hdf"):
        # Create file path by merging the directory and filename
        path = os.path.join(directory_read, filename)
        # Read the hdf files
        f_img = h5py.File(path, 'r', libver='latest')
        # Increment the file index
        file_idx = file_idx + 1
        # The image files are contained in a hierarchical path like 'rgb/pixels'. It can be explored in HDFview software
        imgs = f_img['rgb/pixels'].value
        # Iterate through each image file in the file.
        for i in range(imgs.shape[0]):
            # The image array - dimensions NxMx3.
            img_array = imgs[i]
            # Name each image by the file name.
            matplotlib.image.imsave(directory_write + filename + str(img_idx) + '.png', img_array)
            # Increment image index.
            img_idx += 1
        # Print out progress.
        print('Percetage progress: ' + str(100*file_idx/num_files))

        continue

    else:

        continue




