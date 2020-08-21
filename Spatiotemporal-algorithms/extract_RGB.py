import h5py
import numpy as np
import scipy.misc
import png
import matplotlib.image
import os
from PIL import Image
def calc_img_idx_str(idx):
    if idx < 10:
        return_string = '0000'+ str(idx)
    elif idx < 100:
        return_string = '000' + str(idx)
    elif idx < 1000:
        return_string = '00' + str(idx)
    elif idx < 10000:
        return_string = '0' + str(idx)
    else:
        return_string = str(idx)
    return return_string
## Code that extracts all RGB files from UHI storage to PNG images
def h5_to_PNG(dir_r, dir_w, n_f):
    # Image index
    img_idx = 0
    # File index
    file_idx = 0
    # Loop iterating through all UHI files in the directory of the *.hdf-files.
    for filename in os.listdir(dir_r):
        # Only files ending with hdf.
        if filename.endswith(".hdf"):
            # Split the filename to lose the extension
            filename_no_ext = filename.split('.')[0]

            # Create file path by merging the directory and filename
            path = os.path.join(dir_r, filename)
            # Read the hdf files
            f_img = h5py.File(path, 'r', libver='latest')
            # Increment the file index
            file_idx = file_idx + 1
            # The image files are contained in a hierarchical path like 'rgb/pixels'. It can be explored in HDFview software
            imgs = f_img['rgb/pixels'][()]

            # Iterate through each image file in the file.
            for i in range(imgs.shape[0]):
                img_idx_str = calc_img_idx_str(img_idx)
                # Name each image by the file name.
                matplotlib.image.imsave(dir_w + filename_no_ext + '_' + img_idx_str + '.png', imgs[i])
                #im = Image.fromarray(imgs[i])
                #im.save(dir_w + filename_no_ext + str(img_idx) + '.png')
                # Increment image index.
                img_idx += 1
            # Print out progress.
            print('Percetage progress: ' + str(100 * file_idx / n_f))

            continue

        else:

            continue



# An example from my local directory
if __name__ == '__main__':
    # The directory of the *.hdf-files. Files should be sorted chronologically.
    directory_read = 'D:/TautraUHI/ROV_UHI/'
    # Some directory for storing the red-green-blue images
    directory_write = 'D:/TautraUHI/RGB_imgs_full_new_2/'
    # The number of files in the read directory. Only used to evaluate percentage progress, and can be set to any number.
    num_files = 55
    # Save the images as PNGs
    h5_to_PNG(directory_read, directory_write, num_files)






