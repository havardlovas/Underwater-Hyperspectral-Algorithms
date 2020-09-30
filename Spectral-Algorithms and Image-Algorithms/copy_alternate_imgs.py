import os
import shutil

path_to_your_files = 'D:\TautraUHI\RGB_imgs_corrected'
copy_to_path = 'D:\TautraUHI\RGB_imgs_corrected_subset'

files_list = sorted(os.listdir(path_to_your_files))
orders = range(6, len(files_list), 10)

for order in orders:
    files = files_list[order] # getting 1 image after 3 images
    shutil.copyfile(os.path.join(path_to_your_files, files), os.path.join(copy_to_path, files))  # copying images to destination folder