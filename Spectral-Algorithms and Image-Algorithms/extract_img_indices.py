import pandas as pd
import numpy as np
import os
import shutil

# Should use
filename = 'D:/GIS/nav.txt'

df_navigation = pd.read_csv(filename, sep=',', header=1)

Y = df_navigation['Y_est'].values


X = df_navigation['X_est'].values

Z = df_navigation['Z_est'].values

#X = df_pos['X'].values.reshape(-1)
#Y = df_pos['Y'].values.reshape(-1)


x0 = X[0]
y0 = Y[0]
z0 = Z[0]
count = 0
gridded_indices = np.zeros(3697, dtype=np.int)

dist_min = 0.2
for i in range(X.shape[0]):
    x = X[i]
    y = Y[i]
    z = Z[i]

    sq_dist = (x-x0)**2 + (y-y0)**2 + (z-z0)**2
    if sq_dist > dist_min**2:
        x0 = x
        y0 = y
        z0 = z
        gridded_indices[count] = i
        count += 1

#np.save('gridded_indices.npy')


path_to_your_files = 'D:\TautraUHI\RGB_imgs_gw'
copy_to_path = 'D:\TautraUHI\RGB_imgs_subset_double'

files_list = sorted(os.listdir(path_to_your_files))
orders = gridded_indices
all_ind = np.arange(22169)
not_orders = all_ind[all_ind != orders]
a = 0
for order in all_ind:
    if a < 3697 and order == orders[a]:
        files = files_list[order]
        shutil.copyfile(os.path.join(path_to_your_files, files), os.path.join(copy_to_path, files))  # copying images to destination folder
        a += 1
print('success')


