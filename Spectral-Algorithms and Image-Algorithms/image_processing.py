# creata a class called Images
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
def second_deg_solver(a, b, c):
    term1 = -b/(2*a)
    term2 = np.sqrt(b**2 - 4*a*c)/(2*a)
    root1, root2 = term1 - term2, term1 + term2
    return root1, root2
class Images:
    def __init__(self, n_images, image_dir, n_rows, n_columns, image_dir_write):

        self.image_dir = image_dir
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.image_dir_write = image_dir_write

    def apply_mask(self, r_x, r_y, x_0, y_0):
        # The function follows (x-x_0)^2 / r_x^2 + (y-y_0)^2 / r_y ^2 < 1
        # the unit is pixel number
        # Initiate a mask matrix
        self.mask_mat = np.zeros((self.n_rows,self.n_columns,4), dtype = np.uint8)
        # Find the start row and end row
        if y_0 - r_y < 0:
            self.row_start = 0
        else:
            self.row_start = int(y_0 - r_y)
        if y_0 + r_y >= self.n_rows:
            self.row_end = self.n_rows
        else:
            self.row_end = int(y_0 + r_y)
        # Iterate through the image array
        # Define index array containing all indices
        self.col_idx_arr = np.zeros((self.n_rows,2))
        # Identify the column index array describing which pixels to include
        for row in range(self.row_start,self.row_end):
            y = float(row)
            root1, root2 = second_deg_solver(a = 1, b = -2*x_0, c =
                                            x_0**2 + (r_x**2 /r_y**2 )*( y - y_0 )**2 - r_x**2 )
            if root1 < 0:
                self.col_idx_arr[row,0] = 0
            else:
                self.col_idx_arr[row, 0] = np.ceil(root1)

            if root2 >= self.n_columns:
                self.col_idx_arr[row,1] = self.n_columns
            else:
                self.col_idx_arr[row, 1] = np.floor(root2)

            col_start = int(self.col_idx_arr[row, 0])
            col_end = int(self.col_idx_arr[row, 1])
            for col in range(col_start, col_end):
                self.mask_mat[row, col] = 1

        for i in range(self.n_images):
            self.images[i] = np.multiply(self.images[i], self.mask_mat)

    def avg_image(self):
        # Initiate an index to be used for division
        idx = 0
        # Initiate an average image array
        avg_sum = np.zeros((self.n_rows, self.n_columns, 3), dtype = 'float64')

        for filename in os.listdir(self.image_dir):
            # Create file path by merging the directory and filename
            path = os.path.join(self.image_dir, filename)
            img = np.asarray(Image.open(path))
            avg_sum += img[:,:,0:3]
            idx += 1
            if idx % 100:
                print(idx*100/22169)
        self.avg = (avg_sum / idx).astype('float64')
        self.n_images = idx
    def avg_over_image_channel(self):
        for i in range(self.n_images):
            img = self.images[i,:,:,0:3]
            self.avgRGB = np.mean(img, axis = (0,1)).astype('float64')

            self.avgI = np.mean(self.images[i,:,:,0:3]).astype('float64')

            scale_arr = np.multiply(self.avgI, 1/self.avgRGB)

            self.images[i,:,:,0:3] = np.multiply(scale_arr, self.images[i,:,:,0:3]).astype('uint8')

    def grey_world_correction(self):
        # Define first the average image as defined by the function
        self.avg_image()
        # The average image is thus defined
        self.avg_pixel = np.mean(self.avg).astype('float64')
        # Scale pixels by the mean pixel divided by the average of that pixel location
        scale_arr = np.multiply(self.avg_pixel, 1 / self.avg)
        # Need to do something with pixels of value "inf". Use a threshold
        scale_arr[scale_arr > 100] = 0
        # Idx for counting
        idx = 0
        for filename in os.listdir(self.image_dir):
            path = os.path.join(self.image_dir, filename)
            img_input = np.asarray(Image.open(path))

            img_output = np.multiply(scale_arr, img_input[:, :, 0:3]).astype('uint8')
            filename_no_ext = filename.split('.')[0]
            matplotlib.image.imsave(self.image_dir_write + filename_no_ext + '.png', img_output)
            if idx % 100:
                print(idx*100/self.n_images)
            idx += 1


if __name__ == '__main__':
    img = Images(n_images = 22169, image_dir='D:/TautraUHI/RGB_imgs_full_new_2/', n_rows=486, n_columns=648
                 , image_dir_write= 'D:/TautraUHI/RGB_imgs_corrected/')
    # Current arbitrary mask
    #img.apply_mask(r_x = 300, r_y = 243-5, x_0 = 324.0, y_0 = 195)

    #img.avg_over_image_channel()
    img.grey_world_correction()
    #img_arr = img.images
    #img.avg_image()

    plt.imshow(img.avg.astype('int'))
    plt.show()
