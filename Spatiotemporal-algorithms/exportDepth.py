import sys, Metashape

folder = 'D:/GIS/'


filename = folder + "altitude_all.txt"

# Open the file with writing permission
myfile = open(filename, 'w')




chunk = Metashape.app.document.chunk
scale = chunk.transform.scale
i = 0
for camera in chunk.cameras:
    if bool(camera.transform):
        depth = chunk.model.renderDepth(camera.transform, camera.sensor.calibration)

        #depth_scaled = Metashape.Image(depth.width, depth.height, " ", "F32")

        myfile.write(str(i) + ', ' + str(depth[324, 423][0] * scale) + '\n')
        #for y in range(depth.height):
        #    for x in range(depth.width):
        #        depth_scaled[x, y] = (depth[x, y][0] * scale,)
        #savepath = folder + 'img' + str(i) + '.tif'
        #depth_scaled.save(savepath)
        i = i + 1


# Write a line to the file


# Close the file
myfile.close()