import cv2
import numpy as np
from matplotlib import pyplot as plt

# read the image
image = cv2.imread('input_image.jpg')

# convert the image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image into a 2D array of pixels
pixel_values = image.reshape((-1, 3))

# convert the data type to float32
pixel_values = np.float32(pixel_values)

# define the criteria for KMeans algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# define the number of clusters (K)
K = 2

# apply KMeans algorithm
_, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert the centers to integer values
centers = np.uint8(centers)

# flatten the labels array and build the segmented image
segmented_image = centers[labels.flatten()]

# reshape the segmented image back to the shape of the original image
segmented_image = segmented_image.reshape((image.shape))

# display the original and segmented image
plt.subplot(121),plt.imshow(image)
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(segmented_image)
plt.title('Segmented Image when K = {}'.format(K))
plt.xticks([]), plt.yticks([])
plt.show()
