import cv2
import numpy as np

# read the image
image = cv2.imread('input_image.jpg')

# define the four corners of the original image
pts1 = np.float32([[50,50], [200,50], [50,200], [200,200]])

# define the four corners of the desired perspective
pts2 = np.float32([[10,100], [200,50], [100,250], [300,200]])

# calculate the transformation matrix and apply it
M = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, M, (400, 300))

# display the result
# cv2.imshow('Original Image', image)
# cv2.imshow('Wrapped Image', result)
cv2.imwrite("op.png", result)
cv2.waitKey(0)
