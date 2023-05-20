import cv2
import numpy as np

# read the image
image = cv2.imread('input_image.jpg')

# convert the image to grayscale and apply thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# define kernel for erosion and dilation
kernel = np.ones((5,5), np.uint8)

# apply morphological operations
# erosion
erosion = cv2.erode(thresh, kernel, iterations=1)

# dilation
dilation = cv2.dilate(thresh, kernel, iterations=1)

# opening
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# closing
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# display the result
cv2.imshow('Original Image', image)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imwrite("op.jpg", dilation)