import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread('input_image.jpg')

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur to remove noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# apply adaptive thresholding to segment the image
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

# apply morphological opening to remove small objects
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# find contours in the image
cnts, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw contours on the original image
output = image.copy()
for c in cnts:
    cv2.drawContours(output, [c], -1, (0, 255, 0), 2)

# display the result
plt.imsave("op.png",output[:,:,::-1])
plt.show()