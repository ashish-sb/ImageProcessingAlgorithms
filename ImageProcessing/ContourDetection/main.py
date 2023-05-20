import cv2
import numpy as np

# read the image
image = cv2.imread('input_image.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply thresholding to segment the image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw contours on the image
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

# display the result
cv2.imwrite('op.jpg', image)
cv2.waitKey(0)
