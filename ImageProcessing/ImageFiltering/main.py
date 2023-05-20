import cv2
import numpy as np

# read the image
image = cv2.imread('input_image.jpg')

# apply the filter
blue, green, red = cv2.split(image)
zeros = np.zeros(image.shape[:2], dtype="uint8")

# Apply blue filter and merge with original image
cool_image = cv2.merge([blue + 100, green - 75, red - 75])

# show the filtered image
cv2.imshow('Cool Image', cool_image)
cv2.waitKey(0)

# save the filtered image
cv2.imwrite('cool_image.jpg', cool_image)