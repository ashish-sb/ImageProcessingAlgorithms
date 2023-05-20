import cv2
import numpy as np
from matplotlib import pyplot as plt

# read the image
image = cv2.imread('input_image.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# perform histogram equalization on the image
equalized = cv2.equalizeHist(gray)

# display the original and equalized images
plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(equalized, cmap = 'gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
plt.show()
