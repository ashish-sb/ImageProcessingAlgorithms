import cv2
import numpy as np

# Load the input image
img = cv2.imread('input_image.jpg')
img_float64 = cv2.imread('input_image.jpg').astype('float64')/255.0

# Convert the input image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply different edge detection algorithms on the input image
edges_canny = cv2.Canny(img_gray, 30, 100)
edges_sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
edges_sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
edges_laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=5)

img = np.float64(img)
new_image_list = [img]
img_list = [edges_canny, edges_laplacian]
for each in img_list:
    imgm = cv2.merge((each,each,each))
    new_image_list.append(imgm)

# Create the output image by stacking the results horizontally
#output_img = np.hstack((img_gray, edges_canny, edges_sobelx, edges_sobely, edges_laplacian))
output_img = np.hstack([x for x in new_image_list])

# Display the input and output images side by side
#cv2.imshow('Input vs Output', output_img)
cv2.imwrite('output_img.jpg', output_img)
#cv2.waitKey(0)