# Read image 
import cv2
import numpy as np
img = cv2.imread('input_image.jpg', cv2.IMREAD_COLOR) # road.png is the filename
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)
cv2.imwrite("edges.png", edges)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 75, minLineLength=10, maxLineGap=50)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imwrite("op.png", img)
cv2.waitKey(0)