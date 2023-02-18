import cv2
import numpy as np
# Read the original image
img = cv2.imread('C:/Users\kev/computervision/masaimara.jpg') 
 
#Apply identity kernel to blur the image
kernel1 = np.ones((5, 5), np.float32) / 25

filteredImage = cv2.filter2D(src=img, ddepth=-1, kernel=kernel1)
# Display original image
cv2.imshow('Original', img)
cv2.imshow('Filtered', filteredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
