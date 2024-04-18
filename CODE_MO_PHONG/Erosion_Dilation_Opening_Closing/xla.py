import cv2
import numpy as np

# Read binary images
dilation = cv2.imread('dilation.png', 0)
erosion = cv2.imread('erosion.png', 0)
opening = cv2.imread('opening.png', 0)
closing = cv2.imread('closing.png', 0)

# Define a kernel
kernel = np.ones((5,5), np.uint8)

# Perform erosion on all images
erosion_output = cv2.erode(erosion, kernel, iterations = 1)

# Perform dilation on all images
dilation_output = cv2.dilate(dilation, kernel, iterations = 1)

# Perform opening on all images (erosion followed by dilation)
opening_output = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)

# Perform closing on all images (dilation followed by erosion)
closing_output= cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

# Display the original and processed images
cv2.imshow('Original dilation', dilation)
cv2.imshow('Dilation_ouput', dilation_output)

cv2.imshow('Original erosion', erosion)
cv2.imshow('Erosion_output', erosion_output)

cv2.imshow('Original opening', opening)
cv2.imshow('Opening_output', opening_output)

cv2.imshow('Original closing', closing)
cv2.imshow('Closing_output', closing_output)

# Save the processed images
cv2.imwrite('Erosion_output.png', erosion_output)
cv2.imwrite('Dilation_output.png', dilation_output)
cv2.imwrite('Opening_output.png', opening_output)
cv2.imwrite('Closing_output.png', closing_output)
cv2.waitKey(0)
cv2.destroyAllWindows()