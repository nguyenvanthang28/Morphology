import cv2
import numpy as np

# Read the image
img = cv2.imread('interfere.png', cv2.IMREAD_GRAYSCALE)

# Define the kernel for morphological operations
kernel = np.ones((7,7), np.uint8)

# Perform morphological opening to remove noise from the image
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Perform morphological closing to fill small holes in the image
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Apply thresholding to the image to make it clear
ret, thresh = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Morphological Filtered Image', closing)
cv2.imshow('Thresholded Image', thresh)

# Save the thresholded image
cv2.imwrite('output.png', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()


