import cv2
import numpy as np

# Load the input image
img = cv2.imread('boundary.png', cv2.IMREAD_GRAYSCALE)
# Define the kernel size for erosion and dilation
kernel_size = (5, 5)

# Apply erosion to the grayscale image
erosion = cv2.erode(img, kernel_size, iterations=1)

# Apply dilation to the grayscale image
dilation = cv2.dilate(img, kernel_size, iterations=1)

# Calculate the boundary of the grayscale image
boundary = dilation - erosion

# Display the original and processed images
cv2.imshow('Original Image', img)
cv2.imshow('Boundary', boundary)

# Wait for a key press
cv2.waitKey(0)

# Save the output images
cv2.imwrite('Boundary.jpg', boundary)
cv2.imwrite('Boundary_with_original.jpg', cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.8, cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR), 0.2, 0))

# Close all windows
cv2.destroyAllWindows()
