import cv2
import numpy as np
import os

# Read the image
img = cv2.imread('region filling.png', 0)

# Define the seed point for the flood fill algorithm
seed_point = (100, 100)

# Define the fill value
fill_value = 255

# Set the connectivity for the flood fill algorithm
connectivity = 4

# Set the flags for the flood fill algorithm
flags = connectivity | (fill_value << 8) | cv2.FLOODFILL_FIXED_RANGE

# Copy the image to store the filled region
filled_img = img.copy()

# Perform the flood fill algorithm
num_filled_pixels, filled_img, _, _ = cv2.floodFill(filled_img, mask=None, seedPoint=seed_point, newVal=fill_value, loDiff=0, upDiff=0, flags=flags)

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Filled Image', filled_img)

# Save the filled image in the same directory as the original image
filename = os.path.splitext(os.path.basename('region filling.png'))[0] + '_filled.png'
cv2.imwrite(filename, filled_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
