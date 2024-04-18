import cv2
import numpy as np

# Read the image
img = cv2.imread('the_hit_or_miss.png', cv2.IMREAD_GRAYSCALE)

# Define the kernels for hit-or-miss transform
kernel1 = np.array([[0, 0, 0], [0, 1, 0], [-1, -1, -1]], dtype=np.int32)
kernel2 = np.array([[-1, -1, -1], [0, 1, 0], [0, 0, 0]], dtype=np.int32)

# Perform the hit-or-miss transform
hitmiss1 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel1)
hitmiss2 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel2)

# Combine the two hit-or-miss transformed images
hitmiss = cv2.bitwise_and(hitmiss1, hitmiss2)

# Display and save the results
cv2.imshow('Original Image', img)
cv2.imshow('Hit-or-Miss Transform 1', hitmiss1)
cv2.imshow('Hit-or-Miss Transform 2', hitmiss2)
cv2.imshow('Combined Hit-or-Miss Transform', hitmiss)

cv2.imwrite('hit_or_miss_1.png', hitmiss1)
cv2.imwrite('hit_or_miss_2.png', hitmiss2)
cv2.imwrite('combined_hit_or_miss.png', hitmiss)

cv2.waitKey(0)
cv2.destroyAllWindows()
