import cv2
import numpy as np
import os

# Load the image in grayscale mode
img = cv2.imread('123.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to convert it into binary image
_, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# Apply skeletonization
size = np.size(img_bin)
skel = np.zeros(img_bin.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False
while not done:
    eroded = cv2.erode(img_bin, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img_bin, temp)
    skel = cv2.bitwise_or(skel, temp)
    img_bin = eroded.copy()

    zeros = size - cv2.countNonZero(img_bin)
    if zeros == size:
        done = True

# Save the result in the same directory
filename = os.path.splitext(os.path.basename('123.png'))[0]
output_path = f"{filename}_skeleton.png"
cv2.imwrite(output_path, skel)

# Display the result
cv2.imshow("Original Image", img)
cv2.imshow("Skeletonized Image", skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
