import cv2
import numpy as np
import os

# Load the image
image_filename = "components.png"
image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

# Binarize the image
threshold = 128
binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

# Label connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

# Extract the connected components with an area between 500 and 5000 pixels
connected_components = []
for i in range(1, num_labels):
    component_mask = labels == i
    component_area = stats[i, cv2.CC_STAT_AREA]
    if 500 <= component_area <= 5000:
        component_image = np.zeros_like(image)
        component_image[component_mask] = image[component_mask]
        connected_components.append(component_image)

# Save the extracted components as separate image files in the same directory as the input image
output_dir = os.path.dirname(image_filename)
for i, component_image in enumerate(connected_components):
    filename = os.path.join(output_dir, f"component_{i+1}.png")
    cv2.imwrite(filename, component_image)

# Display the extracted components
for i, component_image in enumerate(connected_components):
    cv2.imshow(f"Component {i+1}", component_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

