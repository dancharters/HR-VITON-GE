import cv2
import numpy as np

# Load image
img = cv2.imread('../../report/images/results/14679_00.png')

# Define the color you want to keep in RGB
target_color = np.array([254, 85, 0], dtype=np.uint8)

# Convert the target color to BGR as OpenCV uses BGR
target_color_bgr = target_color[::-1]

# Create a mask of pixels matching the target color
mask = cv2.inRange(img, target_color_bgr, target_color_bgr)

# Create a black image with the same dimensions as the input image
black_img = np.zeros_like(img)

# Copy the pixels from the original image where the mask is not zero
result = np.where(mask[:,:,None]!=0, img, black_img)

# Save the result
cv2.imwrite('../../report/images/results/garment_region.jpg', result)



