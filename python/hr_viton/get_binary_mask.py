import cv2
import numpy as np

# load an image from file
image = cv2.imread('data/test/cloth/garment_01149_00_inpainted.jpg')

# convert to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# binary thresholding
_, binary_mask = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)

# Save binary mask to file
cv2.imwrite('data/test/cloth-mask/garment_01149_00_inpainted.jpg', binary_mask)
