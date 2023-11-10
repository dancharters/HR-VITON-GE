import cv2
import numpy as np


def extract_colors(image_to_use):
    # Define the color ranges in BGR format
    hair = [np.array([0, 0, 254])]  # Red range
    arm_1 = [np.array([220, 169, 51])]  # Cyan range 1
    arm_2 = [np.array([254, 254, 0])]  # Cyan range 2
    garment = [np.array([0, 85, 254])]  # Orange range

    # Load the image
    image = cv2.imread(image_to_use)

    # Create masks for each color
    mask_hair = cv2.inRange(image, hair[0], hair[0])
    mask_arm_1 = cv2.inRange(image, arm_1[0], arm_1[0])
    mask_arm_2 = cv2.inRange(image, arm_2[0], arm_2[0])
    mask_garment = cv2.inRange(image, garment[0], garment[0])

    # Combine masks
    mask = cv2.bitwise_or(mask_hair, mask_arm_1)
    mask = cv2.bitwise_or(mask, mask_arm_2)
    mask = cv2.bitwise_or(mask, mask_garment)

    # Apply mask to image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


if __name__ == "__main__":
    result = extract_colors('garment_segmentation_map.png')
    # Save the result
    cv2.imwrite('hair_arms_segmentation_map.png', result)
