import cv2
import numpy as np
import os
from multiprocessing import Pool


# Function to process image
def extract_garment_region(image_to_use):
    # Define the color ranges BGR -> Use colour_extractor to find this
    hair = [np.array([0, 0, 254])]  # Red range
    arm_1 = [np.array([220, 169, 51])]  # Cyan range 1
    arm_2 = [np.array([254, 254, 0])]  # Cyan range 2
    garment = [np.array([0, 85, 254])]  # Orange range

    # Load the image
    image = cv2.imread(image_to_use)

    # Separate color channels
    b, g, r = cv2.split(image)

    # Create mask for hair and arm pixels
    mask_hair = cv2.inRange(image, hair[0], hair[0])
    mask_arm_1 = cv2.inRange(image, arm_1[0], arm_1[0])
    mask_arm_2 = cv2.inRange(image, arm_2[0], arm_2[0])
    mask_condition = cv2.bitwise_or(mask_hair, mask_arm_1)
    mask_condition = cv2.bitwise_or(mask_condition, mask_arm_2)

    # Save the hair and arms masks as images
    cv2.imwrite('mask_condition.png', mask_condition)

    # Create mask for garment pixels
    mask_garment = cv2.inRange(image, garment[0], garment[0])

    # Initialize the output matrix
    output = np.zeros_like(mask_garment)

    # Loop over all the pixels
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if mask_condition[i, j]:

                # Horizontal scan to the left and right
                left_scan = np.any(mask_garment[i, :j])
                if left_scan:
                    right_scan = np.any(mask_garment[i, j + 1:])
                else:
                    right_scan = False

                # Vertical scan up and down
                up_scan = np.any(mask_garment[:i, j])
                if left_scan:
                    down_scan = np.any(mask_garment[i + 1:, j])
                else:
                    down_scan = False

                # If there is a garment pixel either left or right, and either up or down, mark the pixel
                if (left_scan and right_scan) or (up_scan and down_scan):
                    output[i, j] = 1

    output_mask = output * 255
    return output_mask


def process_image(args):
    raw_image_file, images_dir, inpainting_mask_output_dir = args
    try:
        raw_image_name, extension = os.path.splitext(raw_image_file)

        # Check if the file is an image (you can add more extensions to the list)
        if extension.lower() not in ['.png', '.jpg', '.jpeg']:
            return

        mask_output_path = os.path.join(inpainting_mask_output_dir, f'{raw_image_name}_inpainting_mask.png')
        if os.path.exists(mask_output_path):
            return

        raw_image_path = os.path.join(images_dir, raw_image_file)

        mask = extract_garment_region(raw_image_path)
        cv2.imwrite(mask_output_path, mask)

    except Exception as e:
        print(f"Failed to process image {raw_image_file}. Error: {e}")


def process_images_in_folder(images_dir, inpainting_mask_output_dir):
    raw_image_files = os.listdir(images_dir)

    if not os.path.exists(inpainting_mask_output_dir):
        os.makedirs(inpainting_mask_output_dir)

    # Prepare arguments for the process_image function
    args = [(img, images_dir, inpainting_mask_output_dir) for img in raw_image_files]

    # Use multiprocessing to process images in parallel
    with Pool() as p:
        p.map(process_image, args)

    print(f"Processed all images in directory {images_dir}")


if __name__ == "__main__":
    images_dir = "images"
    inpainting_mask_output_dir = "masks"
    process_images_in_folder(images_dir, inpainting_mask_output_dir)
