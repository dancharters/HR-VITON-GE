import cv2
import os
import numpy as np


# def create_binary_mask(image_path):
#     # Load an image from file.
#     image = cv2.imread(image_path)
#
#     # Create a mask where white (255, 255, 255) pixels are.
#     white_pixels_mask = np.all(image == [255, 255, 255], axis=-1)
#
#     # Convert the white pixels mask to a binary mask:
#     # Non-white pixels in the original image are True (so they are set to 1) and white pixels are False (set to 0).
#     binary_mask = np.where(white_pixels_mask, 0, 1).astype('uint8') * 255
#
#     # Return the binary mask.
#     return binary_mask
def create_binary_mask(image_path):
    # Load an image from file
    image = cv2.imread(image_path)

    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary thresholding
    _, binary_mask = cv2.threshold(grayscale, 252, 255, cv2.THRESH_BINARY_INV)

    # Return binary mask
    return binary_mask


def process_images_in_folder(raw_images_dir, mask_output_dir):
    raw_image_files = os.listdir(raw_images_dir)

    # Filter for .jpg, .jpeg, .png files
    image_extensions = ['.jpg', '.jpeg', '.png']
    raw_image_files = [f for f in raw_image_files if os.path.splitext(f)[1].lower() in image_extensions]

    total_images = len(raw_image_files)

    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)

    processed_images = 0
    for raw_image_file in raw_image_files:
        raw_image_name, _ = os.path.splitext(raw_image_file)

        raw_image_path = os.path.join(raw_images_dir, raw_image_file)
        mask_output_path = os.path.join(mask_output_dir, f'{raw_image_name}.jpg')

        mask = create_binary_mask(raw_image_path)

        # Save the mask
        cv2.imwrite(mask_output_path, mask)

        processed_images += 1
        # Print progress
        print(f'{total_images - processed_images} images left')

def main():
    raw_images_dir = '../../../data/test/cloth'
    mask_output_dir = '../../../data/test/cloth-mask'
    process_images_in_folder(raw_images_dir, mask_output_dir)


if __name__ == "__main__":
    main()
