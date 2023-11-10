import cv2
import os
import numpy as np

def extract_garment_region(raw_image_path, mask_image_path, mask_output_path):
    raw_image = cv2.imread(raw_image_path)
    mask_image = cv2.imread(mask_image_path)

    # Define the color to look for in the mask (BGR format) -> RGB = 254, 85, 0 -> BGR = 0, 85, 254
    color_to_extract = (0, 85, 254)

    # Create a binary mask where the mask_image has the specified color
    binary_mask = cv2.inRange(mask_image, color_to_extract, color_to_extract)

    # Save the binary mask
    cv2.imwrite(mask_output_path, binary_mask)

    # Extract the region from the raw image using the binary mask
    garment_region = cv2.bitwise_and(raw_image, raw_image, mask=binary_mask)

    # Make all black (0,0,0) pixels white (255,255,255)
    garment_region[(garment_region == [0, 0, 0]).all(axis=2)] = [255, 255, 255]

    # Convert the BGR image to BGRA (with alpha channel)
    garment_region = cv2.cvtColor(garment_region, cv2.COLOR_BGR2BGRA)

    # Make all white (255,255,255) pixels transparent (set alpha to 0)
    garment_region[(garment_region[:, :, 0:3] == [255, 255, 255]).all(axis=2)] = [255, 255, 255, 0]

    return garment_region


def process_images_in_folder(raw_images_dir, masks_dir, output_dir, mask_output_dir):
    raw_image_files = os.listdir(raw_images_dir)
    mask_image_files = os.listdir(masks_dir)
    total_images = len(raw_image_files)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)

    processed_images = 0
    for raw_image_file in raw_image_files:
        raw_image_name, _ = os.path.splitext(raw_image_file)
        mask_image_file = f'{raw_image_name}.png'  # Assuming mask files have a .png extension

        if mask_image_file not in mask_image_files:
            print(f'Mask not found for {raw_image_file}. Skipping...')
            continue

        raw_image_path = os.path.join(raw_images_dir, raw_image_file)
        mask_image_path = os.path.join(masks_dir, mask_image_file)
        mask_output_path = os.path.join(mask_output_dir, f'{raw_image_name}_mask.png')

        garment_region = extract_garment_region(raw_image_path, mask_image_path, mask_output_path)

        # Save the extracted garment region as PNG
        output_image_filename = f'garment_{raw_image_file}.png'
        # Save as .png
        output_image_filename = output_image_filename.replace('.jpg', '')
        output_image_path = os.path.join(output_dir, output_image_filename)
        cv2.imwrite(output_image_path, garment_region)

        processed_images += 1
        # Print progress
        print(f'Processed {processed_images}/{total_images} images, {total_images - processed_images} images left.')


import numpy as np

def keep_specific_color(image_path, output_path, color):
    # Load the image
    image = cv2.imread(image_path)

    # Define the color range
    lower = np.array(color, dtype=np.uint8)
    upper = np.array(color, dtype=np.uint8)

    # Create a mask of the pixels within the color range
    mask = cv2.inRange(image, lower, upper)

    # Create an image only containing the pixels within the color range
    result = cv2.bitwise_and(image, image, mask=mask)

    # Save the result
    cv2.imwrite(output_path, result)


def main():
    raw_images_dir = '../../data/train/image'
    masks_dir = '../../data/train/image-parse-v3'
    output_dir = '../../data/train/extracted_garments'
    mask_output_dir = '../../data/train/extracted_garments_mask'
    process_images_in_folder(raw_images_dir, masks_dir, output_dir, mask_output_dir)

    # Get orange segment for architecture diagrams
    # keep_specific_color('../../data/train/image-parse-v3/00003_00.png', '../../report/images/garment_region.png', [0, 85, 254])


if __name__ == "__main__":
    main()
