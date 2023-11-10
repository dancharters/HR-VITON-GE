import cv2
import os


def extract_hair_region(raw_image_path, mask_image_path):
    raw_image = cv2.imread(raw_image_path)
    mask_image = cv2.imread(mask_image_path)

    # Define the color to look for in the mask (BGR format)
    color_to_extract = (0, 0, 254)

    # Create a binary mask where the mask_image has the specified color
    binary_mask = cv2.inRange(mask_image, color_to_extract, color_to_extract)

    # Extract the region from the raw image using the binary mask
    hair_region = cv2.bitwise_and(raw_image, raw_image, mask=binary_mask)

    return hair_region


def process_images_in_folder(raw_images_dir, masks_dir, output_dir):
    raw_image_files = os.listdir(raw_images_dir)
    mask_image_files = os.listdir(masks_dir)
    total_images = len(raw_image_files)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_images = 0
    for raw_image_file in raw_image_files:
        raw_image_name, _ = os.path.splitext(raw_image_file)
        mask_image_file = f'{raw_image_name}.png'  # Assuming mask files have a .png extension

        if mask_image_file not in mask_image_files:
            print(f'Mask not found for {raw_image_file}. Skipping...')
            continue

        raw_image_path = os.path.join(raw_images_dir, raw_image_file)
        mask_image_path = os.path.join(masks_dir, mask_image_file)

        hair_region = extract_hair_region(raw_image_path, mask_image_path)

        # Save the extracted hair region
        output_image_filename = f'hair_{raw_image_file}'
        output_image_path = os.path.join(output_dir, output_image_filename)
        cv2.imwrite(output_image_path, hair_region)

        processed_images += 1
        # Print progress
        print(f'Processed {processed_images}/{total_images} images, {total_images - processed_images} images left.')


def main():
    raw_images_dir = '../../data/train/image'
    masks_dir = '../../data/train/image-parse-v3'
    output_dir = '../eda_output_data/hair_regions'
    process_images_in_folder(raw_images_dir, masks_dir, output_dir)


if __name__ == "__main__":
    main()
