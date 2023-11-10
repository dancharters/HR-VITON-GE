from PIL import Image
import numpy as np
import os


def create_binary_mask(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert PIL Image to numpy array
    img_np = np.array(image)

    # If the image has an alpha channel
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        # Use the alpha channel as the mask, convert any transparency (alpha != 0) to white, otherwise black
        mask = np.where(img_np[:, :, 3] != 0, 255, 0)

        # Convert the numpy array back to an image
        mask_img = Image.fromarray(mask.astype('uint8'), 'L')

        return mask_img
    else:
        print('Image does not have an alpha channel.')
        return None


def process_images():
    # Directory containing the images with no backgrounf
    input_dir = "../../../data/test/inpainting_steps/garment_no_background"

    # Directory to save the processed images
    output_dir = "../../../data/test/cloth-mask"

    # If the output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Get all files in the directory
    input_files = os.listdir(input_dir)

    # Get a counter
    total_images = len(input_files)
    print(f"{total_images} images to process\n")
    # Define acceptable image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']

    for filename in input_files:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in image_extensions:
            # Counter
            total_images -= 1
            # Print progress
            print(f'{total_images} images left')
            continue

        # Create the full input path
        input_path = os.path.join(input_dir, filename)

        # Process the image
        mask = create_binary_mask(input_path)

        # Save with corrected filename
        output_path = os.path.join(output_dir, filename.replace("_inpainted.png", ".jpg"))

        # Save the processed image
        mask.save(output_path)

        # Counter
        total_images -= 1
        # Print progress
        print(f'{total_images} images left')


if __name__ == "__main__":
    process_images()
