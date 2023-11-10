import torch
from carvekit.api.high import HiInterface
import os

# Check doc strings for more information
interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)


def process_images():
    # Directory containing the original images
    input_dir = "../../../data/test/cloth"

    # Directory to save the processed images
    output_dir = "../../../data/test/inpainting_steps/garment_no_background"

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
        images_without_background = interface([input_path])
        img_wo_bg = images_without_background[0]

        # Create the full output path
        filename = filename.replace(".jpg", ".png")
        output_path = os.path.join(output_dir, filename)

        # Save the processed image
        img_wo_bg.save(output_path)

        # Counter
        total_images -= 1
        # Print progress
        print(f'{total_images} images left')


if __name__ == "__main__":
    process_images()
