from PIL import Image
import numpy as np
import cv2


def extract_colors(image_path):
    image = Image.open(image_path)

    # Convert the image to RGB mode
    image_rgb = image.convert('RGB')

    colors = image_rgb.getcolors(maxcolors=image_rgb.width * image_rgb.height)

    unique_colors = set()
    for count, color in colors:
        unique_colors.add(color)

    return unique_colors


# Keep those colours
import cv2
import numpy as np

def keep_colours(image_path, output_path):
    raw_image = cv2.imread(image_path)

    # Define the colors to look for (BGR format)
    colors_to_extract = [(0, 0, 254), (0, 0, 128), (0, 85, 254), (85, 0, 0)]

    # Initialize an empty binary mask
    binary_mask = np.zeros_like(raw_image[:, :, 0], dtype=np.uint8)

    # Iterate through the colors and update the binary mask
    for color in colors_to_extract:
        color_mask = cv2.inRange(raw_image, color, color)
        binary_mask = cv2.bitwise_or(binary_mask, color_mask)

    # Extract the region from the raw image using the binary mask
    color_filtered_image = cv2.bitwise_and(raw_image, raw_image, mask=binary_mask)

    # Convert non-black pixels to black and black pixels to white
    height, width, _ = color_filtered_image.shape
    for i in range(height):
        for j in range(width):
            if np.all(color_filtered_image[i, j] == [0, 0, 0]):
                color_filtered_image[i, j] = [255, 255, 255]
            else:
                color_filtered_image[i, j] = [0, 0, 0]

    # Save the final image
    cv2.imwrite(output_path, color_filtered_image)

def main(image_path):
    unique_colors = extract_colors(image_path)

    print("Total unique colors:", len(unique_colors))
    print("List of unique colors:")
    for color in unique_colors:
        print(f"RGB: {color}")


if __name__ == "__main__":
    image_path = '../../../data/test_inpainting/inpainting_steps/inpainting_masks/11777_00_inpainting_mask.png'
    main(image_path)
    # # Example usage
    # keep_colours(image_path, 'occlusion_mask.jpg')
