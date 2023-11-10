from PIL import Image
import os

def count_black_pixels(image):
    """
    Count the number of black pixels in the image.
    :param image: PIL.Image object.
    :return: Number of black pixels in the image.
    """
    black_pixel = 0
    pixels = list(image.getdata())
    return pixels.count(black_pixel)

def get_avg_black_pixels(directory):
    """
    Calculate the average number of black pixels in images from the specified directory.
    :param directory: Directory path containing the images.
    :return: Average number of black pixels in the images.
    """
    total_images = 0
    total_black_pixels = 0
    n = 2023
    # Go through all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                total_black_pixels += count_black_pixels(img)
                total_images += 1
        n-=1
        print(n)

    # Calculate the average
    if total_images == 0:
        return 0
    return total_black_pixels / total_images

if __name__ == "__main__":
    directory_path = "../../../data/test_inpainting/inpainting_steps/inpainting_masks"
    avg_black_pixels = get_avg_black_pixels(directory_path)
    print(f"Average number of black pixels: {avg_black_pixels}")


