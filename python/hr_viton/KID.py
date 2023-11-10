import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance
import argparse


def load_image_as_tensor(image_path):
    with Image.open(image_path) as img:
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Ensure it's resized appropriately if not already
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 tensor
        ])
        return transform(img)


def load_images_from_directory(directory):
    image_list = []
    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        if filename.endswith(('.jpg', '.png')):
            filepath = os.path.join(directory, filename)
            tensor = load_image_as_tensor(filepath)
            image_list.append(tensor)
    return torch.stack(image_list)


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compute Kernel Inception Distance (KID) for images.")
    parser.add_argument("--real", type=str, required=True, help="Path to the real images directory.")
    parser.add_argument("--fake", type=str, required=True, help="Path to the fake images directory.")
    args = parser.parse_args()  # Parse the arguments

    # Use the parsed arguments
    directory_real = args.real
    directory_fake = args.fake

    # Load images
    real_images = load_images_from_directory(directory_real)
    fake_images = load_images_from_directory(directory_fake)

    # Compute KID
    subset_size_val = 50
    kid = KernelInceptionDistance(subset_size=subset_size_val)

    kid.update(real_images, real=True)
    kid.update(fake_images, real=False)

    mean, std = kid.compute()
    print(f"KID Mean: {mean.item()}, KID Std: {std.item()}")
