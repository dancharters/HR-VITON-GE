import os
import cv2
from mtcnn import MTCNN
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def process_hair_colors(image_dir, batch_size=1000):
    # Prepare the output file
    output_file_path = "../eda_output_data/hair_colours_unclustered.csv"

    with open(output_file_path, "w") as f:
        f.write("image_name,avg_color_r,avg_color_g,avg_color_b\n")

    image_files = os.listdir(image_dir)
    total_images = len(image_files)
    num_batches = (total_images + batch_size - 1) // batch_size
    image_counter = 0

    rgb_colors = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_images)

        for image_name in image_files[batch_start:batch_end]:
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)

            # Check if the image is loaded correctly
            if image is None:
                print(f"Error loading image: {image_name}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hair_roi = image_rgb  # Hair region is the entire image

            # Compute the average hair color
            # Exclude black pixels when computing the average hair color
            non_black_pixels = hair_roi[(hair_roi != [0, 0, 0]).any(axis=-1)]
            # If there are no non-black pixels, skip the image
            if non_black_pixels.size == 0:
                print(f"Skipped image due to no non-black pixels: {image_name}")
                continue
            avg_hair_color = non_black_pixels.mean(axis=0)

            rgb_colors.append(avg_hair_color)

            with open(output_file_path, "a") as f:
                f.write(f"{image_name},{avg_hair_color[0]},{avg_hair_color[1]},{avg_hair_color[2]}\n")

            image_counter += 1
            print(f"Processed {image_counter} / {total_images} images")

    return rgb_colors


def cluster_hair_colors(rgb_colors, num_clusters=20):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=240898, verbose=0).fit(rgb_colors)

    # Group the colors by their cluster labels
    cluster_groups = [[] for _ in range(num_clusters)]

    for label, rgb_color in zip(kmeans.labels_, rgb_colors):
        cluster_groups[label].append(rgb_color)

    # Compute the average color for each group and convert it to HEX format
    group_info = []

    for group in cluster_groups:
        avg_rgb_color = np.mean(group, axis=0).astype(int)
        hex_color = "#{:02x}{:02x}{:02x}".format(avg_rgb_color[0], avg_rgb_color[1], avg_rgb_color[2])
        group_info.append({"avg_hex_color": hex_color, "avg_rgb_color": tuple(avg_rgb_color), "num_images": len(group)})

    return group_info


if __name__ == "__main__":
    image_dir = '../eda_output_data/hair_regions'  # Update the directory to hair region images
    rgb_colors = process_hair_colors(image_dir)
    hair_color_groups = cluster_hair_colors(rgb_colors)

    # Create a DataFrame from the list of dictionaries and save it as a CSV file
    df = pd.DataFrame(hair_color_groups)
    output_file_path = "../eda_output_data/hair_colours.csv"
    df.to_csv(output_file_path, index=False)
