import os
import cv2
from mtcnn import MTCNN
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def process_skin_colors(image_dir, batch_size=1000):
    # Initialize the face detector
    detector = MTCNN()

    # Prepare the output file
    output_file_path = "../eda_output_data/skin_colours_unclustered.csv"

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
            faces = detector.detect_faces(image_rgb)

            # If a face is detected, compute the average skin color
            if faces:
                x, y, w, h = faces[0]["box"]
                face_roi = image_rgb[y:y+h, x:x+w]
                face_lab = cv2.cvtColor(face_roi, cv2.COLOR_RGB2Lab)
                avg_skin_color = face_lab.mean(axis=(0, 1))

                lab_color = np.array([[[avg_skin_color[0], avg_skin_color[1], avg_skin_color[2]]]], dtype=np.uint8)
                rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_Lab2RGB).squeeze()
                rgb_colors.append(rgb_color)

                with open(output_file_path, "a") as f:
                    f.write(f"{image_name},{rgb_color[0]},{rgb_color[1]},{rgb_color[2]}\n")

            image_counter += 1
            print(f"Processed {image_counter} / {total_images} images")

    return rgb_colors


def cluster_skin_colors(rgb_colors, num_clusters=20):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=240898).fit(rgb_colors)

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
    image_dir = '../../data/train/image'
    rgb_colors = process_skin_colors(image_dir)
    skin_color_groups = cluster_skin_colors(rgb_colors)

    # Create a DataFrame from the list of dictionaries and save it as a CSV file
    df = pd.DataFrame(skin_color_groups)
    output_file_path = "../eda_output_data/skin_colours.csv"
    df.to_csv(output_file_path, index=False)
