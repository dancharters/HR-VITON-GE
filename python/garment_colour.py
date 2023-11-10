import os
import cv2
import numpy as np
import warnings
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd

warnings.filterwarnings("ignore")


def mask_garment(image, lower_bound, upper_bound):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=~mask)
    return result


def get_dominant_color(image, k=3):
    pixels = image.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=-1)]  # Remove black pixels from the background mask
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    most_common = Counter(kmeans.labels_).most_common(1)
    dominant_color = kmeans.cluster_centers_[most_common[0][0]]
    return dominant_color


def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def brightness(color):
    return (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)


def group_colors(hex_colors, n_clusters=40):
    colors = [np.array([int(hex_color[i:i+2], 16) for i in (1, 3, 5)]) for hex_color in hex_colors.keys()]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(colors)
    clustered_colors = [[] for _ in range(n_clusters)]

    for hex_color, count in hex_colors.items():
        color = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
        cluster = kmeans.predict([color])[0]
        clustered_colors[cluster].append((color, count))

    sorted_colors = sorted([(np.mean([color for color, _ in cluster], axis=0), sum(count for _, count in cluster)) for cluster in clustered_colors], key=lambda x: brightness(x[0]))

    return [(rgb_to_hex(color), count) for color, count in sorted_colors]


def brightness(color):
    return (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)


def get_distributions(hex_colors):
    df = pd.DataFrame(hex_colors, columns=['Color', 'Count'])

    # Convert hex colors to RGB and calculate brightness
    df['RGB'] = df['Color'].apply(lambda x: tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
    df['Brightness'] = df['RGB'].apply(brightness)

    # Sort DataFrame by brightness (closeness to white)
    df = df.sort_values(by='Brightness', ascending=True)

    return df.to_csv("../eda_output_data/garment_colours.csv", index=False)


image_folder = "../../data/train/cloth"
n_images = len([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
hex_colors = Counter()

background_lower_bound = np.array([240, 240, 240])  # Lower bound for white-type background
background_upper_bound = np.array([255, 255, 255])  # Upper bound for white-type background

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Unable to read {img_name}. Skipping.")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = mask_garment(img, background_lower_bound, background_upper_bound)

    dominant_color = get_dominant_color(img)
    hex_color = rgb_to_hex(dominant_color)
    hex_colors[hex_color] += 1
    n_images -= 1
    print(f"{n_images} images left")


if __name__ == "__main__":
    # Run the functions
    grouped_hex_colors = group_colors(hex_colors)
    get_distributions(grouped_hex_colors)
