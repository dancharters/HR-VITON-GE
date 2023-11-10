import os
import csv
import warnings
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 240898


def load_resnet_model():
    return ResNet50(weights='imagenet', include_top=False)


def extract_features(img_path, model):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Mask black pixels
    mask = np.all(img_array == [0, 0, 0], axis=-1)
    img_array[mask] = [255, 255, 255]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


def get_image_files(image_folder):
    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(('.jpg', '.png'))]
    return image_files


def process_images_batch(image_folder, model, batch, batch_size):
    image_files = get_image_files(image_folder)
    features = []

    start_index = batch * batch_size
    end_index = min(len(image_files), start_index + batch_size)

    for file in image_files[start_index:end_index]:
        img_path = os.path.join(image_folder, file)
        img_features = extract_features(img_path, model)
        features.append(img_features)

    return image_files[start_index:end_index], features


def cluster_images(features, optimal_k):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features_scaled)

    kmeans = KMeans(n_clusters=optimal_k, random_state=SEED)
    kmeans.fit(features_reduced)

    return kmeans.predict(features_reduced)


def main():
    image_folder = '../../data/train/openpose_img'
    resnet_model = load_resnet_model()

    # Optimal K is 4 based on elbow plot
    optimal_k = 4

    # Prepare the output folder and CSV file
    output_folder = "../eda_output_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_csv_path = os.path.join(output_folder, "pose_clusters.csv")

    # Open the CSV file for writing
    with open(output_csv_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(["Image", "Cluster"])

        batch_size = 1000
        image_files = get_image_files(image_folder)
        num_batches = int(np.ceil(len(image_files) / batch_size))

        for batch in range(num_batches):
            batch_image_files, batch_features = process_images_batch(image_folder, resnet_model, batch, batch_size)
            batch_clusters = cluster_images(batch_features, optimal_k)

            for image_file, cluster in zip(batch_image_files, batch_clusters):
                csv_writer.writerow([image_file, cluster])

            print(f"Processed batch {batch + 1} of {num_batches}")


if __name__ == "__main__":
    main()
