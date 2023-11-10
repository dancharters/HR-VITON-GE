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
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


def get_image_files(image_folder):
    image_files = os.listdir(image_folder)
    image_files = [file for file in image_files if file.endswith(('.jpg', '.png'))]
    return image_files


def process_images(image_folder, model):
    image_files = get_image_files(image_folder)
    features = []
    total_images = len(image_files)

    # Prepare the output folder and CSV file
    output_folder = "../eda_output_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_csv_path = os.path.join(output_folder, "pose_features.csv")

    # Open the CSV file for writing
    with open(output_csv_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(["Image", "Features"])

        for index, file in enumerate(image_files):
            img_path = os.path.join(image_folder, file)
            img_features = extract_features(img_path, model)
            features.append(img_features)

            # Write the image file and its features to the CSV file
            csv_writer.writerow([file, img_features])

            print(f"Processed image {index + 1} of {total_images}")

    return image_files, features



def cluster_images(features, optimal_k):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features_scaled)

    kmeans = KMeans(n_clusters=optimal_k, random_state=SEED)
    kmeans.fit(features_reduced)

    return kmeans.predict(features_reduced)


def plot_elbow(K, distortions):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def find_optimal_k(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features_scaled)

    distortions = []
    K = range(1, 31)
    for k in K:
        kmeans_model = KMeans(n_clusters=k, random_state=SEED, verbose=0)
        kmeans_model.fit(features_reduced)
        distortions.append(kmeans_model.inertia_)

    return K, distortions


def main():
    image_folder = '../../data/train/openpose_img'
    resnet_model = load_resnet_model()
    image_files, features = process_images(image_folder, resnet_model)
    # Write features out to a csv file
    # K, distortions = find_optimal_k(features)
    #
    # plot_elbow(K, distortions)
    #
    # elbow_data = pd.DataFrame({"k": K, "distortion": distortions})
    # elbow_data.to_csv("../eda_output_data/pose_clusters_elbow_data.csv", index=False)
    #
    # optimal_k = int(input("Enter the optimal number of clusters (from the elbow plot): "))
    optimal_k = 6
    clusters = cluster_images(features, optimal_k)
    print("Predicted clusters:", clusters)

    df = pd.DataFrame({"image": image_files, "cluster": clusters})
    df.to_csv("../eda_output_data/pose_clusters.csv", index=False)


if __name__ == "__main__":
    main()
