import os
import warnings
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.applications.vgg16 import VGG16, preprocess_input
import pandas as pd

warnings.filterwarnings("ignore")
# Set the random seed
SEED = 240898


def load_vgg_model():
    return VGG16(weights='imagenet', include_top=False)


def extract_features(img_path, model):
    img = Image.open(img_path).convert('L')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.stack((img_array,) * 3, axis=-1)
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
    for file in image_files:
        img_path = os.path.join(image_folder, file)
        img_features = extract_features(img_path, model)
        features.append(img_features)
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
        kmeans_model = KMeans(n_clusters=k, random_state=SEED)
        kmeans_model.fit(features_reduced)
        distortions.append(kmeans_model.inertia_)

    return K, distortions


def main():
    image_folder = '../../data/train/cloth-mask'
    vgg_model = load_vgg_model()
    image_files, features = process_images(image_folder, vgg_model)
    K, distortions = find_optimal_k(features)

    # Plot elbow plot, but must close afterward to carry on with code
    plot_elbow(K, distortions)

    # Save elbow data to a CSV file
    elbow_data = pd.DataFrame({"k": K, "distortion": distortions})
    elbow_data.to_csv("../eda_output_data/garment_clusters_elbow_data.csv", index=False)

    # Optimal number is around 5, run again and save the distortion per group to plot in R
    # Repeat by using garments as is and removing background?
    optimal_k = int(input("Enter the optimal number of clusters (from the elbow plot): "))
    clusters = cluster_images(features, optimal_k)
    print("Predicted clusters:", clusters)

    # Save clusters to a CSV file
    df = pd.DataFrame({"image": image_files, "cluster": clusters})
    df.to_csv("../eda_output_data/garment_clusters.csv", index=False)


if __name__ == "__main__":
    main()
