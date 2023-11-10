import os
import warnings
import cv2
import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Load the pre-trained VGG16 model without the top layers
vgg_model = VGG16(weights='imagenet', include_top=False)


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
    return [file for file in image_files if file.endswith(('.jpg', '.png'))]


def determine_optimal_clusters(X):
    distortions = []
    K = range(1, 20)
    for k in K:
        kmeans_model = KMeans(n_clusters=k)
        kmeans_model.fit(X)
        distortions.append(kmeans_model.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def main(image_folder='../garment_images'):
    image_files = get_image_files(image_folder)

    features = []
    for file in image_files:
        img_path = os.path.join(image_folder, file)
        img_features = extract_features(img_path, vgg_model)
        features.append(img_features)

    X = np.array(features)

    # Determine the optimal number of clusters using the elbow method
    determine_optimal_clusters(X)

    # Train the K-means model with the optimal number of clusters
    optimal_k = 3  # Replace this with the optimal number of clusters obtained from the elbow plot
    kmeans = KMeans(n_clusters=optimal_k)
    print("Fitting clusters")
    kmeans.fit(X)

    # Assign each image to a cluster
    print("Predicting clusters")
    image_clusters = kmeans.predict(X)
    print(image_clusters)


if __name__ == "__main__":
    main()
