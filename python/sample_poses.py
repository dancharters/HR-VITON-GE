import os
import pandas as pd
import shutil
import random


def main():
    # set the random seed
    random.seed(240898)

    # set the directory paths
    data_dir = "../../data/train/openpose_img"
    output_dir = "../eda_output_data/pose_cluster_samples"

    # load the data
    data = pd.read_csv("../eda_output_data/pose_clusters.csv")

    # group the data by cluster
    grouped_data = data.groupby("Cluster")

    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop over each cluster and sample 3 images
    for cluster, group in grouped_data:
        sample_images = group.sample(n=3, replace=True)["Image"].tolist()
        for image in sample_images:
            # Strip the _rendered.png from the image and add .jpg in its place
            # image = image.replace("_rendered.png", ".jpg")
            # copy the image to the output directory with the cluster added to the file name
            input_path = os.path.join(data_dir, image)
            output_path = os.path.join(output_dir, f"{image[:-4]}_{cluster}.jpg")
            shutil.copyfile(input_path, output_path)


if __name__ == "__main__":
    main()
