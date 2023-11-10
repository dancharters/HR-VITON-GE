from cleanvision.imagelab import Imagelab
import csv
import pandas as pd


def main(type="image"):
    """
    Use cleanvision.Imagelab to get the image quality analysis of all the images in a chosen image folder.
    :param type: Either 'image' or 'garment'
    :return: CSV files in the '../eda_output_data' folder
    """
    if type == "image":
        imagelab = Imagelab(data_path="../../data/train/image")
        # Define the string to remove
        string_to_remove = '/Users/danielcharters/Desktop/dissertation/data/train/image/'
        # Stats filename
        stats_filename = "women_images_stats.csv"
        # Define the CSV file name for duplicates
        csv_file = "../eda_output_data/image_duplicates.csv"
    elif type == "cloth":
        imagelab = Imagelab(data_path="../../data/train/cloth")
        # Define the string to remove
        string_to_remove = '/Users/danielcharters/Desktop/dissertation/data/train/cloth/'
        # Stats filename
        stats_filename = "garment_images_stats.csv"
        # Define the CSV file name for duplicates
        csv_file = "../eda_output_data/garment_duplicates.csv"
    else:
        print("Please enter either 'image' or 'cloth' for 'type' variable")


    imagelab.find_issues()
    imagelab.report(verbosity=4, num_images=0)

    # Get stats
    stats = pd.DataFrame(imagelab.get_stats()).reset_index(names="image")
    # Remove the string from each row in the 'column'
    stats['image'] = stats['image'].str.replace(string_to_remove, '', regex=False)

    # Save this file to csv
    stats.to_csv("../eda_output_data/" + stats_filename, index=False)

    # Keep only the image name and the issues from duplicates
    # Write the rows to the CSV file
    with open(csv_file, mode='w') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['image1', 'image2', 'issue'])

        for key in ["exact_duplicates", "near_duplicates"]:
            duplicates = imagelab.info[key]

            sets = [[img.replace(string_to_remove, '') for img in entry] for entry in duplicates['sets']]

            # Write the rows
            for row in sets:
                writer.writerow(row + [key])


if __name__ == '__main__':
    main(type="image")
    main(type="cloth")
