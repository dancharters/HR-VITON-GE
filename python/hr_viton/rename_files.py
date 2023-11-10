import os


def rename_files_in_directory(directory_path):
    # Get a list of all files in the directory
    file_names = os.listdir(directory_path)

    # Loop over all file names
    for file_name in file_names:
        # If the file name contains "_mask"
        if "_mask" in file_name:
            # Remove "garment_" from the file name
            new_file_name = file_name.replace("_mask", "")
            # Change to jpg and not png
            new_file_name = new_file_name.replace(".png", ".jpg")
            # Generate full file paths
            original_file_path = os.path.join(directory_path, file_name)
            new_file_path = os.path.join(directory_path, new_file_name)

            # Rename the file
            os.rename(original_file_path, new_file_path)


if __name__ == "__main__":
    rename_files_in_directory("../../../data/test/cloth-mask")
