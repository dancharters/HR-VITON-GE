import os


def list_files(directory_path):
    try:
        # List all entries in the directory
        entries = os.listdir(directory_path)
        # Filter the list to include files only
        return [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Get folder paths from the user
folder1_path = "../../../data/results/output_images/with_gt_garments_paired"
folder2_path = "../../../data/results/output_images/with_inpainted_garments_paired"

# List files in both folders
folder1_files = set(list_files(folder1_path))
folder2_files = set(list_files(folder2_path))

# Identify files unique to each folder
unique_to_folder1 = folder1_files - folder2_files
unique_to_folder2 = folder2_files - folder1_files

# Print results
if unique_to_folder1:
    print(f"\nFiles unique to {folder1_path}:")
    for file in sorted(unique_to_folder1):
        print(file)

if unique_to_folder2:
    print(f"\nFiles unique to {folder2_path}:")
    for file in sorted(unique_to_folder2):
        print(file)

if not unique_to_folder1 and not unique_to_folder2:
    print("\nBoth folders contain the same files.")

