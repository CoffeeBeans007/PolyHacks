import os

# Specify the directory where you want to delete files
directory_path = r'D:\ssd\Start_Here_Mac.app\Saves\CGsE\Tryts'

# Specify the file extension you want to delete (e.g., '.txt', '.jpg', '.pdf')
file_extension_to_delete = '.wbpg'

# Loop through files in the directory and delete files with the specified extension
for filename in os.listdir(directory_path):
    if filename.endswith(file_extension_to_delete):
        file_path = os.path.join(directory_path, filename)
        try:
            os.remove(file_path)
            print(f"Deleted: {filename}")
        except OSError as e:
            print(f"Error deleting {filename}: {e}")