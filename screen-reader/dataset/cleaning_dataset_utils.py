import json

# Path to the JSON file
file_path = "dataset.json"

# Load the JSON data from the file
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Modify the imageUrl fields
for item in json_data['data']:
    if 'imageUrl' in item:
        item['imageUrl'] = item['imageUrl'].replace("'", "")

# Save the updated JSON data back to the file
with open(file_path, 'w') as file:
    json.dump(json_data, file, indent=2)

import os

# Path to the images folder
images_folder = "images"

# Iterate over all files in the images folder
for filename in os.listdir(images_folder):
    # Construct the full file path
    old_file_path = os.path.join(images_folder, filename)
    
    # Skip directories (if any)
    if not os.path.isfile(old_file_path):
        continue
    
    # Remove single quotes from the filename
    new_filename = filename.replace("'", "")
    new_file_path = os.path.join(images_folder, new_filename)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {filename} -> {new_filename}")

print("File renaming completed!")