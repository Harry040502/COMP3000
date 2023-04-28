import json
import os
import shutil

def filter_and_copy_images(keyword, source_images_folder, source_annotations_file, output_images_folder, output_annotations_file):
    # Create the output directory if it doesn't exist
    os.makedirs(output_images_folder, exist_ok=True)

    with open(source_annotations_file, 'r') as f:
        data = json.load(f)

    annotations_list = data['annotations']
    filtered_data = [item for item in annotations_list if keyword.lower() in item['caption'].lower()]

    print(f"Number of images found for keyword '{keyword}': {len(filtered_data)}")

    # Copy the images to the output directory
    for item in filtered_data:
        src_image_path = os.path.join(source_images_folder, f"{item['image_id']:012}.jpg")
        dest_image_path = os.path.join(output_images_folder, f"{item['image_id']:012}.jpg")
        
        print(f"Copying {src_image_path} to {dest_image_path}")
        
        shutil.copy(src_image_path, dest_image_path)

    # Save the filtered annotations to the output file
    with open(output_annotations_file, 'w') as f:
        json.dump(filtered_data, f)

keyword = input("Enter a keyword to filter images: ")

source_images_folder = "train2017/train2017/"
source_annotations_file = "train2017/annotations_trainval2017/annotations/captions_train2017.json"
output_images_folder = "filtered_images/"
output_annotations_file = "filtered_annotations.json"

filter_and_copy_images(keyword, source_images_folder, source_annotations_file, output_images_folder, output_annotations_file)
