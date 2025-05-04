import os
import json


images_folder = "data/Breast/train/image"
masks_folder = "data/Breast/train/mask"

images_files = os.listdir(images_folder)
masks_files = os.listdir(masks_folder)


filename_mapping = {}


for image_file in images_files:
    if image_file in masks_files:
        image_path = images_folder+'/'+image_file
        mask_path = masks_folder+'/'+image_file
        filename_mapping[mask_path] = image_path  #Test:label2image_test.json
with open("data/Breast/label2image_test.json", "w") as json_file:
    json.dump(filename_mapping, json_file, indent=4)