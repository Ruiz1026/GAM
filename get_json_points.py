import os
import json


images_folder = "data/Breast/train/image"
masks_folder = "data/Breast/train/mask"
masks_folder1 = "data/Breast/train/points"

images_files = os.listdir(images_folder)
masks_files = os.listdir(masks_folder)


filename_mapping = {}


for image_file in images_files:
    if image_file in masks_files:
        image_path = images_folder+'/'+image_file
        mask_path = masks_folder1+'/'+image_file[:-4]+'.npz'
        #mask_path = masks_folder+'/'+image_file
        filename_mapping[image_path] = mask_path  #Train:image2heatmap_train.json 

with open("data/Breast/image2points_train.json", "w") as json_file:
    json.dump(filename_mapping, json_file, indent=4)