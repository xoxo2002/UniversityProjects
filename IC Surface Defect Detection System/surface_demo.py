#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:33:07 2024

@author: summerthan
"""
import cv2
import os 
import time
from surface_utils import load_img, get_ic_coordinates, preprocess_image, segment_characters, get_alphabet, get_user_labels, detect_surface_defect

#get input for ic labels
labels = get_user_labels() 

total_images = 0
correct_predictions = 0
total_time = 0.0
passed_images = 0
wrong_predictions = []
markings_passed = None

# image path location
img_collection=[]
folder_path = "../original_data_files/defect_rejected/defects/full"

for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    img_collection.append(item_path)
    total_images += 1

correct_markings = total_images
correct_surface = total_images
passed_images = total_images

#load template for image
template_list = load_img("../original_data_files/template_list.tif")
gray_templates=[]
for tmp in template_list:
        gray_templates.append(cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY))

name1 = 0
#loop through images in the image coll ection, these images are in tiff files
for img in img_collection:
    name = os.path.basename(img)
    print(name[:-4])
    start = time.time()
    try:
        img_list = load_img(img)
        # Perform further operations with the coordinates
    except Exception as e:    
        print(f"An error occurred for image {1}: {e}")
        continue  # Skip to the next iteration

    #turn them into gray scale
    gray_images=[]
    for img in img_list:
        gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    #get exact IC coordinates and crop the images
    cropped_images = []
    try:
        x1, x2, y1, y2 = get_ic_coordinates(gray_images[1], gray_templates[1])
        # Perform further operations with the coordinates
    except Exception as e:
        correct_markings -= 1
        correct_surface -= 1
        passed_images -= 1
        total_images -= 1
        print(f"An error occurred for image {1}: {e}")
        continue  # Skip to the next iteration
    if len(img_list) > 1:
        for i in range(len(img_list)):
            cropped_images.append(gray_images[i][y1:y2, x1:x2])
    
    processed_image, for_scratch = preprocess_image(cropped_images[0])  
    #segment characters from the ic chip
    segmented_image, characters= segment_characters(processed_image)
    alphabets = [None] * len(characters)  # Initialize  the list with the same length as characters
    
    img, isdefected, correct_surface, annotated_defect = detect_surface_defect(cropped_images[0],segmented_image, for_scratch, correct_surface, name)
    if isdefected == "FAILED":
                passed_images -= 1

    end = time.time()
    time_this_iteration = end-start
    total_time = total_time+ float(time_this_iteration)

    print("whole program: " + str(end-start))

width = 36
average_time = f"{total_time/total_images:.3f}"

print("#" + "=" * (width) + "#")
print("|   Surface Passed: " + str(correct_surface).ljust(width - 19) + "|")
print("|   Total Images: " + str(total_images).ljust(width - 17) + "|")
print("#" + "=" * (width) + "#")