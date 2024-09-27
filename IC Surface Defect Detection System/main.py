#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:33:07 2024

@author: summerthan
"""
import cv2
import os 
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import load_img, get_ic_coordinates, preprocess_image, segment_characters, get_alphabet, get_user_labels, detect_surface_defect

#initialize variables
total_images = 0
correct_predictions = 0
total_time = 0.0
passed_images = 0
wrong_predictions = []
img_collection=[]
markings_passed = None

print("""
+===================================================================+
|    Instructions:                                                  |
|    1) To use the program, enter the batch number of the images    |
|      you want to process below.                                   | 
|    2) Do not include any spaces, only include the batch number    |
|      according to the sequence you see on the IC chip.            |
|    3) Sequence should follow up to bottom, left to right.         |
+===================================================================+
          """)

#get input for ic labels
labels = get_user_labels() 
if not all(label.isalnum() for label in labels):
    print("""
+===========================================================+
| Invalid input. Please enter only alphanumeric characters. |
+===========================================================+
          """)
    exit()

# image path location
folder_path = "dataset/good/580543KG4AGZ2"

#load images from the folder
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    img_collection.append(item_path)
    total_images += 1

#initialize variables
correct_markings = total_images
correct_surface = total_images
passed_images = total_images

#load template for image
template_list = load_img("dataset/template_list.tif")
gray_templates=[]
for tmp in template_list:
        gray_templates.append(cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY))

name1 = 0
#loop through images in the image collection, these images are in tiff files
for img in img_collection:
    name = os.path.basename(img)
    start = time.time()
    try:
        img_list = load_img(img)
    except Exception as e:    
        print(f"An error occurred for image {1}: {e}")
        continue  # Skip to the next iteration

    #turn the images into gray scale
    gray_images=[]
    for img in img_list:
        gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    #image segmentation
    cropped_images = []
    try:
        x1, x2, y1, y2 = get_ic_coordinates(gray_images[1], gray_templates[1])
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
    
    #image preprocessing
    processed_image, for_scratch = preprocess_image(cropped_images[0])  
   
    #character segmentation
    segmented_image, characters= segment_characters(processed_image)

    #initialize  the list with the same length as characters
    alphabets = [None] * len(characters) 
    
    #concurrent programming
    with ThreadPoolExecutor(max_workers=None) as executor:
        try: 
            #surface defects module
            future_surface_checking = executor.submit(detect_surface_defect, cropped_images[0],segmented_image, for_scratch, correct_surface, name)
            img, isdefected, correct_surface, annotated_defect = future_surface_checking.result()
            
            #alphabet defects module
            futures_alphabet_checking = {executor.submit(get_alphabet, character, labels[index]): index for index, character in enumerate(characters)}
            for future in as_completed(futures_alphabet_checking):
                index = futures_alphabet_checking[future]
                alphabet = future.result()
                if alphabet == "FAILED":
                    correct_markings -= 1
                    markings_passed = "FAILED"
                    break
                else:
                    alphabets[index] = alphabet
        except Exception as e:
            print(e)

    if isdefected == "FAILED" or markings_passed == "FAILED":
                print(isdefected)
                print(markings_passed)
                print("deduct")
                passed_images -= 1

    end = time.time()
    time_this_iteration = end-start
    total_time = total_time+ float(time_this_iteration)
    markings_passed = None

width = 36
average_time = f"{total_time/total_images:.3f}"

print("+" + "=" * (width) + "+")
print("|   Average Processing Time: " + str(average_time).ljust(width - 28) + "|")
print("|   Markings Passed: " + str(correct_markings).ljust(width - 20) + "|")
print("|   Surface Passed: " + str(correct_surface).ljust(width - 19) + "|")
print("|   Total Passed: " + str(passed_images).ljust(width - 17) + "|")
print("|   Total Images: " + str(total_images).ljust(width - 17) + "|")
print("+" + "=" * (width) + "+")