#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:14:20 2024

@author: summerthan
"""
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import imutils
from pytesseract import TesseractError
from PIL import Image
from keras.models import load_model

def get_user_labels():
    labels = []
    labels = input("Enter a label: ")
    return  list(labels)

def load_img(pathname):
    # Path to the tiff file
    path = pathname
    # List to store the loaded image
    images = []
     
    _, images = cv2.imreadmulti(mats=images,
                                  filename=path,
                                  start=0,
                                  count=2,
                                  flags=cv2.IMREAD_ANYCOLOR)
 
    resized_img_list=[]
    # Show the images
    if len(images) > 1:
        for i in range(len(images)):
            resized_img_list.append(cv2.resize(images[i], None, fx=2, fy=2))
    return images

def get_ic_coordinates(img, template):
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x1, y1 = top_left
    x2, y2 = bottom_right

    return x1, x2, y1, y2

def preprocess_image(image):
    # Calculate the average pixel value of the image
    blur = cv2.GaussianBlur(image, (1, 1), 0)
    average_mean = np.mean(image)
    threshold = 1.6 * average_mean
    darkened_image = np.where(blur < threshold, 0, image)
    #erosion and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    darkened_image2 = cv2.erode(darkened_image,kernel,iterations = 1)
    msdg_image = cv2.morphologyEx(darkened_image2, cv2.MORPH_CLOSE, kernel)

    _, binary_image = cv2.threshold(msdg_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        osd_data = pytesseract.image_to_osd(cv2.bitwise_not(binary_image), lang='ocr',config='--psm 0 -c min_characters_to_try=10', output_type = Output.DICT)
        degree = osd_data['rotate']
        if degree == 180:
            binary_image = imutils.rotate(binary_image, angle=180)
            darkened_image = imutils.rotate(darkened_image, angle=180)
    except TesseractError as e:
        print(e)
    return binary_image, darkened_image

#segment the characters and return individual images
def segment_characters(img):
    total_Labels, label_ids, values, centroid = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    characters=[]
    coordinates=[]

    final_coordinates=[]
    for i in range(1, total_Labels):
        #properties of the component 
        w = values[i, cv2.CC_STAT_WIDTH] 
        h = values[i, cv2.CC_STAT_HEIGHT] 
        x1 = values[i, cv2.CC_STAT_LEFT] 
        y1 = values[i, cv2.CC_STAT_TOP] 
        #feature extract words based on width and height
        if ((w<25) and (w>10)) and ((h>15) and (h<50)):
            if  len(coordinates) == 0: 
                coordinates.append((x1, y1, w, h))
            elif (y1 - (coordinates[-1][1])) > 10:
                sorted_row = sorted(coordinates, key=lambda x: x[0])
                final_coordinates += sorted_row
                coordinates = []
                coordinates.append((x1, y1, w, h))
            else:
                coordinates.append((x1, y1, w, h))
    sorted_row = sorted(coordinates, key=lambda x: x[0])
    final_coordinates += sorted_row
    for coord in final_coordinates:
        char_img = img[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        characters.append(char_img)
    return img, characters

def get_alphabet(img, label):
    #treat image as only a single character
    reversed_binary = cv2.bitwise_not(img)
    reversed_binary=cv2.copyMakeBorder(reversed_binary, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255,255,255))
    result_df = pytesseract.image_to_data(reversed_binary, lang="ocr", config="--psm 10 --oem 1", output_type='data.frame')
    text = str(result_df.iloc[-1]["text"])[0]
    conf = result_df.iloc[-1]["conf"]
    if text == label:
        return text
    else:
        result_df = pytesseract.image_to_data(reversed_binary, lang="ocra_test5", config="--psm 10 --oem 1", output_type='data.frame')
        text2 = str(result_df.iloc[-1]["text"])[0]
        conf2 = result_df.iloc[-1]["conf"]
        if (text2 == label):
            return text2
        elif (text2 == "E"):
            text2 = "Z"
        elif (text2 == "n"):
            text2 = "A"
        elif (text2 =="a" or text2 == "‘"):
            text2 = "Q"
        elif (text2 == "u" or text2 == "o" or text2 == "<"):
            text2= "4"
        elif (text=="2" and conf <80):
            text2 ="Z"
        elif (text2=="V" and conf2<80):
            text2= "Q"
        elif ((text2 == "D" and conf2 <85) or (text == "0" and conf < 80)):
            text2 = "P"
        if text2.islower():
            text2 = text2.upper()

        if not (text2 == label):
            return "FAILED"
    return text2

def detect_surface_defect(ori_img, img, scratch_img, correct_predictions, filename):
    is_defected = "PASS"
    defect_img = scratch_img.copy()
    #edge defect areas
    words_image = cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)
    background_color = np.bincount(scratch_img.flatten()).argmax()
    remove_mask = np.zeros_like(scratch_img)
    total_Labels, label_ids, values, centroid = cv2.connectedComponentsWithStats(words_image, 
                                            4, 
                                            cv2.CV_32S)
    for i in range(0, total_Labels):
        
        area = values[i, cv2.CC_STAT_AREA]
        w = values[i, cv2.CC_STAT_WIDTH] 
        h = values[i, cv2.CC_STAT_HEIGHT] 
        x1 = values[i, cv2.CC_STAT_LEFT] 
        y1 = values[i, cv2.CC_STAT_TOP] 
        if (w<45) and (w>12) and (h>20) and (h<50) and area>130:
            if (h<29) and (w>14) and (w<25):
                remove_mask[y1-2:y1 + h+12, x1-2:x1 + w+2,] = 255
            elif h>27:
                remove_mask[y1-2:y1 + h+2, x1-2:x1 + w+2,] = 255
            else: 
                None
        scratch_img = scratch_img.copy()
        scratch_img[remove_mask == 255] = background_color
    
    total_Labels, label_ids, values, centroid = cv2.connectedComponentsWithStats(scratch_img, 
                                            4, 
                                            cv2.CV_32S)
    
    class_labels = ['Dots', "Illegible markings", 'Non-defects', "Notch", 'Scratches']
    model = load_model('surface_defect_model.h5')
    for i in range(1, total_Labels):
        area = values[i, cv2.CC_STAT_AREA]   
        if (area > 40): 
            x1 = values[i, cv2.CC_STAT_LEFT] 
            y1 = values[i, cv2.CC_STAT_TOP] 
            w = values[i, cv2.CC_STAT_WIDTH] 
            h = values[i, cv2.CC_STAT_HEIGHT] 
            cropped_defect = scratch_img[y1-2:y1 + h + 2, x1- 2:x1 + w + 2]
            try:
                img_array = preprocess_image_for_model(cropped_defect)
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)
                predicted_label = class_labels[predicted_class[0]]
                cv2.rectangle(scratch_img, (x1-10, y1-10), (x1 + w + 10, y1 + h + 10), (255,255,255), 1)
                if (predicted_label == "Non-defects") or (predicted_label == "Notch"):
                    cv2.rectangle(defect_img, (x1-5, y1-5), (x1 + w + 5, y1 + h + 5), (255,255,255), 1)
                    #Textbox and text
                    (w, h), _ = cv2.getTextSize(predicted_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(defect_img, (x1-5, y1 - 20), (x1 + w, y1-5), (255,255,255), -1)
                    cv2.putText(defect_img, predicted_label, (x1 - 2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                    continue
                else:  
                    correct_predictions -= 1
                    #Bounding box
                    cv2.rectangle(defect_img, (x1-5, y1-5), (x1 + w + 5, y1 + h + 5), (255,255,255), 1)
                    #Textbox and text
                    (w, h), _ = cv2.getTextSize(predicted_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(defect_img, (x1-5, y1 - 20), (x1 + w, y1-5), (255,255,255), -1)
                    cv2.putText(defect_img, predicted_label, (x1 -5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                    is_defected = "FAILED"
                    break
            except Exception as e:
                # is_defected = "FAILED"
                print(e)
    return img, is_defected, correct_predictions, defect_img


def preprocess_image_for_model(img, target_size=(227, 227)):
    img = Image.fromarray(img).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array