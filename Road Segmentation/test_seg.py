import tensorflow as tf
import os
import numpy as np
import cv2
from glob import glob
from PIL import Image
import math
import matplotlib.pyplot as plt
import keras_ocr
import time
tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

num_classes = 2
model_dir = 'models/output_model_15'

# classes
road_bgr = np.array([[0, 0, 255]]) 

# function to save training images without words at the bottom left corners
def save_image():
    pipeline = keras_ocr.pipeline.Pipeline()
    image_paths = glob(os.path.join("/Users/summerthan/Desktop/currents/computer vision/assignment/Road-Semantic-Segmentation/data/training/images" ,'*.png'))
    inpaint_dir = "/Users/summerthan/Desktop/currents/computer vision/assignment/Road-Semantic-Segmentation/data/training/inpaint"

    for image_file in image_paths: 
        image = inpaint_text(image_file, pipeline)
        cv2.imwrite(os.path.join(inpaint_dir , os.path.basename(image_file)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# function to calculate midpoint
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

# function to inpaint bottom right corner text
def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
                 
    return(img)

def compute_iou(pred, label, num_classes):
    ious = []
    for cls in range(num_classes):
        # pred_inds = pred == cls
        # target_inds = label == cls
        intersection = np.logical_and(pred, label).sum()
        union = np.logical_or(pred, label).sum()
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)  # Return mean IoU over valid classes


def test_image(sess, image_shape, logits, keep_prob, input_image):
    pipeline = keras_ocr.pipeline.Pipeline()
    image_paths = glob(os.path.join("data/testing/images" ,'*.png'))
    label_paths = glob(os.path.join("data/testing/masks", '*.png'))
    label_dict = {os.path.basename(path) : path for path in label_paths}
    total_iou = 0
    total_time = 0
    num_images = len(image_paths)

    if len(image_paths) == 0 or len(label_paths) == 0:
            raise ValueError("Image or mask paths are empty, please ensure your directory paths are correct.")
    
    for image_file in image_paths:
        start_time = time.time()

        # inpaint bottom right corner text
        raw_img = inpaint_text(image_file, pipeline)
        raw_img = cv2.resize(raw_img, (576, 320))
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # run model on image
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        im_softmax = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes)

        # get prediction
        img_classes = np.argmax(im_softmax, axis=2)
        road_seg1 = (img_classes == 0).reshape(image_shape[0], image_shape[1], 1)

        # get prediction in binary to compute IOU
        road_seg2 = np.where(road_seg1, 255, 0).astype(np.uint8)
     
        # overlay mask on image 
        road_mask = np.dot(road_seg1, road_bgr).astype(np.uint8)
        result = cv2.addWeighted(raw_img, 1, road_mask, 0.5, 0)
       
        #get mIOu
        road_rgb = (61, 61, 245)
        gt_image_file = label_dict[os.path.basename(image_file)]
        gt_image = np.array(Image.open(gt_image_file).convert("RGB").resize((image_shape[1], image_shape[0])))
        gt_road = (gt_image == road_rgb).all(axis=2).reshape(*gt_image.shape[:2], 1).astype(np.uint8)*255
        iou = compute_iou(road_seg2, gt_road, num_classes)
        total_iou += iou
        
        #print out MIOU
        text = f"Image: {os.path.basename(image_file)}, IoU: {iou:.4f}"
        box_width = len(text) + 2
        print("+" + "-" * box_width + "+")
        print("| " + text + " |")
        print("+" + "-" * box_width + "+")

        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        # display results 
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
        ax[0].imshow(raw_img, cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('Image')
        ax[1].imshow(gt_road, cmap='gray')
        ax[1].axis('off')
        ax[1].set_title('Ground Truth')
        ax[2].imshow(road_seg1, cmap='gray')
        ax[2].axis('off')
        ax[2].set_title('Predicted Mask')
        ax[3].imshow(result, cmap='gray')
        ax[3].axis('off')
        ax[3].set_title('Composite Image')
        plt.tight_layout()
        plt.show()

    total_time = total_time / num_images
    mean_iou = total_iou / num_images
    text = f"Mean IoU: {mean_iou:.4f}, Mean time: {total_time:.4f}"
    box_width = len(text) + 2

    print()
    print("END OF PROGRAM")
    print("+" + "-" * box_width + "+")
    print("| " + text + "  |")
    print("+" + "-" * box_width + "+")


def main():
    num_classes = 2
    image_shape = (320, 576)

    saver = tf.train.import_meta_graph(model_dir + '/model.meta')
    graph = tf.get_default_graph()

    logits = graph.get_operation_by_name('logits').outputs[0]
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    input_image = graph.get_tensor_by_name('image_input:0')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_dir + '/model')

        # if TEST_ON_IMAGE:
        test_image(sess, image_shape, logits, keep_prob, input_image)
        # else:
        #     test_video(sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    main()

