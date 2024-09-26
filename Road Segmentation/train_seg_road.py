#!/usr/bin/env python3
import os.path
import tensorflow as tf
import random
from glob import glob
import numpy as np
import keras
from PIL import Image
import cv2
import keras_ocr
import math

tf = tf.compat.v1

CLOUD_MODE = True
if CLOUD_MODE:
    data_dir = '/input'
else:
    data_dir = './data'


def resize_with_pad(image, target_height, target_width, pad_color=(0, 0, 0)):
    # Get the original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate the scaling factor
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new size while maintaining aspect ratio
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create a new image with the target size and pad color
    padded_image = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
    
    # Calculate padding
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    # Place the resized image in the center
    padded_image[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized_image
    
    return padded_image

def load_vgg(sess):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    try:
        tf.compat.v1.saved_model.load(sess, [vgg_tag], "models/vgg")
        image_input = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_output = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_output = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_output = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    except Exception as e:
        raise NameError(f"Pretrained VGG16 model is not loaded properly: {e}")
        

    return image_input, keep_prob, layer3_output, layer4_output, layer7_output


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolution on VGG output7
    conv1x1 = tf.compat.v1.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                               kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                               kernel_regularizer=keras.regularizers.l2(1e-2))

    # upsampling
    upsample_1 = tf.compat.v1.layers.conv2d_transpose(conv1x1, num_classes, 4, strides=(2, 2), padding='same',
                                            kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=keras.regularizers.l2(1e-2))

    # convolution on pre-layer to make the layer to be connected have the same shape
    vgg_layer4_reshape = tf.compat.v1.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                          kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer=keras.regularizers.l2(1e-2))

    # skip connection 1(element-wise addition)
    skip_layer_1 = tf.add(upsample_1, vgg_layer4_reshape)

    # upsampling
    upsample_2 = tf.compat.v1.layers.conv2d_transpose(skip_layer_1, num_classes, 4, strides=(2, 2), padding='same',
                                            kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=keras.regularizers.l2(1e-2))

    # conv on vgg_layer3_out to make same shape
    vgg_layer3_reshape = tf.compat.v1.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                          kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer=(keras.regularizers.l2(1e-2)))

    # skip connection 2
    skip_layer_2 = tf.add(upsample_2, vgg_layer3_reshape)

    # final upsampling
    upsample_final = tf.compat.v1.layers.conv2d_transpose(skip_layer_2, num_classes, 16, strides=(8, 8), padding='same',
                                                kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer=keras.regularizers.l2(1e-2))

    return upsample_final


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # each row: a pixel, each column: each class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(correct_label)))
    # optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

def compute_miou(logits, correct_label, num_classes):
    with tf.compat.v1.variable_scope('metrics'):
        predictions = tf.argmax(logits, axis=-1)
        labels = tf.argmax(correct_label, axis=-1)
        miou, miou_op = tf.compat.v1.metrics.mean_iou(labels, predictions, num_classes)
    return miou, miou_op


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, image_shape):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()


    model_dir = "models/output_model"

    print('Start Training...\n')
    for i in range(epochs):
        print('Epoch {} ...'.format(i + 1))
        for image, label in get_batches_fn(batch_size, image_shape):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5 ,
                                          learning_rate: 0.0009})
            print('Loss = {:.3f}'.format(loss))
        print()
    saver.save(sess, os.path.join(model_dir, 'model'))
    print('model saved!')

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

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

#get batch images and labels
def get_batches_fn(batch_size, image_shape):
    img_dir = "data/training/inpaint"
    image_paths = glob(os.path.join(img_dir, '*.png'))
    label_dir = "data/training/masks"
    label_paths = glob(os.path.join(label_dir, '*.png'))
    label_dict = {os.path.basename(path) : path for path in label_paths}

    if len(image_paths) == 0 or len(label_paths) == 0:
        raise ValueError("Image or mask paths are empty, please ensure your directory paths are correct.")

    #road mask color
    road_rgb = (61, 61, 245)

    random.shuffle(image_paths)

    for batch_i in range(0, len(image_paths), batch_size):
        images = []
        gt_images = []
        for image_file in image_paths[batch_i:batch_i + batch_size]:
            
            # get ground truth image
            gt_image_file = label_dict[os.path.basename(image_file)]

            #resize image
            image = np.array(Image.open(image_file).convert("RGB").resize(image_shape))  
            
            # segment road out of ground truth mask
            gt_image = np.array(Image.open(gt_image_file).convert("RGB").resize(image_shape))
            gt_road = (gt_image == road_rgb).all(axis=2).reshape(*gt_image.shape[:2], 1)
            gt_others = ~(gt_road)
            gt_image = np.concatenate((gt_road, gt_others), axis=2)
        
            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)

def run():
    num_classes = 2
    image_shape = (576, 320)

    with tf.compat.v1.Session() as sess:
        # define epoch and batch size and placeholder for learning rate
        epochs = 40
        batch_size = 4
        learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        #hold ground truth label for each pixel image
        correct_label = tf.compat.v1.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')

        #load pretrained vgg model
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess)

        #build FCN using vgg layers
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        #optimize layers
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, image_shape)


if __name__ == '__main__':
    run()

