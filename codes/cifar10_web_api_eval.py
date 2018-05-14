# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:11:36 2018

@author: josef brechler
"""

import re
import tensorflow as tf
import os
import numpy as np
from math import exp
import pandas as pd

from delete_flags import delete_all_flags
delete_all_flags(tf.flags.FLAGS)

import cifar10
from cifar10_input import read_cifar10
from PIL import Image

#CHECKPOINT_DIR = 'C:/tmp/cifar10_train'
IMAGE_SIZE = 24
IMAGE_SIZE_FOR_BINARY = 32
NUM_CLASSES = 100
IMAGE_FNM = "img_for_prediction.jpg"


FLAGS = tf.app.flags.FLAGS
APP_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

tf.app.flags.DEFINE_string('train_dir', os.path.join(APP_ROOT, 'train'),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('image_dir', os.path.join(APP_ROOT, 'images'),
                           """Directory images for classification """
                           """are stored.""")


def evaluate_with_api(image_fnm=IMAGE_FNM):
    """Main function for prediction with API. Wrapper around run_eval.
	Args:
	image_fnm - name of supplied image file
	
	Returns:
    """

    # generate file name for the binary
    bin_fnm = re.sub("jpg$|jpeg$|png$", "bin", image_fnm)
    
    # generate binary file
    construct_binary(image_fnm=image_fnm, bin_fnm=bin_fnm)
    
    # reset graph (sometimes throws an error of not reset)
    tf.reset_default_graph() 
    
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = generate_cnn_inputs(bin_fnm = bin_fnm)
        
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        
        df_result = run_eval(logits, saver, summary_op)
    
    df_result_top = df_result[:15]
    
    return(df_result_top)
        

def construct_binary(image_fnm, bin_fnm):
    """Generates .bin file from supplied image
	Args:
	image_fnm - name of the supplied image file
	bin_fnm - name of the binary output
		
	Returns:
    """
    
    # open image
    im_orig = Image.open(os.path.join(FLAGS.image_dir, image_fnm))
    
    # resize the image to original CIFAR files size
    im_resized = im_orig.resize((IMAGE_SIZE_FOR_BINARY, IMAGE_SIZE_FOR_BINARY), 
                                Image.ANTIALIAS)
    
    im = (np.array(im_resized))
    
    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()
    label = [1]
    
    # convert to required binary string
    if NUM_CLASSES == 10:
        out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
    elif NUM_CLASSES == 100:
        out = np.array(list(label)*2 + list(r) + list(g) + list(b),np.uint8)

    # generate file name for the binary
    bin_fnm = re.sub("jpg$|jpeg$|png$", "bin", image_fnm)
    
    # save to file
    out.tofile(os.path.join(FLAGS.image_dir, bin_fnm))


def run_eval(logits, saver, summary_op):
    """Run Eval once.
    
    Args:
    logits
    saver: Saver.
    summary_op: Summary op.
    """
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
    
        else:
            print('No checkpoint file found')
            return
    
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

            logits_nums = sess.run(logits[0])
            print(logits_nums)
            classification = sess.run(tf.argmax(logits[0], 0))
            
            print(classification)
            
            # get list of classes
            if NUM_CLASSES == 10:
                classes_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", 
                                "frog", "horse", "ship", "truck"]
            elif NUM_CLASSES == 100:
                with open(os.path.join(FLAGS.data_dir,'cifar-100-binary/fine_label_names.txt'), 'r') as f:
                    x = f.readlines()
                    classes_list = [line.rstrip() for line in x]

            # print output
            print(classes_list[classification])
    
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
    
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        
        df = pd.DataFrame({'logit': logits_nums, 'class': classes_list})
        df.reset_index()
        df['prob'] = [(exp(x) / (1 + exp(x))) for x in logits_nums.tolist()]
        df = df.sort_values(by='prob', ascending=False)
		
        return df

	
def generate_cnn_inputs(bin_fnm):
    """Construct input for CIFAR evaluation using the Reader ops.
	Args:
	bin_fnm: filename that is used for binary input into the network
	
	Returns:
	images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	labels: Labels. 1D tensor of [batch_size] size.
    """
    with tf.name_scope('input'):
	# Create a queue that produces the filenames to read.
	#   filenames = ['C:/codes/stylar_hw/tensorflow_cifar10_tutorial/img_1.bin']
        complete_filename = [os.path.join(FLAGS.image_dir, bin_fnm)]
        filename_queue = tf.train.string_input_producer(complete_filename)
        
        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        
        height = IMAGE_SIZE
        width = IMAGE_SIZE
        
        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               height, width)
        
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)
        
        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])
        
        images, label_batch = tf.train.batch(
                [float_image, read_input.label],
                batch_size=1,
                num_threads=1,
                capacity=1)
        
        labels = tf.reshape(label_batch, [1])
        
        if FLAGS.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)
        return images, labels

#if __name__ == '__main__':
#    evaluate_with_api()
    
