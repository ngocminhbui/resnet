from resnet_train import train
from resnet_architecture import *
import tensorflow as tf
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS

''' Load list of  {filename, label_name, label_index} '''
''' Load list of  {filename, label_name, label_index} '''
def load_data(data_dir, data_lst):
    data = []
    train_lst = open(data_lst, 'r').read().splitlines()
    dictionary = open(FLAGS.dictionary, 'r').read().splitlines()

    assert FLAGS.mode == 'depth' or FLAGS.mode == 'rgb' , 'Wrong mode, depth or rgb'

    for img_fn in train_lst:

        if FLAGS.mode == 'depth':
            fn = os.path.join(data_dir, img_fn + '_depthcrop.png')
        else:
            fn = os.path.join(data_dir, img_fn + '_crop.png')

        label_name = img_fn.split('/')[0]
        label_index = dictionary.index(label_name)
        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index
        })
    return data


''' Load input data using queue (feeding)'''


mean_im = None
if mean_im == None:
    if FLAGS.mode == 'rgb':
        mean_fn = FLAGS.mean_rgb
    elif FLAGS.mode == 'depth':
        mean_fn = FLAGS.mean_dep
    mean_im = np.load(mean_fn)
    mean_im = mean_im[..., ::-1]  # BGR to RGB, because decode_png gives RGB image

    print 'assigned mean_im', 'mode',FLAGS.mode,'path', mean_fn

def read_image_from_disk(input_queue):
    global  mean_im

    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)

    example = tf.cast(example, tf.float32) - tf.convert_to_tensor(mean_im, dtype=tf.float32)

    return example, label


def distorted_inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]

    input_queue = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

    # read image and label from disk
    image, label = read_image_from_disk(input_queue)



    # data augmentation
    # resize_method = np.random.randint(4)
    # image = tf.image.resize(image, FLAGS.input_size, FLAGS.input_size, resize_method)
    image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # subtract off the mean and divide the variance of the pixels
    image = tf.image.per_image_standardization(image)

    # generate batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples)

    return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])


''' Non shuffle inputs , just for evaluation because of slow running  '''
def inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [d['filename'] for d in data]
    label_indexes = [d['label_index'] for d in data]

    input_queue = tf.train.slice_input_producer([filenames, label_indexes], shuffle=False)

    # read image and label from disk
    image, label = read_image_from_disk(input_queue)

    ''' Data Augmentation '''
    image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, 3])
    image = tf.image.random_flip_left_right(image)

    # generate batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples)

    return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])
