from resnet_train import train
from resnet_architecture import *
from exp_config import *
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

''' Load list of  {filename, label_name, label_index} '''
def load_data(data_dir, data_lst):
    data = []
    train_lst = open(data_lst, 'r').read().splitlines()
    dictionary = open(FLAGS.dictionary, 'r').read().splitlines()
    for img_fn in train_lst:
        fn = os.path.join(data_dir, img_fn + '_depthcrop.png')
        label_name = img_fn.split('/')[0]
        label_index = dictionary.index(label_name)
        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index
        })
    return data


''' Load input data using queue (feeding)'''


def read_image_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)

    example=tf.cast(example,tf.float32)
    ''' Image Normalization (later...) '''


    return example, label


def distorted_inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]

    input_queue = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

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


def main(_):
    images, labels = distorted_inputs(FLAGS.data_dir, FLAGS.train_lst)

    is_training = tf.placeholder('bool', [], name='is_training')  # placeholder for the fusion part

    logits = inference(images,
                       num_classes=FLAGS.num_classes,
                       is_training=is_training,
                       num_blocks=[3, 4, 6, 3])
    train(is_training,logits, images, labels)


if __name__ == '__main__':
    tf.app.run(main)
