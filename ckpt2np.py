import tensorflow as tf
import numpy as np
import argparse, os


if __name__ == '__main__':
    print 'Example python ckpt2np.py model.ckpt-100.meta model.ckpt-100'

    parser = argparse.ArgumentParser()
    parser.add_argument('meta', help='meta file, (e.g. model.ckpt-100.meta)')
    parser.add_argument('ckpt', help='checkpoint file, (e.g. model.ckpt-100 for tf 1.0, or model.ckpt-100.ckpt for older tf)')
    
    parser.add_argument('save_pth', help='save path')
    args = parser.parse_args()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(args.meta)
    new_saver.restore(sess, args.ckpt)

    net = dict()
    for var in tf.trainable_variables():
        print 'rgb_stream/'+var.name
        net['rgb_stream/'+var.name] = sess.run(var)
    
    name = args.meta.split('.')[0]
    np.save(args.save_pth, net)
