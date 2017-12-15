# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from skimage import color, io
from skimage.io import imsave

with tf.Session() as session:
    elem = session.run(next_element)
    _, height, width, _ = elem[2].shape  # Original size of image
    ckpt = tf.train.get_checkpoint_state('./model/')
    saver.restore(session, ckpt.model_checkpoint_path)
    feed_dict = {X: elem[0], Y_: elem[1], Size: [height, width]}
    _, ab = session.run([optimizer, Y16], feed_dict)

    # Colorize output
    ab = ab

    grey_scale = color.rgb2lab(elem[2])

    predict_bw = np.zeros((height, width, 3))
    predict_bw[:, :, 0] = grey_scale[0][:, :, 0]
    # print bwImg
    predict_bw = color.lab2rgb(predict_bw)
    # print colored
    plotImage(predict_bw)

    colored = np.zeros((height, width, 3))
    colored[:, :, 0] = grey_scale[0][:, :, 0]
    colored[:, :, 1:] = ab[0]

    colored_rgb = color.lab2rgb(colored)

    plotImage(colored_rgb)

    print("SAVING IMAGE")
    imsave("../result/img_yolo.png", colored_rgb)

    # Original Picture
    original_lab = color.rgb2lab(elem[2])

    original_bw = np.zeros((height, width, 3))
    original_bw[:, :, 0] = original_lab[0][:, :, 0]

    plotImage(color.lab2rgb(original_bw))
    plotImage(elem[2][0])