# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np
from skimage import color, io
from skimage.io import imsave


# Define flags
tf.app.flags.DEFINE_integer('training_iteration', 1,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', './', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

evaluate = False
export = True


def plotImage(image):
    plt.imshow(image)
    plt.show()


def convertToLab(image):
    lab = color.rgb2lab(image)
    X_batch = lab[:, :, 0]
    Y_batch = lab[:, :, 1:] / 128
    return X_batch.reshape(X_batch.shape + (1,)), Y_batch


def parseImage(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [400, 400], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


# file_test_paths = ['woman.jpg']
file_test_paths = list(map((lambda x: "../../train3/" + x), os.listdir("../../train3/")))

dataset = tf.data.Dataset.from_tensor_slices(file_test_paths)
dataset = dataset.map(parseImage)
dataset = dataset.map(lambda image:
                      tuple(tf.py_func(
                          convertToLab, [image], [tf.double, tf.double]
                      ))
                      )
dataset = dataset.batch(1)

iterator = dataset.make_one_shot_iterator()


def conv2DRelu(X, W, B, strides, padding):
    # strides: [batch_step, height_step, width_step, channel_step]
    return tf.nn.relu(tf.nn.conv2d(X, W, strides=strides, padding=padding) + B)


def conv2DTanh(X, W, B, strides, padding):
    # strides: [batch_step, height_step, width_step, channel_step]
    return tf.nn.tanh(tf.nn.conv2d(X, W, strides=strides, padding=padding) + B)


def weight(width, height, input_channels, output_channels):
    # [width, height, input channel, output channel]
    return tf.Variable(tf.truncated_normal([width, height, input_channels, output_channels], stddev=0.1))


def bias(outputChannels):
    return tf.Variable(tf.zeros([outputChannels]))  # bias for each output channel.


def upSampling2D(X, height, width):
    return tf.image.resize_images(X, [height, width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def upSampleToOriginalSize(X, size):
    return tf.image.resize_images(X, size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def Conv2D(X, input_channels, output_channels, scan=3, activation='relu', padding='SAME', strides=1):
    W = weight(scan, scan, input_channels, output_channels)
    B = bias(output_channels)
    if activation == 'relu':
        return conv2DRelu(X, W, B, [1, strides, strides, 1], padding)
    else:
        return conv2DTanh(X, W, B, [1, strides, strides, 1], padding)


Y_ = tf.placeholder(tf.float32, shape=[None, 400, 400, 2])  # True Value
X = tf.placeholder(tf.float32, shape=[None, 400, 400, 1])  # Input

Y1 = Conv2D(X, 1, 8, 3, 'relu', 'SAME', 2)
Y2 = Conv2D(Y1, 8, 8, 3, 'relu', 'SAME', 1)
Y3 = Conv2D(Y2, 8, 16, 3, 'relu', 'SAME', 1)
Y4 = Conv2D(Y3, 16, 16, 3, 'relu', 'SAME', 2)
Y5 = Conv2D(Y4, 16, 32, 3, 'relu', 'SAME', 1)
Y6 = Conv2D(Y5, 32, 32, 3, 'relu', 'SAME', 2)
Y7 = upSampling2D(Y6, 100, 100)
Y8 = Conv2D(Y7, 32, 32, 3, 'relu', 'SAME', 1)
Y9 = upSampling2D(Y8, 200, 200)
Y10 = Conv2D(Y9, 32, 16, 3, 'relu', 'SAME', 1)
Y11 = upSampling2D(Y10, 400, 400)
Y12 = Conv2D(Y11, 16, 2, 3, 'tanh', 'SAME', 1)

# Define the loss function
loss = tf.reduce_mean(tf.squared_difference(Y12, Y_), 1)

# Define an optimizer
optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
next_element = iterator.get_next()
saver = tf.train.Saver()
with tf.Session() as sess:
    # initialize the variables
    sess.run(init)

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("from the train set:")
    # images, labels = iterator.get_next()
    step = 0
    while True:

        try:
            elem = sess.run(next_element)
            print("Step:", step)
            for i in range(1):
                # print("Round:", i)
                _, luss = sess.run([optimizer, loss], feed_dict={
                    X: elem[0], Y_: elem[1]
                })
                # print("Loss:", luss[0][0][0])
            step += 1
        except tf.errors.OutOfRangeError:
            # saver.save(sess, './model/' + 'model.ckpt', global_step=step + 1)
            print("End of training dataset.")
            break

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()

print('Training done!')


if evaluate:
    testDataset = tf.data.Dataset.from_tensor_slices(file_test_paths)
    testDataset = testDataset.map(parseImage)
    testDataset = testDataset.map(lambda image:
                                  tuple(tf.py_func(
                                      convertToLab, [image], [tf.double, tf.double]
                                  ))
                                  )
    testDataset = testDataset.batch(1)

    testIterator = testDataset.make_one_shot_iterator()
    next_element = testIterator.get_next()
    with tf.Session() as session:
        elem = session.run(next_element)
        ckpt = tf.train.get_checkpoint_state('../model/')
        saver.restore(session, ckpt.model_checkpoint_path)
        feed_dict = {X: elem[0], Y_: elem[1]}
        _, ab = session.run([optimizer, Y12], feed_dict)

        # Colorize output
        ab = ab * 128

        cur = np.zeros((400, 400, 3))
        cur[:, :, 0] = elem[0][0][:, :, 0]
        cur[:, :, 1:] = ab[0]
        print("PRINTING")
        imsave("face.jpg", color.lab2rgb(cur))
        imsave("okkar_gray_version.png", color.rgb2gray(color.lab2rgb(cur)))

# Export model
if export:
    export_path_base = sys.argv[-1]
    export_path = './export_model'
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    '''
    builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           'predict_images':
               prediction_signature,
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               classification_signature,
      },
      legacy_init_op=legacy_init_op)
    '''
    builder.save()
