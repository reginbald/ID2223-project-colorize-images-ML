import numpy as np
import scipy.io

data = scipy.io.loadmat("imagenet-vgg-verydeep-19.mat")
weights = np.squeeze(data['layers'])

layers = (
    # 'conv1_1', 'relu1_1',
    'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

netDict = {}
for i, name in enumerate(layers):
    kind = name[:4]
    if kind == 'conv':
        kernels, bias = weights[i + 2][0][0][0][0]
        netDict[name] = {'kernels': kernels, 'bias': bias}

np.save('vgg_model.npy', netDict)