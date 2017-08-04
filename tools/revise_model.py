import _init_paths
import os
import sys
import numpy as np
import cv2
import json

def makeifnotexists(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

caffe_root = '../caffe-fast-rcnn/'

sys.path.insert(0, caffe_root + 'python')

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2

prototxt = 'model/train_val.prototxt'
caffe_model = 'model/squeezenet_v1.1.caffemodel'

caffe.set_mode_cpu()

net = caffe.Net(prototxt, caffe_model, caffe.TEST)

net.params['fire6/squeeze1x1_associate'][0].data[...] = net.params['fire6/squeeze1x1'][0].data
net.params['fire6/expand1x1_associate'][0].data[...] = net.params['fire6/expand1x1'][0].data
net.params['fire6/expand3x3_associate'][0].data[...] = net.params['fire6/expand3x3'][0].data

net.params['fire7/squeeze1x1_associate'][0].data[...] = net.params['fire7/squeeze1x1'][0].data
net.params['fire7/expand1x1_associate'][0].data[...] = net.params['fire7/expand1x1'][0].data
net.params['fire7/expand3x3_associate'][0].data[...] = net.params['fire7/expand3x3'][0].data

net.params['fire8/squeeze1x1_associate'][0].data[...] = net.params['fire8/squeeze1x1'][0].data
net.params['fire8/expand1x1_associate'][0].data[...] = net.params['fire8/expand1x1'][0].data
net.params['fire8/expand3x3_associate'][0].data[...] = net.params['fire8/expand3x3'][0].data

net.params['fire9/squeeze1x1_associate'][0].data[...] = net.params['fire9/squeeze1x1'][0].data
net.params['fire9/expand1x1_associate'][0].data[...] = net.params['fire9/expand1x1'][0].data
net.params['fire9/expand3x3_associate'][0].data[...] = net.params['fire9/expand3x3'][0].data

net.save('model/SqueezeNet.v2.caffemodel')

model = caffe_pb2.NetParameter()
f = open(caffe_model, 'rb')
model.ParseFromString(f.read())
f.close()

layers = model.layer

for layer in layers:
	print layer.name
