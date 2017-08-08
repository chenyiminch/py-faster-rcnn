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

prototxt = 'model/train.prototxt'
caffe_model = 'model/MobileNet.v2.caffemodel'

caffe.set_mode_cpu()

net = caffe.Net(prototxt, caffe_model, caffe.TEST)

net.params['conv5_6/sep_associate'][0].data[...] = net.params['conv5_6/sep'][0].data
net.params['conv5_6/sep/bn_associate'][0].data[...] = net.params['conv5_6/sep/bn'][0].data
net.params['conv5_6/sep/scale_associate'][0].data[...] = net.params['conv5_6/sep/scale'][0].data
net.params['conv6/dw/bn_associate'][0].data[...] = net.params['conv6/dw/bn'][0].data
net.params['conv6/dw/scale_associate'][0].data[...] = net.params['conv6/dw/scale'][0].data
net.params['conv6/sep_associate'][0].data[...] = net.params['conv6/sep'][0].data
net.params['conv6/sep/bn_associate'][0].data[...] = net.params['conv6/sep/bn'][0].data
net.params['conv6/sep/scale_associate'][0].data[...] = net.params['conv6/sep/scale'][0].data




net.save('model/New_MobileNet.v2.caffemodel')

model = caffe_pb2.NetParameter()
f = open(caffe_model, 'rb')
model.ParseFromString(f.read())
f.close()

layers = model.layer

for layer in layers:
	print layer.name
