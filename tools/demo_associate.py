#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, im_detect_conv_map
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'upper', 'dress', 'pants', 'skirt')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, net_associate, image_fn, label_fn):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_fn)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, imscales, roi_pooling_feature_map = im_detect_conv_map(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    det_results = np.empty((0, 5), dtype=np.float32)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
        threshold_inds = np.where(dets[:, -1] > CONF_THRESH)[0]
        det_results = np.vstack((det_results, dets[threshold_inds, :]))

    labels = open(label_fn, 'r').readlines()

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    # for i in inds:
    #     bbox = dets[i, :4]
    #     score = dets[i, -1]

    #     ax.add_patch(
    #         plt.Rectangle((bbox[0], bbox[1]),
    #                       bbox[2] - bbox[0],
    #                       bbox[3] - bbox[1], fill=False,
    #                       edgecolor='red', linewidth=3.5)
    #         )
    #     ax.text(bbox[0], bbox[1] - 2,
    #             '{:s} {:.3f}'.format(class_name, score),
    #             bbox=dict(facecolor='blue', alpha=0.5),
    #             fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    

    face_id = 0

    for label in labels:
        class_id = int(float(label.split(' ')[4]))
        if class_id != 9:
            continue
        xmin = int(float(label.split(' ')[0]))
        ymin = int(float(label.split(' ')[1]))
        xlen = int(float(label.split(' ')[2]))
        ylen = int(float(label.split(' ')[3]))

        face_id += 1

        color_str = ''

        if face_id%2 == 0:
            color_str = 'red'
        else:
            color_str = 'blue'

        xmax = xmin + xlen
        ymax = ymin + ylen

        ax.add_patch(
            plt.Rectangle((xmin, ymin),
                          xmax - xmin,
                          ymax - ymin, fill=False,
                          edgecolor=color_str, linewidth=3.5)
            )

        match_det = np.zeros((0, 5))

        for i in range(det_results.shape[0]):
            rect_xmin = min(xmin, det_results[i, 0]) * imscales[0]
            rect_ymin = min(ymin, det_results[i, 1]) * imscales[0]
            rect_xmax = max(xmax, det_results[i, 2]) * imscales[0]
            rect_ymax = max(ymax, det_results[i, 3]) * imscales[0]

            roi = np.zeros((1, 5))
            roi[:, 1] = rect_xmin
            roi[:, 2] = rect_ymin
            roi[:, 3] = rect_xmax
            roi[:, 4] = rect_ymax

            net_associate.blobs['data'].reshape(*roi_pooling_feature_map.shape)
            net_associate.blobs['rois'].reshape(*roi.shape)

            foward_kwargs = {'data': roi_pooling_feature_map.astype(np.float32, copy=False),
            'rois': roi.astype(np.float32, copy=False)}

            blobs_out = net_associate.forward(**foward_kwargs)

            prob_ind = blobs_out['prob_associate'].argmax()

            print rect_xmin, rect_ymin, rect_xmax, rect_ymax
            print blobs_out['prob_associate']

            if prob_ind == 1:
                match_det = np.vstack((match_det, det_results[i, :]))

        print match_det.shape

        if match_det.shape[0] > 0:
            for i in range(match_det.shape[0]):
                ax.add_patch(
                    plt.Rectangle((match_det[i, 0], match_det[i, 1]),
                                    match_det[i, 2] - match_det[i, 0],
                                    match_det[i, 3] - match_det[i, 1], fill=False,
                                    edgecolor=color_str, linewidth=3.5)
                    )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()

                

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = 'models/album/ZF/faster_rcnn_end2end_associate/test.prototxt'
    caffemodel = 'output/faster_rcnn_end2end/album_train/zf_faster_rcnn_iter_100000.caffemodel'

    associate_prototxt = 'models/album/ZF/faster_rcnn_end2end_associate/test_associate.prototxt'

    caffe.set_mode_gpu()
    caffe.set_device(0)
        
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net_associate = caffe.Net(associate_prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    img_fn = '/data/Data/renren_annotation/Process/Images/1/h_large_wRzR_70bf000000882f75.jpg'
    label_fn = '/data/Data/renren_annotation/Process/Labels/1/h_large_wRzR_70bf000000882f75.txt'

    demo(net, net_associate, img_fn, label_fn)

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']

    # for im_name in im_names:
    #     print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #     print 'Demo for data/demo/{}'.format(im_name)
    #     label_name = ''
    #     demo(net, net_associate, im_name, label_name)

    plt.show()
