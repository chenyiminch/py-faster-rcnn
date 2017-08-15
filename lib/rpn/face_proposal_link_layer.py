# ----------------------------------
# Yimin Chen
# Read the face boxes, face_id, proposal, generate the target association labels
# ----------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

class FaceProposalLinkLayer(caffe.Layer):
	"""
	Assign labels to a set of pairs of faces and proposals
	"""
	def setup(self, bottom, top):
		layer_params = yaml.load(self.param_str_)
		self._num_classes = layer_params['num_classes']
		
		# fused boxes (0, x1, y1, x2, y2)
		top[0].reshape(1, 5)
		# labels
		top[1].reshape(1, 1)
	def forward(self, bottom, top):
		# Proposals from RPN
		all_rois = bottom[0].data
		# GT boxes
		gt_boxes = bottom[1].data
		# Face ids
		face_ids = bottom[2].data
		# Face boxes
		face_boxes = bottom[3].data
		# Face flag
		face_flag = bottom[4].data
		
		zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
		all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1])))) 
		
		assert np.all(all_rois[:, 0] == 0), 'Only single iterm batches are supported'
		
		num_images = 1
		
		rois_per_image = cfg.TRAIN.ASSOCIATE_BATCH_SIZE/num_images
		fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
		
		if face_flag[0, 0] == 1:	
			rois, labels = _sample_rois_(all_rois, gt_boxes, face_ids, face_boxes, fg_rois_per_image, rois_per_image)
		else:
			rois = all_rois
			labels = -1 * np.ones((all_rois.shape[0], 1), dtype=np.int16)
		top[0].reshape(*rois.shape)
		top[0].data[...] = rois

		top[1].reshape(*labels.shape)
		top[1].data[...] = labels

	def backward(self, top, propagated_down, bottom):
		"""
		This layer does not propagate gradients.
		"""
		pass
	def reshape(self, bottom, top):
		pass

def _generate_samples_(boxes, face_labels, face_boxes):

	all_pos = np.empty((0, 10), dtype=np.float)
	all_neg = np.empty((0, 10), dtype=np.float)

	for i in range(face_boxes.shape[0]):
		face_label = face_boxes[i, 4] 
		pos_inds = np.where(face_labels[:] == face_label)[0]
		neg_inds = np.where(face_labels[:] != face_label)[0]
		
		rep_face_boxes_pos = np.tile(face_boxes[i, :], (pos_inds.shape[0], 1))
		rep_face_boxes_neg = np.tile(face_boxes[i, :], (neg_inds.shape[0], 1))
		
		pos_labels = np.hstack((boxes[pos_inds], rep_face_boxes_pos))
		neg_labels = np.hstack((boxes[neg_inds], rep_face_boxes_neg))

		all_pos = np.concatenate((all_pos, pos_labels), axis=0)
		all_neg = np.concatenate((all_neg, neg_labels), axis=0)
	return all_pos, all_neg

def _sample_rois_(all_rois, gt_boxes, face_ids, face_boxes, fg_rois_per_image, rois_per_image):
	"""
	Generate a random set of ROIs comprising fg and bf for the association classification
	"""
	overlaps = bbox_overlaps(
		np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
		np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
	gt_assignment = overlaps.argmax(axis=1)
	max_overlaps = overlaps.max(axis=1)
	match_face_ids = face_ids[gt_assignment]
		
	used_inds = np.where(max_overlaps >= cfg.TRAIN.ASSOCIATE_FG_THRESH)[0]

	used_boxes = all_rois[used_inds]
	used_face_ids = match_face_ids[used_inds]
	
	all_pos, all_neg = _generate_samples_(used_boxes, used_face_ids, face_boxes)
	
	fg_rois_per_this_image = min(fg_rois_per_image, all_pos.shape[0])

	pos_inds = np.arange(fg_rois_per_this_image)
		
	if all_pos.shape[0] > 0:
		pos_inds = npr.choice(all_pos.shape[0], size=fg_rois_per_this_image, replace=False)

	bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

	bg_rois_per_this_image = min(bg_rois_per_this_image, all_neg.shape[0])

	neg_inds = np.arange(bg_rois_per_this_image)
	
	if all_neg.shape[0] > 0:
		neg_inds = npr.choice(all_neg.shape[0], size=bg_rois_per_this_image, replace=False)

	labels = np.zeros((fg_rois_per_this_image + bg_rois_per_this_image, 1), dtype = np.int16)
	
	labels[0:fg_rois_per_this_image] = 1

	concat_boxes = np.concatenate((all_pos[pos_inds, :], all_neg[neg_inds, :]), axis=0)

	boxes = np.zeros((concat_boxes.shape[0], 5), dtype=np.int16)

	boxes[:, 1] = np.minimum(concat_boxes[:, 1], concat_boxes[:, 5])
	boxes[:, 2] = np.minimum(concat_boxes[:, 2], concat_boxes[:, 6])
	boxes[:, 3] = np.maximum(concat_boxes[:, 3], concat_boxes[:, 7])
	boxes[:, 4] = np.maximum(concat_boxes[:, 4], concat_boxes[:, 8])

	return boxes, labels
