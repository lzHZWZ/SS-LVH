import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
import gl
import MyDebug
from models_lib.model_gcn import *
from models_lib.model_sdne import *

DEBUG_MODEL = False
IS_NET_CONNECTION = True

class GCNResnet(nn.Module):
	def __init__(self, option, model, num_classes, in_channel=300, t=0, adj_file=None):
		super(GCNResnet, self).__init__()
		self.state = {}
		self.state['use_gpu'] = torch.cuda.is_available()
		self.opt = option
		self.is_hash = option.HASH_TASK
		self.is_usemfb = option.IS_USE_MFB
		self.pooling_stride = option.pooling_stride
		self.features = nn.Sequential(
			model.conv1,
			model.bn1,
			model.relu,
			model.maxpool,
			model.layer1,
			model.layer2,
			model.layer3,
			model.layer4,
		)
		self.num_classes = num_classes
		self.pooling = nn.MaxPool2d(14, 14)
		
		self.gc1 = GraphConvolution(in_channel, 1024)
		self.gc2 = GraphConvolution(1024, 2048)
		self.relu = nn.LeakyReLU(0.2)
		
		_adj = gen_A(self.opt.threshold_p, num_classes, self.opt.threshold_tao, adj_file)
		self.A = Parameter(torch.from_numpy(_adj).float())
		
		self.image_normalization_mean = [0.485, 0.456, 0.406]
		self.image_normalization_std = [0.229, 0.224, 0.225]
		
		self.JOINT_EMB_SIZE = option.linear_intermediate
		
		if self.is_usemfb:
			assert self.JOINT_EMB_SIZE % self.pooling_stride == 0, \
				'linear-intermediate value must can be divided exactly by sum pooling stride value!'
			self.out_in_tmp = int(self.JOINT_EMB_SIZE / self.pooling_stride)
			self.ML_fc_layer = nn.Linear(int(self.num_classes * self.out_in_tmp), int(self.num_classes))
		else:
			self.out_in_tmp = int(1)
		
		self.Linear_imgdataproj = nn.Linear(option.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)
		self.Linear_classifierproj = nn.Linear(option.CLASSIFIER_CHANNEL, self.JOINT_EMB_SIZE)
		
		if self.is_hash:
			self.hash_layer = nn.Linear(int(self.num_classes * self.out_in_tmp), option.HASH_BIT)
			self.use_tanh = nn.Tanh()
	
	def forward(self, feature, inp):
		feature = self.features(feature)
		feature = self.pooling(feature)
		feature = feature.view(feature.size(0), -1)

		inp = inp[0]
		adj = gen_adj(self.A).detach()
		x = self.gc1(inp, adj)
		x = self.relu(x)
		x = self.gc2(x, adj)

		x = torch.transpose(x, 0, 1)
		if self.is_usemfb:
			if self.state['use_gpu']:
				x_out = torch.FloatTensor(torch.FloatStorage()).cuda()
			else:
				x_out = torch.FloatTensor(torch.FloatStorage())
			for i_row in range(int(feature.shape[0])):
				img_linear_row = self.Linear_imgdataproj(feature[i_row, :]).view(1, -1)
				if self.state['use_gpu']:
					out_row = torch.FloatTensor(torch.FloatStorage()).cuda()
				else:
					out_row = torch.FloatTensor(torch.FloatStorage())
				for col in range(int(x.shape[1])):
					tmp_x = x[:, col].view(1, -1)
					classifier_linear = self.Linear_classifierproj(tmp_x)
					iq = torch.mul(img_linear_row, classifier_linear)
					iq = F.dropout(iq, self.opt.DROPOUT_RATIO, training=self.training)
					iq = torch.sum(iq.view(1, self.out_in_tmp, -1), 2)

					out_row = torch.cat((out_row, iq), 1)

				if self.is_hash == False and self.out_in_tmp != 1:
					temp_out = self.ML_fc_layer(out_row)
					out_row = temp_out

				x_out = torch.cat((x_out, out_row), 0)
		else:
			x_out = torch.matmul(feature, x)
		gl.GLOBAL_TENSOR = x_out
		
		if self.is_hash:
			hash_tmp = self.hash_layer(x_out)
			if gl.LOCAL_USE_TANH:
				hash_code_out = self.use_tanh(hash_tmp)
				hash_code_out[hash_code_out > 0] = 1
				hash_code_out[hash_code_out <= 0] = -1
				hash_code_out = hash_code_out
			else:
				hash_code_out = hash_tmp
			return hash_code_out
		
		return x_out
	
	def get_config_optim(self, lr, lrp):
		if self.is_hash:
			return [
				{'params': self.features.parameters(), 'lr': lr * lrp},
				{'params': self.hash_layer.parameters(), 'lr': lr * 0.001},
				{'params': self.gc1.parameters(), 'lr': lr},
				{'params': self.gc2.parameters(), 'lr': lr},
			]
		else:
			return [
				{'params': self.features.parameters(), 'lr': lr * lrp},
				{'params': self.gc1.parameters(), 'lr': lr},
				{'params': self.gc2.parameters(), 'lr': lr},
			]

def gcn_resnet101(opt, num_classes, t, pretrained=True, adj_file=None, in_channel=300):
	if IS_NET_CONNECTION:
		model = models.resnet101(pretrained=pretrained)
	else:
		model = models.resnet101(pretrained=False)
		model.load_state_dict(torch.load('./checkpoint/pretrained_resnet101/resnet101-5d3b4d8f.pth'))

	return GCNResnet(opt, model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)


class GCNSDNEResnet(nn.Module):
	def __init__(self, model, state={}, in_channel=300, t=0, adj_file=None):
		super(GCNSDNEResnet, self).__init__()
		self.state = {}
		self.state['use_gpu'] = torch.cuda.is_available()
		self.state = state
		self.is_hash = self.state['HASH_TASK']
		self.is_usemfb = self.state['IS_USE_MFB']
		self.pooling_stride = self.state['pooling_stride']
		self.num_classes = self.state['num_classes']
		self.features = nn.Sequential(
			model.conv1,
			model.bn1,
			model.relu,
			model.maxpool,
			model.layer1,
			model.layer2,
			model.layer3,
			model.layer4,
		)
		self.pooling = nn.MaxPool2d(14, 14)
		
		self.gc1 = GraphConvolution(in_channel, 1024)
		self.gc2 = GraphConvolution(1024, 2048)
		self.relu = nn.LeakyReLU(0.2)
		
		_adj = gen_A(self.opt.threshold_p, self.num_classes, self.opt.threshold_tao, adj_file)
		self.A = Parameter(torch.from_numpy(_adj).float())
		
		self.image_normalization_mean = [0.485, 0.456, 0.406]
		self.image_normalization_std = [0.229, 0.224, 0.225]
		
		self.JOINT_EMB_SIZE = self.state['linear_intermediate']
		
		if self.is_usemfb:
			assert self.JOINT_EMB_SIZE % self.pooling_stride == 0, \
				'linear-intermediate value must can be divided exactly by sum pooling stride value!'
			self.out_in_tmp = int(self.JOINT_EMB_SIZE / self.pooling_stride)
			self.ML_fc_layer = nn.Linear(int(self.num_classes * self.out_in_tmp), int(self.num_classes))
		else:
			self.out_in_tmp = int(1)
		
		self.Linear_imgdataproj = nn.Linear(self.state['IMAGE_CHANNEL'], self.state['JOINT_EMB_SIZE'])
		self.Linear_classifierproj = nn.Linear(self.state['CLASSIFIER_CHANNEL'], self.state['JOINT_EMB_SIZE'])
		
		if self.is_hash:
			self.hash_layer = nn.Linear(int(self.num_classes * self.out_in_tmp), self.state['HASH_BIT'])
			self.use_tanh = nn.Tanh()
	
	def forward(self, feature, inp):
		feature = self.features(feature)
		feature = self.pooling(feature)
		feature = feature.view(feature.size(0), -1)

		inp = inp[0]
		adj = gen_adj(self.A).detach()
		x = self.gc1(inp, adj)
		x = self.relu(x)
		x = self.gc2(x, adj)

		x = torch.transpose(x, 0, 1)
		if self.is_usemfb:
			if self.state['use_gpu']:
				x_out = torch.FloatTensor(torch.FloatStorage()).cuda()
			else:
				x_out = torch.FloatTensor(torch.FloatStorage())
			for i_row in range(int(feature.shape[0])):
				img_linear_row = self.Linear_imgdataproj(feature[i_row, :]).view(1, -1)
				if self.state['use_gpu']:
					out_row = torch.FloatTensor(torch.FloatStorage()).cuda()
				else:
					out_row = torch.FloatTensor(torch.FloatStorage())
				for col in range(int(x.shape[1])):
					tmp_x = x[:, col].view(1, -1)
					classifier_linear = self.Linear_classifierproj(tmp_x)
					iq = torch.mul(img_linear_row, classifier_linear)
					iq = F.dropout(iq, self.opt.DROPOUT_RATIO, training=self.training)
					iq = torch.sum(iq.view(1, self.out_in_tmp, -1), 2)
					out_row = torch.cat((out_row, iq), 1)

				if self.is_hash == False and self.out_in_tmp != 1:
					temp_out = self.ML_fc_layer(out_row)
					out_row = temp_out

				x_out = torch.cat((x_out, out_row), 0)
		else:
			x_out = torch.matmul(feature, x)

		gl.GLOBAL_TENSOR = x_out
		
		if self.is_hash:
			hash_tmp = self.hash_layer(x_out)
			if gl.LOCAL_USE_TANH:
				hash_code_out = self.use_tanh(hash_tmp)
				hash_code_out[hash_code_out > 0] = 1
				hash_code_out[hash_code_out <= 0] = -1
				hash_code_out = hash_code_out
			else:
				hash_code_out = hash_tmp
			return hash_code_out
		
		return x_out
	
	def get_config_optim(self, lr, lrp):
		if self.is_hash:
			return [
				{'params': self.features.parameters(), 'lr': lr * lrp},
				{'params': self.hash_layer.parameters(), 'lr': lr * 0.001},
				{'params': self.gc1.parameters(), 'lr': lr},
				{'params': self.gc2.parameters(), 'lr': lr},
			]
		else:
			return [
				{'params': self.features.parameters(), 'lr': lr * lrp},
				{'params': self.gc1.parameters(), 'lr': lr},
				{'params': self.gc2.parameters(), 'lr': lr},
			]


def gcn_hybrid(state, t, pretrained=True, adj_file=None, in_channel=300, pre_model=''):
	if pre_model=='' or pre_model==None:
		model = models.resnet101(pretrained=pretrained)
	else:
		model = models.resnet101(pretrained=False)
		model.load_state_dict(torch.load(str(pre_model)))

	return GCNSDNEResnet(model=model, state=state, t=t, adj_file=adj_file, in_channel=in_channel)