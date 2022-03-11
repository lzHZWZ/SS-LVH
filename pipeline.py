import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
import gl
from models_lib.model_gcn import *
from models_lib.model_sdne import *

DEBUG_MODEL = False
IS_NET_CONNECTION = True


class SSDResnet(nn.Module):
	def __init__(self, option, model, num_classes, in_channel=300, t=0, adj_file=None, cor_adj=None):
		super(SSDResnet, self).__init__()
		self.image_normalization_mean = [0.485, 0.456, 0.406]
		self.image_normalization_std = [0.229, 0.224, 0.225]
		
		self.state = {}
		self.state['use_gpu'] = torch.cuda.is_available()
		self.opt = option
		self.is_hash = option.HASH_TASK
		self.is_usemfb = option.IS_USE_MFB
		self.pooling_stride = option.pooling_stride
		self.img_channel = option.IMAGE_CHANNEL
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
		self.relu1 = nn.LeakyReLU(0.2)
		self.gc3 = GraphConvolution(1536, 4096)
		self.gc4 = GraphConvolution(4096, 2048)
		self.relu2 = nn.LeakyReLU(0.2)
		
		_adj = gen_adj(None, adj_file=adj_file)
		self.A1 = Parameter(_adj.float())
		self.A2 = Parameter(gen_adj(None, adj_file=cor_adj))
		
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
			self.hash_layer = nn.Linear(int(2 * self.num_classes * (self.img_channel - self.pooling_stride + 1)),
										option.HASH_BIT)
			self.use_tanh = nn.Tanh()
	
	def forward(self, feature, inp, cor):
		feature = self.features(feature)
		feature = self.pooling(feature)
		feature = feature.view(feature.size(0), -1)
		
		inp = inp[0]
		label_adj = self.A1.detach()
		x_local = self.gc1(inp, label_adj)
		x_local = self.relu1(x_local)
		x_local = self.gc2(x_local, label_adj)
		
		cor = cor[0]
		global_adj = self.A2.detach()
		x_global = self.gc3(cor, global_adj)
		x_global = self.relu2(x_global)
		x_global = self.gc4(x_global, global_adj)
		
		x_local = torch.transpose(x_local, 0, 1)
		x_global = torch.transpose(x_global, 0, 1)
		if self.state['use_gpu']:
			x_out1 = torch.FloatTensor(torch.FloatStorage()).cuda()
			x_out2 = torch.FloatTensor(torch.FloatStorage()).cuda()
		else:
			x_out1 = torch.FloatTensor(torch.FloatStorage())
			x_out2 = torch.FloatTensor(torch.FloatStorage())
		transfer_mat = torch.zeros([self.img_channel, self.img_channel - self.pooling_stride + 1]).float().cuda() \
			if torch.cuda.is_available() else torch.zeros([self.img_channel, self.img_channel - self.pooling_stride + 1]).float()
		for i in range(transfer_mat.shape[1]):
			for kk in range(self.pooling_stride):
				if i + kk <= transfer_mat.shape[0]:
					transfer_mat[i + kk][i] = 1.0
		for i_row in range(int(feature.shape[0])):
			img_linear_row = feature[i_row, :].view(1, -1)
			out_row1 = torch.mul(img_linear_row, torch.transpose(x_local, 0, 1))
			out_row2 = torch.mul(img_linear_row, torch.transpose(x_global, 0, 1))
			overlap_sum1 = torch.matmul(out_row1, transfer_mat).view(1, -1)
			overlap_sum2 = torch.matmul(out_row2, transfer_mat).view(1, -1)
			x_out1 = torch.cat((x_out1, overlap_sum1), 0)
			x_out2 = torch.cat((x_out2, overlap_sum2), 0)
		x_out = torch.cat((x_out1, x_out2), dim=1)
		
		gl.GLOBAL_TENSOR1 = x_out
		
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
				{'params': self.gc3.parameters(), 'lr': lr},
				{'params': self.gc4.parameters(), 'lr': lr},
			]


def gcn_resnet101(opt, num_classes, t, pretrained=True, adj_file=None, cor_file=None, in_channel=300):
	if IS_NET_CONNECTION:
		model = models.resnet101(pretrained=pretrained)
	else:
		model = models.resnet101(pretrained=False)
		model.load_state_dict(torch.load('./checkpoint/pretrained_resnet101/resnet101-5d3b4d8f.pth'))

	return SSDResnet(opt, model, num_classes, t=t, adj_file=adj_file, cor_adj=cor_file, in_channel=in_channel)
