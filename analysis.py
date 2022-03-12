# -*- coding=utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import sys, os, pickle



## wirte
def write2pkl( content, pkl_path):
	'''write content into .pkl file with bin format'''
	with open(str(pkl_path), 'ab') as fi:
		pickle.dump(content, fi)


def readpkl( pkl_path):
	'''read .pkl file and return the content'''
	with open(str(pkl_path), 'rb') as fi:
		content = pickle.load(fi, encoding='bytes')
	return content

class LayerActivations:
	features = None
	
	def __init__(self, model, layer_num):
		self.hook = model[layer_num].register_forward_hook(self.hook_fn)
	
	def hook_fn(self, module, input, output):
		self.features = output.cpu()
	
	def remove(self):
		self.hook.remove()


class NetDissection:
	def __init__(self, model):
		'''
		:param model: input a model
		'''
		self.model = model
		
	def displayStructure(self,):
		'''
		display the overall Net structure
		:return:
		'''
		print("Display the model's structure:")
		print(self.model)
		print('\n')
		
	def displayEveryLayerInfo(self, displayValue=False):
		'''
		display the name and value of every layer.
		:param displayValue: whether display the value
		:return:
		'''
		print("Print every layer Info:")
		for k, v in self.model.state_dict().items():
			print("Layer name: {0}".format(k))
			if displayValue:
				print("The Parameters value is:\n{0}".format( v))
				print('Its shape: ', v.shape)
				
	def extractLayerValue(self, layername):
		'''
		Get the given layer parameters value (e.g. weight or bias)
		:param layername:	layer name
		:return:	the value, usually a Tensor/Matrix
		'''
		if str(layername) in self.model.state_dict():
			return self.model.state_dict()[str(layername)]
		else:
			print("Please Check the 'layer_name'")
			
	def GetInterlayerOutput(self, input, layer_num):
		'''
		Get a interlayer output value with given layer num
		:param input:	the input of Net
		:param layer_num:	the num of layer which interested
		:return:	interlayer output
		'''
		pass
		conv_out = LayerActivations(self.model.features, int(layer_num))
		self.model(input)
		conv_out.remove()
		act = conv_out.features
		return act
