# -*- coding=utf-8 -*-
import argparse
import pandas as pd
import cv2
import pretrainedmodels.utils as utils
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm, trange

import voc, coco, mirflickr25k, nuswide
from analysis import *
from inceptionv4 import inceptionv4
from util import *


IS_EXISTS_PERSON = False
imagenet_labels = (np.load('./imagenet_labels/ImageNetLabels.npy', allow_pickle=True).tolist())['imagenet_labels']


def par_option():
	parser = argparse.ArgumentParser(description='WILDCAT Training')
	parser.add_argument('data', metavar='DIR',
						help='path to dataset (e.g. data/')
	parser.add_argument('--image-size', '-i', default=224, type=int,
						metavar='N', help='image size (default: 224)')
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('--epochs', default=100, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
						help='number of epochs to change learning rate')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
						help='manual epoch number (useful on restarts)')
	parser.add_argument('-b', '--batch-size', default=8, type=int,
						metavar='N', help='mini-batch size (default: 256)')
	parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
						metavar='LR', help='initial learning rate')
	parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
						metavar='LR', help='learning rate for pre-trained layers')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--print-freq', '-p', default=0, type=int,
						metavar='N', help='print frequency (default: 10)')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
						help='evaluate model on validation set')
	parser.add_argument('--word2vec_file', type=str, default='data/mirflickr25k/mirflickr25k_glove_word2vec.pkl')
	parser.add_argument('--test_set_amount', type=int, default=1)
	parser.add_argument('--query_code_amount', type=int, default=496)
	parser.add_argument('--testset_pkl_path', type=str,
						default='./data/mirflickr25k/mirflickr25k_test_set.pkl')
	parser.add_argument("--query_pkl_path", type=str,
						default='./data/mirflickr25k/mirflickr25k_query_set.pkl')
	parser.add_argument("--hashcode_pool", type=str,
						default='./data/mirflickr25k/mirflickr25k_hashcode_pool.pkl')
	parser.add_argument("--hashcode_pool_limit", type=int, default=2400)
	
	parser.add_argument('--DROPOUT_RATIO', type=float, default=0.1)
	parser.add_argument('--CLASSIFIER_CHANNEL', type=str, default=2048)
	parser.add_argument('--IMAGE_CHANNEL', type=int,
						default=2048)
	parser.add_argument("--linear_intermediate", type=int,
						default=358)
	parser.add_argument('--pooling_stride', type=int, default=358)
	parser.add_argument("--threshold_p", type=float, default=0.15)
	parser.add_argument("--threshold_tao", type=float, default=0.4)
	parser.add_argument("--degree", type=int, default=2)
	parser.add_argument("--accumulate_steps", type=int, default=0)

	parser.add_argument('--img_instance', type=str, default='')
	parser.add_argument('--min_epoch_num', '--min_n', type=int, default=0)
	parser.add_argument('--max_epoch_num', '--max_n', type=int, default=500)
	parser.add_argument('--cos_threshold', '--c_t', type=float, default=0.06, help="the bigger the loose")

	parser.add_argument('-f', '--fusion', dest='FUSION', action='store_true')
	parser.add_argument('-d', '--display', dest='DISPLAY', action='store_true')
	
	return parser



def calc_dist(vector_a, vector_b, mode='cos'):
	sim = ''
	if mode == 'cos':
		vector_a = np.mat(vector_a)
		vector_b = np.mat(vector_b)
		num = float(vector_a * vector_b.T)
		denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
		sim = num / denom
	elif mode == 'manhattan':
		sim = abs(vector_a - vector_b)
		x = vector_a.size(0)
		sim = sim.sum()
	elif mode == "euclidean":
		sim = np.sqrt(sum((vector_a - vector_b).t() * (vector_a - vector_b)))
	return sim


def wbin_pkl(file_dir, content):
	'''write content into .pkl file with bin format'''
	with open(str(file_dir), 'ab') as fi:
		pickle.dump(content, fi)


def rbin_pkl(file_dir):
	'''read .pkl file and return the content'''
	with open(str(file_dir), 'rb') as fi:
		content = pickle.load(fi, encoding='bytes')
	return content


class Processing:
	
	def __init__(self, state={}, pretrain_path='./pretrained_models/inceptionv4-8e4777a0.pth',
				 imgpath="", mode='direct',
				 linear_weight=readpkl('./linearweight.pkl'),
				 layer_num=21,
				 noiseCsv='noiseLabelCsv'):
		'''
		
		:param pretrain_path: load pretrained model,here is `inceptionv4`
		:param imgpath:  load one image use the image-path
		:param mode: 	select mode, mode=direct means transfer `input` matrix into model, otherwise transfer a imgpath into model
		'''
		self.state = state
		self.device_ids = None
		self.model = inceptionv4(str(pretrain_path), pretrained='imagenet-me',
								 GP_flag=True)
		self.model = self.model.eval()
		self.input, self.softmax_out, self.GPout, self.lastconvlayerout = None, None, None, None
		self.imgpath = imgpath
		self.mode = mode
		self.conv_out = LayerActivations(self.model.features, int(layer_num))
		self.linearweight = linear_weight
		self.noiseLabelCsv = str(noiseCsv) + '.csv'
		self.gapPkl = str(noiseCsv) + "_gap.pkl"
		
		if torch.cuda.is_available():
			self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
	
	def preprocessInput_oldversion(self, ):
		load_img = utils.LoadImage()
		tf_img = utils.TransformImage(self.model)
		
		input_img = load_img(self.imgpath)
		self.size_upsample = input_img.size
		print("size_upsample = ", self.size_upsample)
		input_tensor = tf_img(input_img)
		input_tensor = input_tensor.unsqueeze(0)
		return torch.autograd.Variable(input_tensor, requires_grad=False)
	
	def preprocessInput(self, image_size=448, normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
													  std=[0.229, 0.224, 0.225]),):
		self.transform = transforms.Compose([
			MultiScaleCrop(int(image_size), scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		img = Image.open(self.imgpath).convert('RGB')
		np_img = np.array(img, dtype=np.float32)
		self.size_upsample = (np_img.shape[1],np_img.shape[0])
		print("upsample size = ", self.size_upsample)
		if self.transform is not None:
			img_tensor = self.transform(img)
		input_tensor = img_tensor.unsqueeze(0)
		return input_tensor
	
	def getGPout(self, input=None, ):
		'''
		get the output after GAP layer
		:param path_img: input img path
		:return: terminal softmax output, GAP-layer output and feature maps
 		'''
		if self.mode == 'direct':
			if input is None:
				assert False, "Input is wrong format..."
		else:
			if self.imgpath is not None or self.imgpath == '':
				input = self.preprocessInput()
			else:
				assert False, 'if use indirect mode, please fill the "imgpath"'
		self.input = input.clone()
		_, self.channel, self.height, self.width = input.size()
		z, GPout = self.model(input)
		output_softmax = F.softmax(z, dim=1)
		self.softmax_out, self.GPout = output_softmax, GPout
		self.conv_out.remove()
		self.lastconvlayerout = self.conv_out.features
		return self.softmax_out, self.GPout, self.lastconvlayerout
	
	def fusionheatmap(self, class_seq, display=False):
		'''
		fusion the original map and heat map
		:param class_seq:the sequence number which corresponding to the
		 interesting probability in the softmax output
		:return:save the heat map
		'''
		if self.lastconvlayerout is not None:
			bz, nc, h, w = self.lastconvlayerout.shape
			imgname = self.imgpath.split('/')[-1]
			if '.' in imgname:
				pure = imgname.split('.')[0]
				imgname = pure
			target_lineweight = self.linearweight[class_seq, ...]
			cam = torch.matmul(target_lineweight, (self.lastconvlayerout.reshape((nc, h * w)))).detach().numpy()
			cam = cam.reshape(h, w)
			cam -= np.min(cam)
			cam_img = cam / np.max(cam)
			cam_img = np.uint8(255 * cam_img)
			if self.mode == 'indirect':
				cam_img = cv2.resize(cam_img,
									 self.size_upsample)
			elif self.mode == 'direct':
				cam_img = cv2.resize(cam_img, (self.height, self.width))
			else:
				assert False, 'mode type error!'
			if display:
				print("every pixel in cam_img:")
				for i in range(cam_img.shape[0]):
					for j in range(cam_img.shape[1]):
						print(cam_img[i, j], (i, j))
					print()
				cv2.imwrite("./testgray.jpg", cam_img)
			
			heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
			if self.imgpath != '' and self.imgpath is not None:
				original_img = cv2.imread(self.imgpath)
			else:
				original_img = self.input.squeeze(0).permute(1, 2, 0)
			array_original_img = ''
			if torch.is_tensor(original_img):
				array_original_img = original_img.cpu().numpy() if torch.cuda.is_available() else original_img.numpy()
			else:
				array_original_img = original_img
			result = heatmap * 0.5 + array_original_img * 0.5
			if not os.path.exists('./' + imgname): os.mkdir('./' + imgname)
			cv2.imwrite('./' + imgname + '/' + imgname + '_' + str(class_seq) + '.jpg', result)
			cv2.imwrite('./' + imgname + '/' + imgname + '_heatmap_' + str(class_seq) + '.jpg', heatmap)
			return heatmap, imgname, str(class_seq)
	
	def extractLayerParams(self, layername):
		'''
		Get the given layer parameters value (e.g. weight or bias)
		:param layername:	layer name
		:return:	the value, usually a Tensor/Matrix
		'''
		if str(layername) in self.model.state_dict():
			return self.model.state_dict()[str(layername)]
		else:
			print("Please Check the 'layer_name'")
	
	def storeprobab(self, namelist=[], display=False, fusion=False):
		NL = {}
		
		for x in range(int(self.softmax_out.shape[0])):
			if display == False:
				if len(namelist) and self.mode == 'direct':
					print("now img name :", namelist[x])
				else:
					print("now img name = ", self.imgpath)
			out_array = self.softmax_out[x, :].detach().cpu().numpy()
			out_list = out_array.tolist()
			above_milli = {}

			max_item = max(out_list)
			for i in range(len(out_list)):
				item = float(out_list[i])
				if item > 0.001 and float(max_item) / item < 30.0:
					above_milli[str(i)] = item
			above_milli = sorted(above_milli.items(), key=lambda item: item[1], reverse=True)
			if display:
				for j in above_milli:
					print(imagenet_labels[str(j[0])], "\t", j[1])
				
				print("\n********************************************************************\n")
			last_result = self.overlap(above_milli, self.linearweight)
			if fusion:
				print('last_result = ', last_result)
			name_label = [0 for x in range(len(imagenet_labels))]
			for i in range(len(last_result)):
				seq = last_result[i][0]
				name_label[int(seq)] = 1
				if fusion:
					self.fusionheatmap(class_seq=int(seq))
			if display:
				print("name_label = ", name_label)
			if namelist:
				NL[str(namelist[x])] = name_label
			
			if os.path.exists(str(self.noiseLabelCsv)):
				pass
		
		return NL
	
	def overlap(self, abovemilli_dict, fcweight=None, display=False):
		'''
		search overlap heatmap region, for the abovemilli_dict generated based on one image
		:param abovemilli_dict:	the probability which above milli
		:param fcweight:		the last linear weight
		:return: the labels which are filtrated
		'''
		if fcweight is None:
			fcweight = readpkl('linearweight.pkl')
		
		cos_threshold = self.state['cos_threshold']
		man_threshold = 0.01
		omit_index = []
		for i in range(len(abovemilli_dict)):
			if i in omit_index: continue
			item = abovemilli_dict[int(i)]
			label_seq = item[0]
			prob = item[1]
			cosdist, mandist, eucdist = [], [], []
			tmp_omit = []
			for j in range(i + 1, len(abovemilli_dict)):
				contrast_item = abovemilli_dict[int(j)]
				contrast_labelseq = contrast_item[0]
				contrast_prob = contrast_item[1]
				
				cos_dist = calc_dist(fcweight[int(label_seq)], fcweight[int(contrast_labelseq)], mode='cos')
				man_dist = calc_dist(fcweight[int(label_seq)], fcweight[int(contrast_labelseq)], mode='manhattan')
				euc_dist = calc_dist(fcweight[int(label_seq)], fcweight[int(contrast_labelseq)], mode='euclidean')
				
				if display:
					cosdist.append(cos_dist)
					mandist.append(man_dist)
					eucdist.append(euc_dist)
					print("label1={0}:probab1={1},\nlabel2={2}:probab2={3}". \
						  format(imagenet_labels[label_seq], prob, imagenet_labels[contrast_labelseq], contrast_prob))
					print("cos_dist = ", cos_dist)
					print('manhattan_dist = ', man_dist)
					print("euclidean dist = ", euc_dist)
					print("\n")
				
				if cos_dist >= cos_threshold:
					tmp_omit.append(imagenet_labels[str(contrast_labelseq)])
					omit_index.append(int(j))
					continue
			
			if display:
				print('at the {0}:\nneed be omitted :{1},\ncos={2},\nman={3},\neuc={4}\n'.
					  format(imagenet_labels[label_seq], tmp_omit, cosdist, mandist, eucdist))
				
				print("finalout", '>' * 50, '\n', )
		tmp = []
		for ii in range(len(abovemilli_dict)):
			if ii not in omit_index:
				tmp.append(abovemilli_dict[int(ii)])
				seq = abovemilli_dict[int(ii)][0]
				if display:
					print(seq, end=': ')
					print(imagenet_labels[seq])
		
		if display: print("tmp = ", tmp, "\n", "<" * 50, 'finalout\n')
		return tmp
	
	def clustercenter(self, abovemilli_dict, center_num=3, fcweight=None):
		'''
		use k-means algorthm
		:param abovemilli_dict:
		:param fcweight:
		:return:
		'''
		if fcweight is None:
			fcweight = readpkl('linearweight.pkl')
		
		fc_vectors = np.array(fcweight[int(abovemilli_dict[0][0])]).reshape(1, -1)
		for i in range(1, len(abovemilli_dict)):
			fc_vectors = np.row_stack((fc_vectors, fcweight[int(abovemilli_dict[i][0])]))
		print("fc_vectors.shape = ", fc_vectors.shape, type(fc_vectors))
		fc_vectors.dtype = np.float64
		
		model_pca = PCA(n_components=10)
		fc_vectors_pca = model_pca.fit(fc_vectors).transform(fc_vectors)
		
		print("fc_pca.shape = ", fc_vectors_pca.shape)
		
		rice_cluster = KMeans(n_clusters=5)
		rice_cluster.fit(fc_vectors_pca)
		label = rice_cluster.labels_
		print("labels = ", label, len(label))
		
		omit_label = []
		result = []
		for i in range(len(label)):
			item = label[i]
			if item in omit_label:
				continue
			else:
				result.append(i)
				omit_label.append(item)
				print(imagenet_labels[str(abovemilli_dict[i][0])])


def calc_IOU(map1, map2):
	contrast_threshold = 230
	print("map1_shape = {0}, map2_shape = {1}".format(map1.shape, map2.shape))
	
	map1_count, map2_count = 0, 0
	interact_count = 0
	
	for i in range(map1.shape[0]):
		for j in range(map2.shape[1]):
			pass


class ProcessData:
	'''
	get voc, coco, mirflckr25k, nuswide dataset ,
	'''
	
	def __init__(self, normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
													  std=[0.229, 0.224, 0.225]),
				 batchsize=8, image_size=448, workers=4):
		self.train_transform = transforms.Compose([
			MultiScaleCrop(int(image_size), scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		self.test_transform = transforms.Compose([
			Warp(int(image_size)),
			transforms.ToTensor(),
			normalize,
		])
		self.bs = int(batchsize)
		self.workers = int(workers)
	
	def get_voc(self, ):
		voc_train = voc.Voc2007Classification('../Amatrix_ML/data/voc', 'trainval', )
		voc_test = voc.Voc2007Classification('../Amatrix_ML/data/voc', 'test', )
		voc_train.transform = self.test_transform
		voc_test.transform = self.test_transform
		self.voc_train_loader = torch.utils.data.DataLoader(voc_train,
															batch_size=self.bs, shuffle=True,
															num_workers=self.workers)
		self.voc_val_loader = torch.utils.data.DataLoader(voc_test,
														  batch_size=self.bs, shuffle=False,
														  num_workers=self.workers)
		
		if torch.cuda.is_available():
			self.voc_train_loader.pin_memory = True
			self.voc_val_loader.pin_memory = True


class NoiseGeneration:
	def __init__(self, state={}):
		self.state = state
		if self._state("batch_size") is None:
			self.state['batch_size'] = 8
		if self._state('image_size') is None:
			self.state['image_size'] = 448
		if self._state('data') is None:
			self.state['data'] = './data/voc'
		if self._state('workers') is None:
			self.state['workers'] = 4
		if self._state("inception_pretrained_model") is None:
			self.state["inception_pretrained_model"] = './pretrained_models/inceptionv4-8e4777a0.pth'
		
		self.state['transform'] = None
	
	def _state(self, name):
		if str(name) in self.state.keys():
			return self.state[str(name)]
	
	def init_generation(self, ):
		'''
		define the transform module
		:return:
		'''
		if self._state('transform') is None:
			normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
											 std=[0.229, 0.224, 0.225])
			
			self.state['transform'] = transforms.Compose([
				Warp(self.state['image_size']),
				transforms.ToTensor(),
				normalize,
			])
	
	def get_voc(self, ):
		voc_trainset = voc.Voc2007Classification(self.state['data'], 'trainval', )
		voc_testset = voc.Voc2007Classification(self.state['data'], 'test', )
		voc_trainset.transform = self.state['transform']
		voc_testset.transform = self.state['transform']
		voc_train_loader = torch.utils.data.DataLoader(voc_trainset,
													   batch_size=self.state['batch_size'], shuffle=False,
													   num_workers=self.state['workers'])
		voc_val_loader = torch.utils.data.DataLoader(voc_testset,
													 batch_size=self.state['batch_size'], shuffle=False,
													 num_workers=self.state['workers'])
		
		if torch.cuda.is_available():
			voc_train_loader.pin_memory = True
			voc_val_loader.pin_memory = True
		
		return voc_train_loader, voc_val_loader
	
	def get_coco(self, ):
		coco_trainset = coco.COCO2014(self.state['data'], phase='train')
		coco_testset = coco.COCO2014(self.state['data'], phase='val')
		coco_trainset.transform = self.state['transform']
		coco_testset.transform = self.state['transform']
		coco_train_loader = torch.utils.data.DataLoader(coco_trainset,
														batch_size=self.state['batch_size'], shuffle=False,
														num_workers=self.state['workers'])
		coco_val_loader = torch.utils.data.DataLoader(coco_testset,
													  batch_size=self.state['batch_size'], shuffle=False,
													  num_workers=self.state['workers'])
		
		if torch.cuda.is_available():
			coco_train_loader.pin_memory = True
			coco_val_loader.pin_memory = True
		
		return coco_train_loader, coco_val_loader
	
	def get_mirflickr25k(self, ):
		mirflickr25k_trainset = mirflickr25k.MirFlickr25kPreProcessing(self.state['data'], 'train', )
		mirflickr25k_valset = mirflickr25k.MirFlickr25kPreProcessing(self.state['data'], 'test', )
		mirflickr25k_trainset.transform = self.state['transform']
		mirflickr25k_valset.transform = self.state['transform']
		mirflickr25k_train_loader = torch.utils.data.DataLoader(mirflickr25k_trainset,
																batch_size=self.state['batch_size'], shuffle=False,
																num_workers=self.state['workers'])
		mirflickr25k_val_loader = torch.utils.data.DataLoader(mirflickr25k_valset,
															  batch_size=self.state['batch_size'], shuffle=False,
															  num_workers=self.state['workers'])
		
		if torch.cuda.is_available():
			mirflickr25k_train_loader.pin_memory = True
			mirflickr25k_val_loader.pin_memory = True
		
		return mirflickr25k_train_loader, mirflickr25k_val_loader
	
	def get_nuswide(self, ):
		nuswide_trainset = nuswide.NuswideClassification(self.state['data'], 'train')
		nuswide_valset = nuswide.NuswideClassification(self.state['data'], 'test')
		nuswide_trainset.transform = self.state['transform']
		nuswide_valset.transform = self.state['transform']
		nuswide_train_loader = torch.utils.data.DataLoader(nuswide_trainset,
														   batch_size=self.state['batch_size'], shuffle=False,
														   num_workers=self.state['workers'])
		nuswide_val_loader = torch.utils.data.DataLoader(nuswide_valset,
														 batch_size=self.state['batch_size'], shuffle=False,
														 num_workers=self.state['workers'])
		
		if torch.cuda.is_available():
			nuswide_train_loader.pin_memory = True
			nuswide_val_loader.pin_memory = True
		
		return nuswide_train_loader, nuswide_val_loader
	
	def inception_validate(self, imgpath='', display=False, fusion=False):
		'''
		use inception structure , pretrained model on ImageNet dataset, forward propagation
		:return:
		'''
		if imgpath == '' or imgpath == None:
			if 'voc' in self.state['data']:
				self.train_loader, self.val_loader = self.get_voc()
				noise_Csv = "VOC_inceptionv4"
			elif 'coco' in self.state['data']:
				self.train_loader, self.val_loader = self.get_coco()
				noise_Csv = "COCO_inceptionv4"
			elif 'mirflickr25k' in self.state['data']:
				self.train_loader, self.val_loader = self.get_mirflickr25k()
				noise_Csv = "MIRFLICKR25K_inceptionv4"
			elif 'nuswide' in self.state['data']:
				self.train_loader, self.val_loader = self.get_nuswide()
				noise_Csv = "NUSWIDE_inceptionv4"
			else:
				assert False, "***Cannot find such a dataset...***"
			inceptionv4_model = Processing(state=self.state, pretrain_path=self.state['inception_pretrained_model'],
										   noiseCsv=noise_Csv)
			for i, (input, target) in enumerate(self.train_loader):
				print("the {0}-th batch".format(int(i)))
				if i < self.state['min_epoch_num']:
					print("skip..")
					continue
				if i >= self.state['max_epoch_num']:
					print("halt...")
					break
				true_name = []
				for i in range(target.shape[0]):
					if '.' in input[1][i]:
						img_name, rear = str(input[1][i]).split('.')[0], "." + str(input[1][i]).split('.')[1]
					else:
						img_name, rear = str(input[1][i]), ''
					if 'voc' in self.state['data']:
						if target[i][14] == -1:
							true_name.append(img_name + rear)
						else:
							true_name.append(img_name + "_p" + rear)
					elif 'mirflickr' in self.state['data']:
						if target[i][13] == 1:
							true_name.append(img_name + "_p" + rear)
						else:
							true_name.append(img_name + rear)
					elif 'coco' in self.state['data']:
						if target[i][49] == 1:
							true_name.append(img_name + "_p" + rear)
						else:
							true_name.append(img_name + rear)
					elif 'nuswide' in self.state['data']:
						if target[i][42] == 1:
							true_name.append(img_name + "_p" + rear)
						else:
							true_name.append(img_name + rear)
				out1, gapout, lastconvout = inceptionv4_model.getGPout(input=input[0], )
				tmp_dict = inceptionv4_model.storeprobab(namelist=true_name, display=display)
				tmp_df = pd.DataFrame.from_dict(tmp_dict, orient='index', )
				print()
				if os.path.exists(inceptionv4_model.noiseLabelCsv) is None:
					tmp_df.to_csv(inceptionv4_model.noiseLabelCsv, header=True)
				else:
					tmp_df.to_csv(inceptionv4_model.noiseLabelCsv, mode='a', header=None, )

				content = [true_name, gapout]
				write2pkl(content, inceptionv4_model.gapPkl)
		else:
			print("here in the else branch")
			testobj = Processing(state=self.state, pretrain_path=self.state['inception_pretrained_model'],
								 imgpath=imgpath, mode='indirect')
			out, GPout, lastconvout = testobj.getGPout()
			print("out.shape, lastconvout.shape = ", out.shape, lastconvout.shape)
			img_name = imgpath.split('/')[-1]
			if '.' in img_name:
				pure_name = img_name.split('.')[0]
				img_name = pure_name
			if not os.path.exists('./' + img_name): os.mkdir('./' + img_name)
			write2pkl(out, "./" + img_name + "/" + img_name + '_probability.pkl')
			write2pkl(lastconvout, "./" + img_name + "/" + img_name + '_feature_map.pkl')
			testobj.storeprobab(display=display, fusion=fusion)


def main_generation():
	parser = par_option()
	args = parser.parse_args()
	state = {'batch_size': args.batch_size,
			 "image_size": 448,
			 'data': args.data,
			 'img_path': args.img_instance,
			 'max_epoch_num': args.max_epoch_num,
			 'min_epoch_num': args.min_epoch_num,
			 'cos_threshold': args.cos_threshold,
			 'fusion': args.FUSION,
			 'display': args.DISPLAY,
			 }
	mainobj = NoiseGeneration(state=state, )
	mainobj.init_generation()
	mainobj.inception_validate(imgpath=state['img_path'], fusion=state['fusion'], display=state['display'])


def old_main_version():
	path_img = "../Amatrix_ML/data/voc/VOCdevkit/VOC2007/JPEGImages/009397.jpg"
	testobj = Processing(imgpath=path_img, mode='indirect')
	out, GPout, lastconvout = testobj.getGPout()
	print("out = ", out.shape, '\n', out)
	print('GPout = ', GPout.shape)
	
	testobj.storeprobab()
	
	outobj = ProcessData()
	outobj.get_voc()
	
	for i, (input, target) in enumerate(outobj.voc_train_loader):
		true_name = []
		if torch.cuda.is_available():
			postprocessinput = torch.FloatTensor(torch.FloatStorage()).cuda()
		else:
			postprocessinput = torch.FloatTensor(torch.FloatStorage())
		for i in range(target.shape[0]):
			if target[i][14] == -1:
				true_name.append(input[1][i])
				postprocessinput = torch.cat((postprocessinput, input[0][i].unsqueeze(0).cuda()), 0) \
					if torch.cuda.is_available() else torch.cat((postprocessinput, input[0][i].unsqueeze(0)), 0)
		out1, out2, lastconvout = testobj.getGPout(input=postprocessinput, )
		testobj.storeprobab(namelist=true_name, )


if __name__ == "__main__":
	main_generation()
