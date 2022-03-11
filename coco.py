import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *
import random

urls = {'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
		'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
		'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

labels = ["airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove", "bear", "bed",
		  "bench", "bicycle", "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus", "cake",
		  "car", "carrot", "cat", "cell phone", "chair", "clock", "couch", "cow", "cup", "dining table",
		  "dog", "donut", "elephant", "fire hydrant", "fork", "frisbee", "giraffe", "hair drier",
		  "handbag", "horse", "hot dog", "keyboard", "kite", "knife", "laptop", "microwave", "motorcycle",
		  "mouse", "orange", "oven", "parking meter", "person", "pizza", "potted plant", "refrigerator",
		  "remote", "sandwich", "scissors", "sheep", "sink", "skateboard", "skis", "snowboard", "spoon",
		  "sports ball", "stop sign", "suitcase", "surfboard", "teddy bear", "tennis racket", "tie",
		  "toaster", "toilet", "toothbrush", "traffic light", "train", "truck", "tv", "umbrella", "vase",
		  "wine glass", "zebra"]


def download_coco2014(root, phase):
	if not os.path.exists(root):
		os.makedirs(root)
	tmpdir = os.path.join(root, 'tmp/')  
	data = os.path.join(root, 'data/')  
	if not os.path.exists(data):
		os.makedirs(data)
	if not os.path.exists(tmpdir):
		os.makedirs(tmpdir)
	if phase == 'train':
		filename = 'train2014.zip'
	elif phase == 'val':
		filename = 'val2014.zip'
	cached_file = os.path.join(tmpdir, filename)
	if not os.path.exists(cached_file):
		print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
		os.chdir(tmpdir)
		subprocess.call('wget ' + urls[phase + '_img'], shell=True)
		os.chdir(root)
	img_data = os.path.join(data, filename.split('.')[0])
	if not os.path.exists(img_data):
		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
		command = 'unzip {} -d {}'.format(cached_file, data)
		os.system(command)
	print('[dataset] Done!')
	
	cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
	if not os.path.exists(cached_file):
		print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
		os.chdir(tmpdir)
		subprocess.Popen('wget ' + urls['annotations'], shell=True)
		os.chdir(root)
	annotations_data = os.path.join(data, 'annotations')
	if not os.path.exists(annotations_data):
		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
		command = 'unzip {} -d {}'.format(cached_file, data)
		os.system(command)
	print('[annotation] Done!')
	
	anno = os.path.join(data, '{}_anno.json'.format(phase))
	img_id = {}
	annotations_id = {}
	if not os.path.exists(anno):
		annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
		annotations = annotations_file['annotations']
		category = annotations_file['categories']
		category_id = {}
		for cat in category:
			category_id[cat['id']] = cat['name']
		cat2idx = categoty_to_idx(sorted(category_id.values()))
		images = annotations_file['images']
		for annotation in annotations:
			if annotation['image_id'] not in annotations_id:
				annotations_id[annotation['image_id']] = set()
			annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
		for img in images:
			if img['id'] not in annotations_id:
				continue
			if img['id'] not in img_id:
				img_id[img['id']] = {}
			img_id[img['id']]['file_name'] = img['file_name']
			img_id[img['id']]['labels'] = list(annotations_id[img['id']])
		anno_list = []
		for k, v in img_id.items():
			anno_list.append(v)
		json.dump(anno_list, open(anno, 'w'))
		if not os.path.exists(os.path.join(data, 'category.json')):
			json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
		del img_id
		del anno_list
		del images
		del annotations_id
		del annotations
		del category
		del category_id
	print('[json] Done!')


def categoty_to_idx(category):
	cat2idx = {}
	for cat in category:
		cat2idx[cat] = len(cat2idx)
	return cat2idx


class COCO2014(data.Dataset):
	def __init__(self, root, transform=None, phase='train',
				 inp_name=None, cor_name=None, f_p=-1):  
		print("load {0} file\n".format(inp_name))
		self.root = root
		self.phase = phase
		self.img_list = []
		self.transform = transform
		download_coco2014(root, phase)
		self.get_anno()
		self.num_classes = len(self.cat2idx)
		self.f_p = f_p
		
		if inp_name:
			with open(inp_name, 'rb') as f:
				self.inp = pickle.load(f) 
		if cor_name:
			with open(cor_name, 'rb') as f:
				self.cor = pickle.load(f)
		self.inp_name = inp_name
		self.cor_name = cor_name
		print("self.cor_name = ", self.cor_name)
	
	def get_anno(self):
		list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
		self.img_list = json.load(open(list_path, 'r'))
		self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))
	
	def __len__(self):
		return len(self.img_list)
	
	def __getitem__(self, index):
		'''
		the main purpose of __getitem_() is to make this object have iterable
		:param index:
		:return:
		'''
		if self.f_p > 0 and self.f_p <= 1:
			x = random.randint(0, 10) // 10
			if x <= self.f_p / 2.0:
				item1 = self.img_list[index]
				offset = random.randint(0, 100)
				if index + offset <= len(self.img_list):
					item2 = self.img_list[index + offset]
				else:
					item2 = self.img_list[index - offset]
				return self.blend_two_images(item1, item2, self.f_p)
			else:
				item1 = self.img_list[index]
				offset = random.randint(0, 100)
				if index + offset <= len(self.img_list):
					item2 = self.img_list[index + offset]
				else:
					item2 = self.img_list[index - offset]
				return self.cat_two_images(item1, item2)
		else:
			item = self.img_list[index]
			return self.get(item)
	
	def get(self, item, ):
		filename = item['file_name']
		labels = sorted(item['labels'])
		img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		target = np.zeros(self.num_classes, np.float32) - 1
		target[labels] = 1
		
		if self.inp_name:
			if self.cor_name:
				return (img, filename, self.inp, self.cor), target
			else:
				return (img, filename, self.inp), target
		else:
			return (img, filename), target
	
	def blend_two_images(self, item1, item2, p=0.5):
		filename1, filename2 = item1['file_name'], item2['file_name']
		labels1, labels2 = sorted(item1['labels']), sorted(item2['labels'])
		img1 = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename1)).convert('RGB')
		img2 = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename2)).convert('RGB')
		w1, h1 = img1.size
		w2, h2 = img2.size
		img1 = img1.resize((min(w1, w2), min(h1, h2)))
		img2 = img2.resize((min(w1, w2), min(h1, h2)))
		
		img = Image.blend(img1, img2, p)
		if self.transform is not None:
			img = self.transform(img)
		target = np.zeros(self.num_classes, np.float32) - 1
		target[labels1] = 1
		target[labels2] = 1
		
		if self.inp_name:
			if self.cor_name:
				return (img, filename1.split('.')[0] + filename2.split('.')[0] + "_blend.jpg", self.inp,
						self.cor), target
			else:
				return (img, filename1.split('.')[0] + filename2.split('.')[0] + "_blend.jpg", self.inp), target
		else:
			return (img, filename1.split('.')[0] + filename2.split('.')[0] + "_blend.jpg"), target
	
	def cat_two_images(self, item1, item2):
		filename1, filename2 = item1['file_name'], item2['file_name']
		labels1, labels2 = sorted(item1['labels']), sorted(item2['labels'])
		img1 = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename1)).convert('RGB')
		img2 = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename2)).convert('RGB')
		w1, h1 = img1.size
		w2, h2 = img2.size
		img1 = img1.resize((min(w1, w2), min(h1, h2)))
		img2 = img2.resize((min(w1, w2), min(h1, h2)))
		img = Image.new(img1.mode, (min(w1, w2) * 2, min(h1, h2)))
		img.paste(img1, box=(0, 0))
		img.paste(img2, box=(1 * min(w1, w2), 0))
		if self.transform is not None:
			img = self.transform(img)
		target = np.zeros(self.num_classes, np.float32) - 1
		target[labels1] = 1
		target[labels2] = 1
		
		if self.inp_name:
			if self.cor_name:
				return (img, filename1.split('.')[0] + filename2.split('.')[0] + "_cat.jpg", self.inp,
						self.cor), target
			else:
				return (img, filename1.split('.')[0] + filename2.split('.')[0] + "_cat.jpg", self.inp), target
		else:
			return (img, filename1.split('.')[0] + filename2.split('.')[0] + "_cat.jpg"), target
