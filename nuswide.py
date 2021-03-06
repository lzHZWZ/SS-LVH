# -*- coding=utf-8 -*-

import csv
import os, sys
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

object_categories = ['airport','animal','beach','bear','birds','boats','book','bridge','buildings',
					'cars','castle','cat','cityscape','clouds','computer','coral','cow','dancing',
					'dog','earthquake','elk','fire','fish','flags','flowers','food','fox','frost',
					'garden','glacier','grass','harbor','horses','house','lake','leaf','map','military',
					'moon','mountain','nighttime','ocean','person','plane','plants','police','protest',
					'railroad','rainbow','reflection','road','rocks','running','sand','sign','sky',
					'snow','soccer','sports','statue','street','sun','sunset','surf','swimmers','tattoo',
					'temple','tiger','tower','town','toy','train','tree','valley','vehicle','water',
					'waterfall','wedding','whales','window','zebra',
                     ]

urls = {
	# 'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
	# 'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
	# 'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
	# 'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


count = 0
def walkFile(file):
	for root, dirs, files in os.walk(file):

		for f in files:
			global count
			if '.jpg' in f and '.mat' not in f:
				count += 1
				print(os.path.join(root, f))

		for d in dirs:
			print(os.path.join(root, d))
	print("the overall amount of the files is :", count)

def read_image_name(file):
	print('[dataset] read the image name' + file)
	namelist = []
	imgpathlist = []
	name_imgpath_dic = {}
	with open(file, 'r') as f:
		for line in f:
			tmp = line[:-1].split('\\')
			imgpath = '/'.join(tmp)
			name = tmp[-1].split('.')[0]
			namelist.append(name)
			imgpathlist.append(imgpath)
			name_imgpath_dic[str(name)] = imgpath
	return name_imgpath_dic, namelist, imgpathlist

def read_image_label(file):
	print('[dataset] read ' + file)
	data = []
	with open(file, 'r') as f:
		for line in f:
			tmp = line[:-1]
			data.append(tmp[-1])
	return data


def read_object_labels(root, dataset, set):
	path_labels = os.path.join(root,'data', 'Groundtruth','TrainTestLabels')
	path_imglist = os.path.join(root, 'data','ImageList')
	labeled_data = dict()
	num_classes = len(object_categories)

	set = 'T' + (str.lower(set))[1:]

	imglist_file = os.path.join(path_imglist, set+'Imagelist.txt')
	name_img_dic, imgnamelist, imgpathlist = read_image_name(imglist_file)

	for i in range(num_classes):
		data={}
		file = os.path.join(path_labels, 'Labels_' + object_categories[i] + '_' + set + '.txt')
		data_x = read_image_label(file)


		for x in range(len(data_x)):
			data[str(imgpathlist[x])] = data_x[x]

		if i == 0:
			for (name, label) in data.items():
				labels = np.zeros(num_classes)
				labels[i] = label
				labeled_data[name] = labels
		else:
			for (name, label) in data.items():
				labeled_data[name][i] = label

	return labeled_data


def write_object_labels_csv(file, labeled_data):
	print('[dataset] write file %s' % file)
	with open(file, 'w') as csvfile:
		fieldnames = ['name']
		fieldnames.extend(object_categories)
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		for (name, labels) in labeled_data.items():
			example = {'name': name}
			for i in range(len(object_categories)):
				example[fieldnames[i + 1]] = int(labels[i])
			writer.writerow(example)

	csvfile.close()


def read_object_labels_csv(file, header=True):
	images = []
	num_categories = 0
	print('[dataset] read', file)
	with open(file, 'r') as f:
		reader = csv.reader(f)
		rownum = 0
		for row in reader:
			if header and rownum == 0:
				header = row
			else:
				if num_categories == 0:
					num_categories = len(row) - 1
				name = row[0]
				labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
				labels = torch.from_numpy(labels)
				item = (name, labels)
				images.append(item)
			rownum += 1
	return images


def find_images_classification(root, dataset, set):
	path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
	images = []
	file = os.path.join(path_labels, set + '.txt')
	with open(file, 'r') as f:
		for line in f:
			images.append(line)
	return images


def download_nuswide(root):
	path_devkit = os.path.join(root, 'VOCdevkit')
	path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
	tmpdir = os.path.join(root, 'tmp')

	if not os.path.exists(root):
		os.makedirs(root)

	if not os.path.exists(path_devkit):

		if not os.path.exists(tmpdir):
			os.makedirs(tmpdir)

		parts = urlparse(urls['devkit'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)

		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['devkit'], cached_file))
			util.download_url(urls['devkit'], cached_file)

		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')

	if not os.path.exists(path_images):

		parts = urlparse(urls['trainval_2007'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)

		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['trainval_2007'], cached_file))
			util.download_url(urls['trainval_2007'], cached_file)

		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')

	test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
	if not os.path.exists(test_anno):

		parts = urlparse(urls['test_images_2007'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)

		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['test_images_2007'], cached_file))
			util.download_url(urls['test_images_2007'], cached_file)

		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')

	test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
	if not os.path.exists(test_image):

		parts = urlparse(urls['test_anno_2007'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)

		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['test_anno_2007'], cached_file))
			util.download_url(urls['test_anno_2007'], cached_file)

		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')


class NuswideClassification(data.Dataset):
	def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
		print("load {0} file\n".format(inp_name))
		self.root = root
		self.path_imglist = os.path.join(root, 'data','ImageList')
		self.path_traintestlabellist = os.path.join(root,'data','Groundtruth','TrainTestLabels')
		self.path_images = os.path.join(root, 'data', 'NUSWIDE', 'Flickr')
		self.set = set
		self.transform = transform
		self.target_transform = target_transform

		path_csv = os.path.join(self.root,'data', 'files')
		file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

		if not os.path.exists(file_csv):
			if not os.path.exists(path_csv):
				os.makedirs(path_csv)
			labeled_data = read_object_labels(self.root, '', self.set)
			write_object_labels_csv(file_csv, labeled_data)

		self.classes = object_categories
		self.images = read_object_labels_csv(file_csv)
		
		if self.inp_name:
			with open(inp_name, 'rb') as f:
				self.inp = pickle.load(f)
		self.inp_name = inp_name

		print('[dataset] NUSWIDE classification set=%s number of classes=%d  number of images=%d' % (
			set, len(self.classes), len(self.images)))

	def __getitem__(self, index):

		path, target = self.images[index]
		img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		if self.inp_name:
			return (img, path, self.inp), target
		else:
			return (img, path), target

	def __len__(self):
		'''
		return the amount of elements
		:return:
		'''
		return len(self.images)

	def get_number_classes(self):
		return len(self.classes)


if __name__=="__main__":
	file='./data/nuswide/ImageList/TrainImagelist.txt'
	labeldata = read_object_labels('./data/nuswide/', '', "Train")
	write_object_labels_csv('./classification_Train.csv', labeldata)
	img_list = read_object_labels_csv('./classification_Train.csv')
