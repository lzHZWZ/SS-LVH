import pickle
import torch
import parameter as pm
import KMeans as km
import result_process as rp
import copy
import numpy as np

def save_cluster_result(label_result, cluster_centers):
	result = np.array(label_result, dtype=object)
	np.save('label_result/label_result500', result)
	
	result = np.array(cluster_centers.to('cpu'), dtype=object)
	np.save('label_result/cluster_centers500', result)


def choose_device(cuda=False):
	if cuda:
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
	return device


if __name__ == "__main__":
	
	device = choose_device(True)
	
	file = open(pm.feature_label_weight_path, 'rb')
	feature_label_weight = pickle.load(file)
	
	file = open(pm.probabilty_path, 'rb')
	label_probability = pickle.load(file)
	
	min_distance = None
	for i in range(pm.kmean_iter):
		
		kmeans = km.KMeans(pm.n_cluster, pm.max_iter, device=device)
		label_result = kmeans.fit(feature_label_weight.to(device))
		
		distance = torch.mean(kmeans.dists)
		
		if (min_distance == None or distance < min_distance):
			min_distance = distance
			min_label_result = label_result
			min_cluster_centers = kmeans.centers
	
	print("min_distance", min_distance)
	save_cluster_result(min_label_result, min_cluster_centers)
	
	min_label_result = np.load('label_result/label_result500.npy', allow_pickle=True).tolist()
	min_cluster_centers = np.load('label_result/cluster_centers500.npy', allow_pickle=True).tolist()
	
	result_process = rp.Result_process(feature_label_weight, min_label_result, min_cluster_centers, label_probability,
									   True)
	max_label_index = result_process.get_result_by_way1(pm.max_cluster_number, pm.max_index_number)

	imagenet_reverse_coarse_labels = (np.load('../imagenet_labels/ImageNetLabels.npy', allow_pickle=True).tolist())["imagenet_reverse_coarse_labels"]
	print('****scheme 1****')
	for i in range(len(max_label_index)):
		print('--------------------------------')
		for j in range(len(max_label_index[i])):
			print(imagenet_reverse_coarse_labels[str(max_label_index[i][j])])
	
	print('****scheme 2****')
	max_label_index = result_process.get_result_by_way2(pm.max_cluster_number, pm.max_index_number)
	for i in range(len(max_label_index)):
		print('--------------------------------')
		for j in range(len(max_label_index[i])):
			print(imagenet_reverse_coarse_labels[str(max_label_index[i][j])])
