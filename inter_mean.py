import torch as th

def inter_mean(*listarray):
	list_list = []
	length = len(listarray)
	for i in range(length):
		list_list.append(th.tensor(listarray[i]).reshape(1, -1))
	x = th.FloatTensor(th.FloatStorage())
	for i in range(length):
		x = th.cat((x, list_list[i]), 0)
		
	y = th.sum(x, 0)
	return y / length

