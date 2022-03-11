import torch
import random

class KMeans:

    def __init__(self, n_cluster=20, max_iter=None, verbose=True, device=torch.device):
        self.n_cluster = n_cluster
        self.labels = None
        self.dists = None
        self.centers = None
        self.device = device
        self.started = False
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.max_iter = max_iter
        self.count = 0

    def nearest_center(self, x):

        labels = torch.empty((x.shape[0],)).long().to(self.device)

        dists = torch.empty((0,self.n_cluster)).to(self.device)

        for i, sample in enumerate(x):

            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers),(1))

            labels[i] = torch.argmin(dist)

            dists = torch.cat([dists,dist.unsqueeze(0)],(0))

        self.labels = labels


        if self.started:
            self.variation = torch.sum(self.dists - dists)

        self.dists = dists
        self.started = True

    def update_center(self,x):

        centers = torch.empty((0,x.shape[1])).to(self.device)

        for i in range(self.n_cluster):

            mask = self.labels == i
            cluster_sample = x[mask]
            centers = torch.cat([centers,torch.mean(cluster_sample,(0)).unsqueeze(0)],(0))

        self.centers = centers

    def get_cluster_result(self,x):

        label_result = []
        for i in range(self.n_cluster):
            index = self.labels == i
            label_result.append(torch.where(index == True)[0].to('cpu').numpy().tolist())

        return label_result


    def fit(self, x):

        init_center_index = random.sample(range(0,x.shape[0]),self.n_cluster)

        init_center = x[init_center_index]
        self.centers = init_center

        while True:
            self.nearest_center(x)
            self.update_center(x)

            if torch.abs(self.variation) < 1e-10 or self.count == self.max_iter:
                break

            self.count += 1

        print('num of iteration:', self.count)
        print('accumulative distance：{:.10f}'.format(torch.mean(self.dists)))
        return self.get_cluster_result(x)
