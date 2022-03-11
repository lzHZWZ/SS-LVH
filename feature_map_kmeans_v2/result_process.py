import torch
import numpy as np
import copy
import sys


class Result_process:

    def __init__(self, feature_label_weight, label_result,
                 cluster_center, label_probability, is_removal=False):

        self.feature_label_weight = feature_label_weight
        self.label_result = label_result
        self.cluster_center = cluster_center
        self.label_probability = label_probability

        self.cluster_result = None
        self.cluster_dist = None
        self.max_cluster_index = None
        self.max_cluster = None
        self.max_center = None
        self.weight = 100


        if (is_removal == True):
            self.pre_label_removal()
            print('label operating end')

        self.get_cluster_dist()



    def pre_label_removal(self):

        mean_p = 1 / len(self.label_probability[0])

        label_result = []
        cluster_result = []
        cluster_center = []

        for i in range(len(self.label_result)):
            label_set = []
            cluster_set = []

            for j in range(len(self.label_result[i])):
                if(self.label_probability[0][self.label_result[i][j]] >= mean_p):
                    label_set.append(self.label_result[i][j])
                    cluster_set.append(self.feature_label_weight[self.label_result[i][j]])

            if(len(label_set)):
                label_result.append(label_set)
                cluster_result.append(cluster_set)
                cluster_center.append(self.cluster_center[i])

        self.label_result = label_result
        self.cluster_result = cluster_result
        self.cluster_center = cluster_center


    def get_cluster_dist(self):

        cluster_dist = []

        for i in range(len(self.label_result)):
            dist = 0
            for j in range(len(self.label_result[i])):
                dist += np.sqrt(np.sum(np.square(np.array(self.cluster_center[i])-np.array(self.cluster_result[i][j]))))
            cluster_dist.append(dist)

        self.cluster_dist = cluster_dist

    def get_smallest_index(self, data, k):

        copy_data = copy.deepcopy(data)
        min_index = []

        for _ in range(k):
            min_data = min(copy_data)
            index = copy_data.index(min_data)
            copy_data[index] = sys.maxsize
            min_index.append(index)
        return min_index

    def get_max_cluster_index_by_way1(self, max_cluster_number):

        metric_set = []

        max_number = max_cluster_number
        if len(self.label_result) < max_cluster_number:
            max_number = len(self.label_result)

        for i in range(len(self.label_result)):

            metric = 0
            for j in range(len(self.label_result[i])):
                dist = np.sqrt(np.sum(np.square(np.array(self.cluster_center[i])-np.array(self.cluster_result[i][j]))))
                pro = self.label_probability[0][self.label_result[i][j]]

                if(self.cluster_dist[i] == 0):
                    metric = metric - (pro * self.weight)
                else:
                    metric = metric + ( (dist / self.cluster_dist[i]) - ( pro * self.weight))

            metric = metric / len(self.label_result[i])
            metric_set.append(metric.to('cpu').detach().numpy())

        index = self.get_smallest_index(metric_set, max_number)


        max_cluster_index = []
        max_cluster = []
        max_center = []

        for i in index:
            max_cluster_index.append(self.label_result[i])
            max_cluster.append(self.cluster_result[i])
            max_center.append(self.cluster_center[i])

        self.max_cluster_index = max_cluster_index
        self.max_cluster = max_cluster
        self.max_center = max_center

    def get_max_cluster_index_by_way2(self, max_cluster_number):

        metric_set = []

        max_number = max_cluster_number
        if len(self.label_result) < max_cluster_number:
            max_number = len(self.label_result)

        for i in range(len(self.label_result)):

            metric = 0
            for j in range(len(self.label_result[i])):
                dist = np.sqrt(
                    np.sum(np.square(np.array(self.cluster_center[i]) - np.array(self.cluster_result[i][j]))))
                pro = self.label_probability[0][self.label_result[i][j]]

                metric = metric + (dist / pro * self.weight)

            metric = metric / len(self.label_result[i])
            metric_set.append(metric.to('cpu').detach().numpy())

        index = self.get_smallest_index(metric_set, max_number)

        max_cluster_index = []
        max_cluster = []
        max_center = []

        for i in index:
            max_cluster_index.append(self.label_result[i])
            max_cluster.append(self.cluster_result[i])
            max_center.append(self.cluster_center[i])

        self.max_cluster_index = max_cluster_index
        self.max_cluster = max_cluster
        self.max_center = max_center

    def get_max_label_index_by_way1(self, max_index_number):

        max_label_index = []

        for i in range(len(self.max_cluster_index)):

            max_number = max_index_number

            if (len(self.max_cluster_index[i]) < max_index_number):
                max_number = len(self.max_cluster_index[i])

            metric = []
            for j in range(len(self.max_cluster_index[i])):

                dist = np.sqrt(np.sum(np.square(np.array(self.max_center[i]) - np.array(self.max_cluster[i][j]))))
                pro = self.label_probability[0][self.max_cluster_index[i][j]]

                if (self.cluster_dist[i] == 0):
                    metric.append((-pro * self.weight).to('cpu').detach().numpy())
                else:
                    metric.append(((dist / self.cluster_dist[i]) - (pro * self.weight)).to('cpu').detach().numpy())

            index = self.get_smallest_index(metric, max_number)

            index_set = []

            for j in index:
                index_set.append(self.max_cluster_index[i][j])
            max_label_index.append(index_set)

        return max_label_index

    def get_max_label_index_by_way2(self, max_index_number):

        max_label_index = []

        for i in range(len(self.max_cluster_index)):

            max_number = max_index_number

            if (len(self.max_cluster_index[i]) < max_index_number):
                max_number = len(self.max_cluster_index[i])

            metric = []
            for j in range(len(self.max_cluster_index[i])):

                dist = np.sqrt(np.sum(np.square(np.array(self.max_center[i]) - np.array(self.max_cluster[i][j]))))
                pro = self.label_probability[0][self.max_cluster_index[i][j]]

                metric.append((dist / pro.to('cpu').detach().numpy() * self.weight))

            index = self.get_smallest_index(metric, max_number)

            index_set = []

            for j in index:
                index_set.append(self.max_cluster_index[i][j])
            max_label_index.append(index_set)

        return max_label_index




    def get_result_by_way1(self,max_cluster_number, max_index_number):

        self.get_max_cluster_index_by_way1(max_cluster_number)

        max_label_index = self.get_max_label_index_by_way1(max_index_number)

        return max_label_index

    def get_result_by_way2(self, max_cluster_number, max_index_number):

        self.get_max_cluster_index_by_way2(max_cluster_number)

        max_label_index = self.get_max_label_index_by_way2(max_index_number)

        return max_label_index