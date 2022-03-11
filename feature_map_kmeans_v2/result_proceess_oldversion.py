import torch
import numpy as np
import copy


class Result_process:

    def __init__(self, label_result,label_probability):
        self.label_result = label_result
        self.label_probability = label_probability
        self.result_pro = None
        self.max_cluster = None
        self.max_cluster_index = None


    def get_largest_index(self, data, k):

        copy_data = copy.deepcopy(data)
        max_index = []

        for _ in range(k):
            max_data = max(copy_data)
            index = copy_data.index(max_data)
            copy_data[index] = 0
            max_index.append(index)
        return max_index

    def get_probability_with_label_result(self):

        probability_result = []
        for i in range(len(self.label_result)):

            probability_cluster = []
            for j in range(len(self.label_result[i])):
                probability_cluster.append(self.label_probability[0][self.label_result[i][j]].to('cpu').detach().numpy())
            probability_result.append(probability_cluster)

        self.result_pro = probability_result

    def get_max_cluster_index(self, max_cluster_number):

        mean_cluster_pro = []
        cluster_len = len(self.result_pro)

        for i in range(cluster_len):
            mean_cluster_pro.append(np.mean(self.result_pro[i]))

        max_number = max_cluster_number
        if cluster_len < max_cluster_number:
            max_number = cluster_len

        index = self.get_largest_index(mean_cluster_pro, max_number)

        max_cluster = []
        max_cluster_index = []

        for i in index:
            max_cluster.append(self.result_pro[i])
            max_cluster_index.append(self.label_result[i])

        self.max_cluster = max_cluster
        self.max_cluster_index = max_cluster_index



    def get_max_label_index(self, max_index_number):

        max_label_index = []

        for i in range(len(self.max_cluster)):
            max_number = max_index_number

            if (len(self.max_cluster[i]) < max_index_number):
                max_number = len(self.max_cluster[i])

            index = self.get_largest_index(self.max_cluster[i], max_number)

            index_set = []
            for j in index:
                index_set.append(self.max_cluster_index[i][j])
            max_label_index.append(index_set)

        return max_label_index


    def get_result(self,max_cluster_number, max_index_number):

        self.get_probability_with_label_result()

        self.get_max_cluster_index(max_cluster_number)

        max_label_index = self.get_max_label_index(max_index_number)


        return max_label_index