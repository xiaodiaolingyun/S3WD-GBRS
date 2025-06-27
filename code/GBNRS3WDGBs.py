# -------------------------------------------------------------------------
# Aim:
# introduce granular-ball and propose a granular-ball-based DP algorithm, called GB-DP
# -------------------------------------------------------------------------
# Written by Dongdong Cheng, Ya Li
# Chongqing University of Posts and Telecommunications
# 2023


"""
granular ball neighborhood rough set
"""

import numpy as np
from collections import Counter
from sklearn.cluster import k_means
import warnings
from numpy import mean

warnings.filterwarnings("ignore")


class GranularBall:
    """class of the granular ball"""

    def __init__(self, data):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.data = data[:, :]
        self.data_no_label = data[:, :-1]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.label, self.purity = self.__get_label_and_purity()
        self.radius = self.__get_gbradis()

    def __get_gbradis(self):  # 获取粒球的半径
        return float(mean(np.sqrt(np.sum(np.asarray(self.center - self.data_no_label) ** 2, axis=1))))

    def __get_label_and_purity(self):
        """
        :return: the label and purity of the granular ball.
        """
        count = Counter(self.data[:, -1])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        return label, purity

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        label_cluster = k_means(X=self.data_no_label, n_clusters=2)[1]
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :])
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2





class GBList:
    """class of the list of granular ball"""

    def __init__(self, data=None):
        self.data = data[:, :]
        self.granular_balls = [GranularBall(self.data)] # gbs is initialized with all data



    def calculate_num(self):
        if self.data is not None:
            return int(np.ceil(np.sqrt(self.data.shape[0])))
        else:
            return 0

    def init_granular_balls(self, ):
        num = self.calculate_num()
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ll = len(self.granular_balls)
        GB_list_new = []
        i = 0
        while True:
            if self.granular_balls[i].num > num:
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)

    def del_balls(self, purity=0, num_data=0):
        """
        Deleting the balls that meets following conditions from the list, updating self.granular_balls and self.data.
        :param purity: delete the balls that purity is large than this value.
        :param num_data: delete the balls that the number of samples is large than this value.
        :return: None
        """
        self.granular_balls = [ball for ball in self.granular_balls if ball.purity >= purity and ball.num >= num_data]
        self.data = self.get_data()

    def re_k_means(self):
        """
        Global k-means clustering for data with the center of the ball as the initial center point.
        """
        k = len(self.granular_balls)
        label_cluster = k_means(X=self.data[:, :-2], n_clusters=k, init=self.get_center())[1]
        for i in range(k):
            self.granular_balls[i] = GranularBall(self.data[label_cluster == i, :])

    def re_division(self, i):
        """
        Data division with the center of the ball.
        :return: a list of new granular balls after divisions.
        """
        k = len(self.granular_balls)
        attributes = list(range(self.data.shape[1] - 2))
        attributes.remove(i)
        label_cluster = k_means(X=self.data[:, attributes], n_clusters=k,
                                init=self.get_center()[:, attributes], max_iter=1)[1]
        granular_balls_division = []
        for i in set(label_cluster):
            granular_balls_division.append(GranularBall(self.data[label_cluster == i, :]))
        return granular_balls_division
