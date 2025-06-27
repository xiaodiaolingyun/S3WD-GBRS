import numpy as np

from matplotlib import pyplot as plt
from timeit import default_timer as timer
from pandas import DataFrame
import math
# from tool import get_membership_table,compute_best_beta


# 使用GBKNN算法计算标签,k=1
def GB_compute_label(x, gbList,):
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    distances = []
    for i in range(len(gbList.granular_balls)):
        distance1 = np.linalg.norm(x - gbList.granular_balls[i].center) - \
                    gbList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]

    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = gbList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        x_label = gbList.granular_balls[int(distances[0][1])].label
    return x_label

# 使用GB3WD算法计算标签,k=5,10,15
def GBKNN3WD_compute_label(x, GBList,best_beta, best_alpha):
    # 遍历粒球列表，计算目标点到各个粒球的距离：
    distances = []
    for i in range(len(GBList.granular_balls)):
        # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
        distance1 = np.linalg.norm(x - GBList.granular_balls[i].center) - \
                    GBList.granular_balls[i].radius
        distance1 = np.hstack([distance1, i])
        distances.append(distance1)
    distances = np.array(distances)
    distances = distances[np.argsort(distances[:, 0])]
    # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
    if distances[0][0] < 0:
        # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
        x_label = GBList.granular_balls[int(distances[0][1])].label
        # print("存在小于0的距离，按照最小距离的粒球标签赋值")
    else:
        gb_labels = []
        for i in range(10):
           gb_labels.append(GBList.granular_balls[int(distances[i][1])].label)
        belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
        if belong_value > best_alpha:
            x_label = 1
        elif belong_value < best_beta:
            x_label = 0
        else:
            x_label = -1
    return x_label

# 使用密度粒球KNN算法计算单个元素的隶属度
# def new_compute_x_membership(x, GBList):
#     # 遍历粒球列表，计算目标点到各个粒球的距离：
#     distances = []
#     tic_0 = timer()
#     l = int(len(GBList.granular_balls) * 0.96)
#     for i in range(int(len(GBList.granular_balls))):
#         # print(GBList.granular_balls[i].center, GBList.granular_balls[i].radius)
#         distance1 = np.linalg.norm(x - GBList.granular_balls[i].center) - \
#                     GBList.granular_balls[i].radius
#         distance1 = np.hstack([distance1, i])
#         distances.append(distance1)
#     distances = np.array(distances)
#     distances = distances[np.argsort(distances[:, 0])]
#     print('distances:',distances)
#     # print('密度粒球KNN计算距离矩阵的时间为：{}'.format(tic_2 - tic_0))
#     # 通过距离列表计算K近邻的K值    以及  确定目标点对应的标签
#     k = 1
#     if distances[0][0] < 0:
#         # 如果距离列表存在小于0的元素，则将最小的距离所对应的粒球标签赋值给目标点
#         x_belong_value = GBList.granular_balls[int(distances[0][1])].label
#         # print("存在小于0的距离，按照最小距离的粒球标签赋值")
#     else:
#         gb_labels = []
#         num_centers, cluster_centers_ = cluster_centers(distances[:, 0], 0.48, 0.02, l)
#         for i in range(len(cluster_centers_)):
#             for j in range(len(distances[:, 0])):
#                 if cluster_centers_[i] == distances[j][0]:
#                     # print(distances[j][1])
#                     gb_labels.append(GBList.granular_balls[int(distances[j][1])].label)
#         x_belong_value = sum(gb_labels) / (len(gb_labels) + float(1e-8))
#         # gb_labels = []
#         # for i in range(15):
#         #     gb_labels.append(GBList.granular_balls[int(distances[i][1])].label)
#         # x_belong_value= sum(gb_labels) / (len(gb_labels) + float(1e-8))
#         # # gb_labels = []
#         # num_centers, cluster_centers_= cluster_centers(distances[:, 0], 0.48, 0.02, 50)
#         # for i in range(len(cluster_centers_)):
#         #     gb_labels.append(GBList.granular_balls[int(distances[i][1])].label)
#         # # print(gb_labels)
#         # # 这里假设为2分类 且标签要么为0 要么为1
#         # x_belong_value = sum(gb_labels) / (len(gb_labels) + float("1e-8"))
#     return x_belong_value


def compute_ac(test_data, compute_type,granular_balls_original,  best_beta_improved, best_alpha_improved,):
    test_data_no_label = test_data[:, :-1]
    right_num = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    x_label = 0
    for i in range(len(test_data_no_label)):
        x = test_data_no_label[i]
        if compute_type == 'OK':
            x_label = GB_compute_label(x, granular_balls_original,)
        elif compute_type == 'OK3':
            x_label = GBKNN3WD_compute_label(x, granular_balls_original, best_beta_improved, best_alpha_improved)
        else:
            print('计算类型输入错误，无法计算')
        if x_label == test_data[i, -1]:
            right_num += 1
        if test_data[i, -1] == 1 and x_label == 1:
            tp += 1
        elif test_data[i, -1] == 1 and x_label == 0:
            fp += 1
        elif test_data[i, -1] == 0 and x_label == 1:
            fn += 1
        elif test_data[i, -1] == 0 and x_label == 0:
            tn += 1
    accuracy = right_num / (tp+fp+fn+tn+float("1e-8"))
    # 查准率，精确率 越高越好
    P = tp / (tp + fp + float("1e-8"))
    # 查全率，召回率 越高越好
    R = tp / (tp + fn + float("1e-8"))
    # f1评分，数值越大越稳定，（但还要考虑模型的泛化能力，不能造成过拟合）
    f1_score = (2 * P * R) / (P + R + float("1e-8"))
    print("right_num,tp,fp ,fn,tn:",right_num,tp,fp ,fn,tn)
    return accuracy, f1_score, P, R,

