import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import mean, linspace

def get_membership_table(gblist):
    global membership
    membership_list=[]
    for i in range(len(gblist.granular_balls)):
        # ball_purity = gblist.granular_balls[i].purity
        num = np.shape(gblist.granular_balls[i].data)[0]
        # r = gblist.granular_balls[i].radius
        count_zeros = np.sum(gblist.granular_balls[i].data[:, -1] == 0)
        membership=count_zeros/num
        # membership = ball_purity / (1 + (np.exp(-r * num)))
        # print("粒球" + str(i + 1) + "中局部密度为：\n", membership_list)
        membership_list.append(membership),
        #membership_table_df = DataFrame(membership_q_table)
    print("membership_list_num,membership_list:",np.shape(membership_list),membership_list,)
    return membership_list


def calculate_av_fuzziness(gblist,membership_list):
    fuzziness_ = 0
    num_1_all=0
    for i in range(len(gblist.granular_balls)):
        num_1= np.shape(gblist.granular_balls[i].data)[0]
        num_1_all += num_1
        value =membership_list[i]
        if 0 < value <1:
            single_fuzziness = 4 * value * (1 - value)*num_1
            fuzziness_ += single_fuzziness
    av_fuzziness = fuzziness_/num_1_all
    # print('calculate_av_fuzziness:',num_2,av_fuzziness)
    return float(av_fuzziness)

def calculate_3way_fuzziness(gblist,membership_list, beta, alpha):
    num_1_all=0
    count=0
    for i in range(len(gblist.granular_balls)):
        num_1 = np.shape(gblist.granular_balls[i].data)[0]
        num_1_all += num_1
        value = membership_list[i]
        if beta < value < alpha:
            count += num_1
    fuzziness = 4 * count*(0.5 * alpha * alpha - (1 / 3) * alpha * alpha * alpha - 0.5 * beta * beta + (
                1 / 3) * beta * beta * beta) /num_1_all   #count指粒球个数，普通例子里面的样本个数
    return  float(fuzziness)

def compute_best_beta(gblist,membership_list):
    av_fuzziness = calculate_av_fuzziness(gblist,membership_list)
    # 设置参数迭代步长
    min_fuzziness = av_fuzziness
    best_beta = 0
    best_alpha = 0.5
    for beta in linspace(0.01, 0.5, 50,endpoint = False):
        for alpha in linspace(0.5, 1, 50,endpoint = False):
            _3way_fuzziness = calculate_3way_fuzziness(gblist,membership_list, beta, alpha)
            minus = abs(_3way_fuzziness - av_fuzziness)
            if minus < min_fuzziness:
                min_fuzziness = minus
                best_beta = beta
                best_alpha = alpha
    print('best_beta,best_alpha,min_fuzziness:', best_beta, best_alpha, min_fuzziness)
    return best_beta, best_alpha, min_fuzziness

