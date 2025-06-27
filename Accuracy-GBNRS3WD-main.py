import datetime
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold
from timeit import default_timer as timer
# 数据集读取
from sklearn.preprocessing import MinMaxScaler
from tool import get_membership_table,compute_best_beta

from GBNRS3WDGBs import GBList
from tools import compute_ac






GB_accuracy = []
GB3WD_accuracy = []

GB_f1 = []
GB3WD_f1 = []

GB_P = []
GB3WD_P = []

GB_R = []
GB3WD_R = []

GB_runtime = []
GB3WD_runtime = []






filenames=['hcv(596,10).csv',]

for filename in filenames:
    '''数据读取与预处理'''
    # 效果比较好的数据集wifi_localization(2000,8).csv
    # df = pd.read_csv("datasets/待取三个数据集/" + filename)
    df = pd.read_csv(filename )
    print(df)
    print("———————————————————————————正在处理数据集:" + filename + "—————————————————————————————————————————————")
    # data = df.values
    '''归一化数据集'''
    da1 = df.values
    print(da1)
    da2 = np.unique(da1, axis=0)
    min_max = MinMaxScaler()
    da3 = min_max.fit_transform(da2[:, :-1])
    data= np.hstack([da3, da2[:, -1:]])

    # print()
    # data_ml_knn = data[:, :-1]
    # target_ml_knn = data[:, -1:]#data[:, -1:] 返回一个二维子数组，而 data[:, -1] 返回一个一维数组。
    '''十折交叉验证'''             #如果您需要保留维度信息，可以使用 data[:, -1:]。
    # 原始十折交叉验证
    k_fold = StratifiedKFold(n_splits=10)
    k_fold = k_fold.split(data, data[:, -1])
    # 随机排列交叉验证
    # k_fold = ShuffleSplit(n_splits=10)
    # k_fold = k_fold.split(data)


    original_accuracy_list = []
    improved_accuracy_list = []

    original_f1_list = []
    improved_f1_list = []

    original_P_list = []
    improved_P_list = []

    original_R_list = []
    improved_R_list = []

    original_time_list=[]
    improved_time_list=[]


    for k, (train_data_index, test_data_index) in enumerate(k_fold):
        # if k == 0:
        print('迭代次数：{}'.format(k + 1))
        print('训练数据长度：{}'.format(len(train_data_index)))
        print('测试数据长度：{}'.format(len(test_data_index)))
        train_data = data[train_data_index, :]
        test_data = data[test_data_index, :]
        # best_beta_original, best_alpha_original, min_fuzziness_original = compute_best_beta_traditional(train_data)
        # 使用训练数据生成粒球
        '''step 1 生成原始粒球列表'''
        granular_balls_original = GBList(train_data)  # create the list of granular balls
        granular_balls_original.init_granular_balls()  # initialize the list
        # 是否删除元素小于2的粒球对结果影响很大
        granular_balls_original.del_balls(num_data=2)  # delete the ball with 1 (less than 2) sample
        print('原始粒球数：{}'.format(len(granular_balls_original.granular_balls)))
        membership_list_original = get_membership_table(granular_balls_original)
        best_beta_improved,best_alpha_improved, min_fuzziness_original = compute_best_beta (granular_balls_original,membership_list_original)

        tic_1= timer()
        original_accuracy, original_f1, original_P, original_R = compute_ac(test_data,"OK",granular_balls_original,  best_beta_improved,best_alpha_improved,)
        tic_2 = timer()
        improved_accuracy, improved_f1, improved_P, improved_R=compute_ac(test_data,"OK3",granular_balls_original,  best_beta_improved,best_alpha_improved,)
        tic_3 = timer()

        # ________________________待填坑___________________________________
        # 十折交叉验证的十次准确率

        original_accuracy_list.append(original_accuracy)
        improved_accuracy_list.append(improved_accuracy)

        # 十折交叉验证的十次F1评分

        original_f1_list.append(original_f1)
        improved_f1_list.append(improved_f1)


        original_P_list.append(original_P)
        improved_P_list.append(improved_P)


        original_R_list.append(original_R)
        improved_R_list.append(improved_R)


        # 十折交叉验证的十次执行时间

        original_time = tic_2 - tic_1
        improved_time = tic_3 - tic_2

        # 十折交叉验证的十次执行时间

        original_time_list.append(original_time)
        improved_time_list.append(improved_time)




    # 不同数据集的准确率均值

    GB_accuracy.append(np.mean(original_accuracy_list))
    GB3WD_accuracy.append(np.mean(improved_accuracy_list))
    print("Accuracy:",GB_accuracy, GB3WD_accuracy,)

    # 不同数据集的Precision均值
    GB_P.append(np.mean(original_P_list))
    GB3WD_P.append(np.mean(improved_P_list))
    print("Precision:", GB_P, GB3WD_P, )

    # 不同数据集的Recall均值
    GB_R.append(np.mean(original_R_list))
    GB3WD_R.append(np.mean(improved_R_list))
    print("Recall:", GB_R, GB3WD_R, )

    # 不同数据集的f1_score均值

    GB_f1.append(np.mean(original_f1_list))
    GB3WD_f1.append(np.mean(improved_f1_list))
    print("F1:",GB_f1, GB3WD_f1,)

    # 不同数据集的执行时间均值
    GB_runtime.append(np.mean(original_time_list))
    GB3WD_runtime.append(np.mean(improved_time_list))
    print("Runtime:", GB_runtime, GB3WD_runtime)

    data = {
            'GBKNN': [GB_accuracy, GB_f1,GB_R, GB_P,  GB_runtime, ],
            '3WD-GBNRS': [GB3WD_accuracy, GB3WD_f1,GB3WD_R,GB3WD_P,  GB3WD_runtime, ],
            }
    index_names = ['Accuracy', 'F1','Recall','Precision',  'Runtime']

    #  使用pandas创建DataFrame，并指定行名
    df = pd.DataFrame(data, index=index_names)
    # 输出表格到控制台
    print(df)

    # df.to_csv('result/Experiment-3-4/GBKNN-3WD-GBNRS.csv', mode='a', index_label=filename,index=True)

# # 将结果写入表格中
# date = str(datetime.date.today())
# excel_name = 'test' + date + '.xlsx'
# result = pd.DataFrame({'数据集': filenames1,
#                        "DB-KNN":GB_accuracy_k,
#                        "DBSCAN-KNN":DBGB_accuracy_k,
#                        'DB-3WD-精度': GB_accuracy,
#                        'DBSCAN-3WD-精度': DBGB_accuracy,
#                        '0.5_AM_粒球精度': GB_H_accuracy,
#                        '0.5_AM_DBSCAN粒球精度': DBGB_H_accuracy,
#
#                        'DB-KNN时间':GB_runtime_k,
#                        'DBSCAN-KNN时间':DBGB_runtime_k,
#                        'DB-3WD-时间': GB_runtime,
#                        'DBSCAN-3WD-时间': DBGB_runtime,
#                        '0.5_AM_粒球-F1': GB_H_runtime,
#                        '0.5_AM_DBSCAN粒球-F1': DBGB_H_runtime,
#
#                        'DB-KNN-f1':GB_f1_k,
#                        'DBSCAN-KNN-f1':DBGB_f1_k,
#                        'DB-3WD-f1': GB_f1,
#                        'DBSCAN-3WD-f1': DBGB_f1,
#                        '0.5_AM_粒球-f1': GB_H_f1,
#                        '0.5_AM_DBSCAN粒球-f1': DBGB_H_f1,
#
#                        'DB-KNN-P':GB_P_k,
#                        'DBSCAN-KNN-P':DBGB_P_k,
#                        'DB-3WD-P': GB_P,
#                        'DBSCAN-3WD-P': DBGB_P,
#                        '0.5_AM_粒球-P': GB_H_P,
#                        '0.5_AM_DBSCAN粒球-P': DBGB_H_P,
#
#                        'DB-KNN-R':GB_R_k,
#                        'DBSCAN-KNN-R':GB_R_k,
#                        'DB-3WD-R': GB_R,
#                        'DBSCAN-3WD-R': DBGB_R,
#                        '0.5_AM_粒球-R': GB_H_R,
#                        '0.5_AM_DBSCAN粒球-R': DBGB_H_R,
#
#                        })
# # result.to_excel('result/various3WD/粒球-3WD-compare' + '-' + excel_name, sheet_name='sheet1', index=False)

# # 将不同数据集不同算法下的准确率画为图形并保存
# labels = ['a', ]
# x = np.arange(len(labels))
# width = 0.2
# for i in range(0,5):
#     old=[GB_accuracy_k,GB_f1_k,GB_R_k,GB_P_k,GB_runtime_k,]
#     new=[DBGB_accuracy_k,DBGB_f1_k,DBGB_R_k,DBGB_P_k,DBGB_runtime_k,]
#     DB=[GB_accuracy,GB_f1,GB_R,GB_P,GB_runtime,]
#     DBGB=[DBGB_accuracy,DBGB_f1,DBGB_R,DBGB_P,DBGB_runtime,]
#     DB_HAM=[DBGB_H_accuracy, DBGB_H_f1, DBGB_H_R, DBGB_H_P, DBGB_H_runtime, ],
#     DBGB_HAM=[DBGB_H_accuracy, DBGB_H_f1, DBGB_H_R, DBGB_H_P, DBGB_H_runtime, ],
#
#     ylabel=['accuracy','f1','R','P','runtime']
#     plt.figure()
#
#     plt.bar(x - 2.5*width, old[i], width, label='GB-KNN')
#     plt.bar(x - 1.5*width, new[i], width, label='DBGB-KNN')
#     plt.bar(x + 0.5*width, DB[i], width, label='GB-3WD')
#     plt.bar(x + 0.5*width, DBGB[i], width, label='DBGB-3WD')
#     plt.bar(x + 1.5*width, DB_HAM[i], width, label='GB-HAM')
#     plt.bar(x + 2.5*width, DBGB_HAM[i], width, label='DBGB-HAM')
#
#     plt.xlabel('The various datasets')
#     plt.title('The '+ ylabel[i]  + ' of various algorithms')
#     plt.xticks(x, labels)
#     plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 1))
#     # plt.savefig('result/ac_and_time/total_experiment', dpi=600, bbox_inches='tight')
#     plt.show()



