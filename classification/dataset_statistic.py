# -*- coding: utf-8 -*-
"""
统计ModelNet40数据集类别的分布
@author: zgh
"""

import numpy as np
import json
import os 
import sys
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
#定义路径(根据实际情况适当更改)
hdf5_data_dir = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048')
NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

TRAINING_FILES = os.path.join(hdf5_data_dir, 'train_files.txt')
train_files = provider.getDataFiles(TRAINING_FILES)
num_train_files = len(train_files)

    
#--------------输出数据并且保存--------------------------
def printout(flog, data):
	print(data)
	flog.write(data + '\n')

#--------------训练数据进行统计分析-----------------------
class_statistic=np.zeros((NUM_CLASSES)).astype(np.int32)
count_object=0
for i in range(num_train_files):
    cur_train_filename = os.path.join(BASE_DIR, train_files[i])
    current_data, current_label = provider.loadDataFile(cur_train_filename)
    num_object = current_label.shape[0]
    for i in range(num_object):
        class_statistic[current_label[i]]  += 1
        count_object += 1

flog = open('class_statistic.txt', 'w')
for i in range(NUM_CLASSES):
    printout(flog,'%10s:\t %f' % (SHAPE_NAMES[i],class_statistic[i]*100.0/count_object))
flog.close()

# 绘图
x=range(0,40)   #x轴数据的范围
y= class_statistic*100.0/count_object  #y轴数据
y_idx=np.argsort(-y) #对y按照降序排序，返回降序后的标签顺序
y_sort=y[y_idx]
SHAPE_NAMES_SORT=[]
for i in range(len(y_idx)):
    SHAPE_NAMES_SORT.append(SHAPE_NAMES[y_idx[i]])
SHAPE_NAMES_XTICKS = ["$"+name+"$"for name in SHAPE_NAMES_SORT]
plt.figure(figsize=(8,4)) 
plt.plot(x,y_sort,"r*-")
plt.xlabel("class name")
plt.ylabel("percents(%)")
plt.xticks(x,SHAPE_NAMES_XTICKS, rotation=90)
plt.title("ModelNet40 statistic")
plt.show()