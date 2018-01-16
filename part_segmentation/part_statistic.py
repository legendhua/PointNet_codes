# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:56:43 2017

统计所有物体的part类别的分布
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
hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_PART_CATS = len(all_cats)
#---解决part名称和物体的对应
all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()
part2cat = json.load(open(os.path.join(hdf5_data_dir,'part_belong_to_object.json'),'r'))


TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
num_train_file = len(train_file_list)

output_dir=  os.path.join(BASE_DIR, 'data_statistic')
if not os.path.exists(output_dir):
    os.mkdir(output_dir )
    
STATISTIC_FOLDER =  os.path.join(output_dir, 'part_statistic')
if not os.path.exists(STATISTIC_FOLDER):
    os.mkdir(STATISTIC_FOLDER )
#--------------输出数据并且保存--------------------------
def printout(flog, data):
	print(data)
	flog.write(data + '\n')

#-----------------------定义数据的统计量---------------------------
total_part_statistic=np.zeros((NUM_PART_CATS)).astype(np.int32)
count_part=0

for i in range(num_train_file):
    cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[i])
    cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)#读取一个文件
    num_object=cur_seg.shape[0]    #当前文件包含的物体数目
    print num_object
    for object_idx in range(num_object):
        seg_de_repeat=set(cur_seg[object_idx])  #去除当前列表的重复项
        count_part=count_part+len(seg_de_repeat)#统计所有物体的所有part的个数
        
        for item in seg_de_repeat:
            total_part_statistic[item]+=1#单独对part计数
#-------------------------结果展示--------------------------------------------
flog = open(os.path.join(STATISTIC_FOLDER , 'part_statistic.txt'), 'w')
PART2OBJECT_NAME = []
for i in range(NUM_PART_CATS):
    printout(flog,'num %d percents:%f' % (i,total_part_statistic[i]*100.0/count_part))
    PART2OBJECT_NAME.append(all_obj_cats[part2cat[i]][0])
flog.close()

#-----------------------绘图---------------------------
x=range(0,50)
y= total_part_statistic*100.0/count_part
y_idx=np.argsort(-y)
y_sort=y[y_idx]
SHAPE_NAMES_SORT = []
for i in range(len(y_idx)):
   SHAPE_NAMES_SORT.append(PART2OBJECT_NAME[y_idx[i]])
SHAPE_NAMES_XTICKS = ["$"+name+"$"for name in SHAPE_NAMES_SORT]
plt.figure(figsize=(8,4)) 
plt.plot(x,y_sort,"r*-")
plt.xlabel("part&class_name")
plt.ylabel("percents(%)")
plt.xticks(x,SHAPE_NAMES_XTICKS , rotation=90)
plt.title("ShapeNet part statistic")
plt.show()
#plt.savefig("part_statistics.png")
