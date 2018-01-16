#-*- coding: utf-8 -*-
import os
import sys
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
num_train_file = len(train_file_list)
cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[0])
cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)
#选择可视化的目标
cur_data=cur_data[4,:,:].reshape(2048,3)#chair
cur_seg = cur_seg[4,:]
#对应的颜色
color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))
#灰色点，未分割
def output_point_cloud(data,out_file):
    with open(out_file,'w') as f:
        for i in range(data.shape[0]):
            f.write('v %f %f %f\r\n'%(data[i][0],data[i][1],data[i][2]))
#分割结果
def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
if __name__=='__main__':
    render_file=os.path.join(BASE_DIR,'render_chair.obj')
    output_point_cloud(cur_data,render_file)
    render_file_color=os.path.join(BASE_DIR,'render_chair_color.obj')
    output_color_point_cloud(cur_data, cur_seg, render_file_color)
    
