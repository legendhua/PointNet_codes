#-*-coding:utf-8-*-
#数据量少的类别的数据加扰动增强
import h5py
import numpy as np
import os 
import sys
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider

data_dir = os.path.join(BASE_DIR,'data/modelnet40_ply_hdf5_2048')
# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
# 数据增强主函数
def data_aug(method=0):
    ''' 输入参数
        method:0--------增加噪声
               1--------旋转物体
    '''
    object_data=np.zeros((1,2048,3))
    object_labels=np.zeros((1),np.int32)
    for fn in range(len(TRAIN_FILES)):
        print('loading the file'+str(fn))
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[fn])
        for label in range(current_label.shape[0]):
            if current_label[label] == 10 or current_label[label] == 38 or current_label[label] == 32 :
                object_data=np.vstack((object_data,current_data[label,:,:].reshape(1,2048,3)))
                object_labels=np.vstack((object_labels,current_label[label]))

    object_data=np.delete(object_data, 0, axis=0)
    object_labels=np.delete(object_labels,0,axis=0)
    #加噪后的数据
    if method == 0 :
        object_data = provider.jitter_point_cloud(object_data,sigma=0.001,clip=0.005)
    elif method == 1:
        need_to_rotate_labels = object_labels
        # 旋转6个角度
        for i in range(6):
            print(i)
            rotation_angle = i * (np.pi / 3.)
            rotate_data = provider.rotate_point_cloud_by_angle(object_data,rotation_angle)
            object_data = np.vstack((object_data,rotate_data))
            object_labels = np.vstack((object_labels,need_to_rotate_labels))

    for i in range(len(TRAIN_FILES)):
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[fn])
        object_data = np.vstack((object_data,current_data))
        object_labels = np.vstack((object_labels,current_label))
        object_data, object_labels, _ = provider.shuffle_data(object_data,object_labels)
        n_object=object_data.shape[0]
        num_each_file=n_object//6
        for i in range(6):
            f=h5py.File(data_dir+'/object_aug_rotate'+str(i)+'.h5','w')
            f['data']=object_data[(i*num_each_file):(i+1)*num_each_file,:,:]
            f['label']=object_labels[i*(num_each_file):(i+1)*num_each_file]
            f.close()

if __name__ == '__main__':
    data_aug(1)
