#-*-coding:utf-8-*-
#数据量少的part加扰动增强
import h5py
import numpy as np
import os 
import sys
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider

hdf5_data_dir=os.path.join(BASE_DIR,'./hdf5_data')
train_file_list=provider.getDataFiles(os.path.join(hdf5_data_dir,'train_hdf5_file_list.txt'))
num_train_file=len(train_file_list)
#渲染文件
def output_point_cloud(data,out_file):
    with open(out_file,'w') as f:
        for i in range(data.shape[0]):
            f.write('v %f %f %f\r\n'%(data[i][0],data[i][1],data[i][2]))
def find_all_needto_aug_part():
    '''
    提取出源文件中符合条件的数据
    '''
    #初始化一个跟数据格式一样的数据原本
    object_data=np.zeros((1,2048,3))
    object_labels=np.zeros((1),np.int32)
    object_seg=np.zeros((1,2048),np.int32)
    #提取出所有符合标签的数据
    sigma=0.001
    clip=0.005
    for i in range(num_train_file):
        print('load the num '+str(i)+' train file')
        cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[i])
        cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)
        for nlabel in range(cur_data.shape[0]):
             jittered_data = np.clip(sigma * np.random.randn(cur_data.shape[0],cur_data.shape[1],cur_data.shape[2]),-clip,clip)
             for npoint in range(cur_data.shape[1]):
                 #寻找满足条件的点云
                if cur_seg[nlabel][npoint]!=4 or cur_seg[nlabel][npoint]!=8 or cur_seg[nlabel][npoint]!=27 or cur_seg[nlabel][npoint]!=30 or  \
                cur_seg[nlabel][npoint]!=31 or cur_seg[nlabel][npoint]!=34 or cur_seg[nlabel][npoint]!=40 or cur_seg[nlabel][npoint]!=42 or  \
                cur_seg[nlabel][npoint]!=46 or cur_seg[nlabel][npoint]!=49:
                    jittered_data[nlabel][npoint][:]=0
        cur_data=cur_data+jittered_data
        object_data=np.vstack((object_data,cur_data))
        object_labels=np.vstack((object_labels,cur_labels))
        object_seg=np.vstack((object_seg,cur_seg))	
        print('train_file '+str(i)+'  success')
                #上述object_data、object_lables、object_seg即包含了所有符合条件的数据
    num=object_data.shape[0]
    idx=np.random.randint(1,num,size=1)#随机将第一个初始化的数据赋成数据中的一个
    object_data[0,:,:]=object_data[idx,:,:]#数据第一个0项初始化为任意一项
    object_labels[0]=object_labels[idx]
    object_seg[0,:]=object_seg[idx,:]
    return object_data,object_labels,object_seg
    
def object_aug(object_data,object_labels,object_seg):
    #合并所有数据
    #渲染一个数据观察
    render_file=os.path.join(BASE_DIR,'render_aug_part.obj')
    output_point_cloud(object_data[3,:,:],render_file)
    
    for i in range(num_train_file):
        cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[i])
        cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)
        object_data=np.vstack((object_data,cur_data))
        object_labels=np.vstack((object_labels,cur_labels))
        object_seg=np.vstack((object_seg,cur_seg))	
    object_data,object_labels,object_seg=provider.shuffle_data_with_seg(object_data,object_labels,object_seg)
    #将数据分成几个文件
    n_object=object_data.shape[0]
    num_every_file=n_object//8
    for i in range(8):
        f=h5py.File(hdf5_data_dir+'/object_part_aug'+str(i)+'.h5','w')
        f['data']=object_data[i*(num_every_file):(i+1)*num_every_file,:,:]
        f['label']=object_labels[i*(num_every_file):(i+1)*num_every_file]
        f['pid']=object_seg[i*(num_every_file):(i+1)*num_every_file,:]
        f.close()

if __name__=='__main__':
    data,labels,seg=find_all_needto_aug_part()
    #f=h5py.File(hdf5_data_dir+'/object_part_aug.h5','w')
    #f['data']=data
    #f['label']=labels
    #f['pid']=seg
    # f.close()
    object_aug(data,labels,seg)
