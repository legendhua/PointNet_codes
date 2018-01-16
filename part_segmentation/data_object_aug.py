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

hdf5_data_dir=os.path.join(BASE_DIR,'./hdf5_data')
train_file_list=provider.getDataFiles(os.path.join(hdf5_data_dir,'train_hdf5_file_list.txt'))
num_train_file=len(train_file_list)
#渲染文件
def output_point_cloud(data,out_file):
    with open(out_file,'w') as f:
        for i in range(data.shape[0]):
            f.write('v %f %f %f\r\n'%(data[i][0],data[i][1],data[i][2]))
def find_all_needto_aug():
    '''
    提取出源文件中符合条件的数据
    '''
    #初始化一个跟数据格式一样的数据原本
    object_data=np.zeros((1,2048,3))
    object_labels=np.zeros((1),np.int32)
    object_seg=np.zeros((1,2048),np.int32)
    count=0#统计所有数据个数
    #提取出所有符合标签的数据
    for i in range(num_train_file):
        print('load the num '+str(i)+' train file')
        cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[i])
        cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)
        for nlabel in range(cur_data.shape[0]):
            if cur_labels[nlabel]==10 or cur_labels[nlabel]==13 :
                count+=1
                object_data=np.vstack((object_data,cur_data[nlabel,:,:].reshape(1,2048,3)))
                object_labels=np.vstack((object_labels,cur_labels[nlabel]))
                object_seg=np.vstack((object_seg,cur_seg[nlabel,:]))
                #上述object_data、object_lables、object_seg即包含了所有符合条件的数据
    idx=np.random.randint(count,size=1)#随机将第一个初始化的数据赋成数据中的一个
    object_data[0,:,:]=cur_data[idx,:,:]#将数据第一个0项初始化为任意一项
    object_labels[0]=cur_labels[idx]
    object_seg[0,:]=cur_seg[idx,:]
    return object_data,object_labels,object_seg

def object_aug(object_data,object_labels,object_seg):
    object_data1=provider.jitter_point_cloud(object_data,sigma=0.001,clip=0.005)#给数据加噪一次
    object_data2=provider.jitter_point_cloud(object_data,sigma=0.001,clip=0.005)#给数据加噪两次
    object_data=np.vstack((object_data1,object_data2))
    object_labels=np.vstack((object_labels,object_labels))
    object_seg=np.vstack((object_seg,object_seg))
    
	#合并所有数据
    
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
        f=h5py.File(hdf5_data_dir+'/object_aug'+str(i)+'.h5','w')
        f['data']=object_data[i*(num_every_file):(i+1)*num_every_file,:,:]
        f['label']=object_labels[i*(num_every_file):(i+1)*num_every_file]
        f['pid']=object_seg[i*(num_every_file):(i+1)*num_every_file,:]
        f.close()

if __name__=='__main__':
    data,labels,seg=find_all_needto_aug()
    object_aug(data,labels,seg)
