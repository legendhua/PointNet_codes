#-*- coding:utf-8 -*-
# ensemble测试代码：选择多个训练好的模型
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls_ensemble', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--first_model_path', default='log_ensemble/shallow3/model163.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--second_model_path', default='log_ensemble/shallow4/model290.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
#add 2018-0109
FIRST_MODEL_PATH = FLAGS.first_model_path
SECOND_MODEL_PATH = FLAGS.second_model_path
#end
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate_first_model(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_first_model(pointclouds_pl, is_training_pl)#此处修改为对应的模型
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, FIRST_MODEL_PATH)
    log_string("FIRST Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}
    #add 2018-0109
    pred_first= np.zeros((1,NUM_CLASSES))#add one all-zeros column
    #end
    total_pred_first_model = eval_one_epoch(sess, ops, num_votes, pred_first)
    return total_pred_first_model

def evaluate_second_model(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_second_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, SECOND_MODEL_PATH)
    log_string("SECOND Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}
    #add 2018-0109
    pred_second= np.zeros((1,NUM_CLASSES))#add one all-zeros column
    #end
    total_pred_second_model = eval_one_epoch(sess, ops, num_votes, pred_second)
    return total_pred_second_model 
   
def eval_one_epoch(sess, ops, num_votes, total_test_pred_model):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx
            
            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
            
            for vote_idx in range(num_votes):
                rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                                  vote_idx/float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)
                batch_pred_sum += pred_val
                #add 2018-0109
                total_test_pred_model = np.row_stack((total_test_pred_model,pred_val))
                #end
    sess.close()
    return total_test_pred_model



    
def test_ensemble(num_votes):
    with tf.Graph().as_default():
        total_pred_first_value=evaluate_first_model(num_votes)
    with tf.Graph().as_default(): 
        total_pred_second_value=evaluate_second_model(num_votes)
    #ensemble_pred = np.maximum(total_pred_first_value,total_pred_second_value)
    ensemble_pred = (total_pred_first_value + total_pred_second_value)/2
    #ensemble_pred = total_pred_first_value
    pred_val = np.argmax(ensemble_pred, 1)#当前所有模型预测结果取最大值
    test_size = pred_val.shape[0]-1#减去第一个全0行
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_correct = 0
    start = 1
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_label = np.squeeze(current_label)
        current_size = current_label.shape[0]
        end = start+current_size
        correct = np.sum(pred_val[start:end] == current_label[0:current_size])
        for i in range(current_size):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[start+i] == l)
        start = end
        total_correct +=correct
        #计算平均类别精确度
        
    log_string('eval accuracy: %f' % (total_correct / float(test_size)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
if __name__=='__main__':
    test_ensemble(num_votes=1)
    LOG_FOUT.close()
