import sys
import os
# sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__))))
# sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), './optimizer'))
sys.path.append('.')
from os.path import join
import json
import numpy as np
import tensorflow as tf
from utils.tfVars import varInit
from utils.optimizer import Optimizer
from utils.loss import LossFunc
tf.enable_eager_execution()

baseDir = './test_tw'
handposeRoot = join(baseDir, 'onlyHandWorldCoordinate_uvd.json')
recordDir = join('./self', 'record_multi', '220405_hand')
BATCH = 20
REF_CAM = 2

def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def getPredicedPoses():
    # data : (cam_idx, frame_idx, joint_idx, 3) ~ (3, 116, 21, 3)
    #data = np.load(handposeRoot)
    _assert_exist(handposeRoot)
    with open(handposeRoot, 'r') as fi:
        data = json.load(fi)
    return data

def gen(batch_size, ref_cam):
    source = getPredicedPoses()
    handpose_0 = np.array(source['0_%d'%ref_cam])
    handpose_1 = np.array(source['1_%d'%ref_cam])
    handpose_2 = np.array(source['2_%d'%ref_cam])
    handpose_set = np.array([handpose_0, handpose_1, handpose_2])
    refpose_0 = np.array(source['0_0'])
    refpose_1 = np.array(source['1_1'])
    refpose_2 = np.array(source['2_2'])
    refpose_set = np.array([refpose_0, refpose_1, refpose_2])
    len = handpose_set.shape[1]
    numBatches = len // batch_size

    for i in range(numBatches):
        handpose_batch = handpose_set[:, i*batch_size:(i+1)*batch_size].transpose(1,0,2,3)
        refpose_batch = refpose_set[:, i*batch_size:(i+1)*batch_size].transpose(1,0,2,3)

        yield (handpose_batch, refpose_batch)
    
def poseMV():
    ds = tf.data.Dataset.from_generator(lambda: gen(BATCH, REF_CAM), (tf.float32, tf.float32), (tf.TensorShape([None, 3, 21, 3]), tf.TensorShape([None, 3, 21, 3])))
    return ds

if __name__=='__main__':
    initVars, refpose = varInit(REF_CAM)
    print(initVars[0][5:10])
    lossfunc = LossFunc(initVars, refpose, REF_CAM)
    loss = lossfunc.getlossFunc
    optm = Optimizer(loss, initVars)
    resetOptm = tf.variables_initializer(optm.optimizer.variables())
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.__enter__()
    tf.global_variables_initializer().run()
    sess.run(resetOptm)
    optm.runOptm(sess, 1)
    print(initVars[0][5:10])
    sess.close()
