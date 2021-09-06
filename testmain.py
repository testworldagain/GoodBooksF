# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:49:10 2019

@author: testworld
"""

import tensorflow as tf
import datapreproc as dataset
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
from matplotlib.pylab import date2num
from tqdm import tqdm


class Configs(object):
    def __init__(self):
        # Attention
        self.num_heads = 8
        self.num_hidden = 16
        self.num_stacks = 3
        
        self.batch_size = 24
        self.feaNum = 8
        
        self.rnn_layer = 2
        self.seqlth = 50
        
        self.dec_hidden = 16
        self.output_num = self.feaNum
        
        self.learn_rate = 0.0001
        self.learn_decayrate = 0.96
        self.learn_decaystep = 500
        pass
    def saveConfig(self):
        curTime = time.localtime()
        Folder = "C:/Files/codeSVN_code/huobi/huobi/Logs/y{}_m{}_d{}_h{}_min{}_sec{}".format(curTime.tm_year,curTime.tm_mon,curTime.tm_mday,curTime.tm_hour,curTime.tm_min,curTime.tm_sec)
        if not os.path.exists(Folder):
            os.makedirs(Folder)
            pass
        with open(Folder+"configs.csv","w") as fid:
            fid.write("self.num_heads,{},attention机制的多头数\n".format(self.num_heads))
            fid.write("self.num_hidden,{},隐藏态个数\n".format(self.num_hidden))
            fid.write("self.num_stacks,{},attention层数\n".format(self.num_stacks))
            fid.write("self.batch_size,{},batchsize\n".format(self.batch_size))
            fid.write("self.rnn_layer,{},rnn层数\n".format(self.rnn_layer))
            fid.write("self.seqlth,{},seqlth\n".format(self.seqlth))
            pass
        pass
    pass


def mulitAtt(inputs,num_heads = 16,dropout_rate = 0.1,is_train = True,seed = 111):
    with tf.variable_scope("attention"):
        print("asdfasdfasfdasfas")
        print(inputs)
        
        Q = tf.layers.dense(inputs,inputs.shape[2].value,activation = tf.nn.relu)#[batch size,seqlth,numhid]
        K = tf.layers.dense(inputs,inputs.shape[2].value,activation = tf.nn.relu)#[batch size,seqlth,numhid]
        V = tf.layers.dense(inputs,inputs.shape[2].value,activation = tf.nn.relu)#[batch size,seqlth,numhid]
        
        Q_ = tf.concat(tf.split(Q,num_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,num_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,num_heads,axis=2),axis=0)
        
        outputs = tf.matmul(Q_,tf.transpose(K_,[0,2,1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.nn.softmax(outputs)  # num_heads*[]
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_train), seed=seed)
        outputs = tf.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [batch_size, seq_length, n_hidden]
        outputs += inputs  # [batch_size, seq_length, n_hidden]
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_train, name='ln',reuse=None)  # [batch_size, seq_length, n_hidden]
        return outputs

def feedforward(inputs, num_units=[2048, 512], is_training=True, seed=128):
    with tf.variable_scope("feedforward"):#, reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu,"use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln',reuse=None)  # [batch_size, seq_length, n_hidden]
        pass
    return outputs


def Transformer(inputs,num_hidden,num_heads,is_train,dropout_rate):
    enc = inputs
    with tf.variable_scope("Transformer"):
        enc = mulitAtt(enc,num_heads = num_heads,dropout_rate = dropout_rate,is_train = is_train)
        enc = feedforward(enc, num_units=[num_hidden*4, num_hidden], is_training=is_train)
        pass
    return enc
def MultiTransformer(stacks,inputs,num_hidden,num_heads,is_train,dropout_rate):
    output = inputs
    for i in range(stacks):
        with tf.variable_scope("stack{}".format(i)):
            output = Transformer(output,num_hidden,num_heads,is_train,dropout_rate)
        pass
    return output
def decoder(inputs,outputdim,seqlth,feanum):
    with tf.variable_scope("decoder"):
        outputs = tf.layers.conv1d(inputs,filters = seqlth,kernel_size = 1)
        outputs = tf.layers.conv1d(tf.transpose(outputs,perm=[0,2,1]),filters = feanum,kernel_size = 1)
        outputs = tf.squeeze(tf.sigmoid(tf.layers.conv2d(tf.expand_dims(outputs,axis = 3),filters = outputdim,kernel_size = (seqlth,feanum))))
    return outputs
def Pred(inputs,configs,istrain):
    output = tf.layers.conv1d(tf.transpose(inputs,perm=(0,2,1)),filters = configs.num_hidden,kernel_size = 1)
    output = tf.layers.batch_normalization(output,axis = -2,training = istrain)
    #output = tf.layers.conv1d(output,filters = c.num_hidden,kernel_size = 1)
    output = MultiTransformer(configs.num_stacks,output,configs.num_hidden,configs.num_heads,istrain,0.1)
    output = tf.layers.conv1d(tf.transpose(output,perm=(0,2,1)),filters = configs.dec_hidden,kernel_size = 1)
    output = decoder(output,configs.output_num,configs.seqlth,configs.feaNum)
    return output
    pass
def ActorLossFun(pred,target):
    loss = tf.reduce_mean(tf.square(pred[:,:-1] - target[:,:-1]),axis = 1)
    return loss
    pass
class BulitOptions(object):
    def __init__(self,config):
        self.data = tf.placeholder(dtype = tf.float32,shape=(config.batch_size,config.seqlth,config.feaNum),name = 'input')
        self.target = tf.placeholder(dtype = tf.float32,shape=(config.batch_size,config.output_num),name = 'target')
        self.global_step = tf.Variable(0,name = 'globalsetp',trainable = False)
        self.tf_istrain = tf.placeholder(dtype = tf.bool,name = "mode")
        self.pred = Pred(self.data,config,self.tf_istrain)
        self.loss = ActorLossFun(self.pred,self.target)
        
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(ops):
            lr = tf.train.exponential_decay(config.learn_rate, global_step = self.global_step, decay_steps = config.learn_decaystep, decay_rate = config.learn_decayrate, staircase=False, name=None)
            opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-08)
            grdvar = opt.compute_gradients(self.loss)
            clipgrd = [(tf.clip_by_norm(grv,1.0),std) for grv,std in grdvar if grv is not None ]
            self.optpos = opt.apply_gradients(clipgrd,global_step = self.global_step)
            self.lossopt = tf.reduce_mean(self.loss)
        pass
    pass

tf.reset_default_graph()
c = Configs()
dd = dataset.DataProc()
Data = dd.GenSampleType3(AimSymbol = 'usdt',TranSymbol = 'btc',Period = '1day')
opts = BulitOptions(c)
gloss = []
Session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
Session.run(tf.global_variables_initializer())
CurPair = []
for i in tqdm(range(500)):
    try:
        #Data = dd.GenSampleType3(AimSymbol = 'usdt',TranSymbol = 'btc',Period = '15min')
        traindata,target = dd.NextBatch(Data,batchsize = c.batch_size,DateLth = c.seqlth)
        feed_dict = {opts.data:traindata,
                     opts.target:target,
                     opts.tf_istrain:True}
        pred,_,loss = Session.run([opts.pred,opts.optpos,opts.lossopt],feed_dict = feed_dict)
        kp = pred
        kt = target
    except Exception as e:
        print(e)
        break
        pass
    CurPair = [pred,target]
    gloss.append(loss)
    if i% 1000 == 0:
        plt.plot(gloss)
        plt.show()
        pass
plt.plot(gloss)
plt.title("Loss")
plt.show()
import numpy as np
for i in range(7):
    plt.plot(kp[:,0],kt[:,0],'r.')
    plt.plot(0.01*np.array(range(100)),0.01*np.array(range(100)))
    plt.plot(0.01*np.array(range(100)),0.5*np.ones_like(range(100)))
    plt.plot(0.5*np.ones_like(range(100)),0.01*np.array(range(100)))
    plt.title(str(i))
    plt.show()

#kt = pd.DataFrame(kt,columns = ['amount','close','count','high','low','open','vol'])
#kp = pd.DataFrame(kp,columns = ['amount','close','count','high','low','open','vol'])

for j in range(c.batch_size):traindata,target = dd.NextBatch(Data,batchsize = c.batch_size,DateLth = c.seqlth)
feed_dict = {opts.data:traindata,opts.target:target,opts.tf_istrain:True}
pred,loss = Session.run([opts.pred,opts.lossopt],feed_dict = feed_dict)

"""
for i in range(7):
    plt.plot(pred[:,0],target[:,0],'r.')
    plt.plot(0.01*np.array(range(100)),0.01*np.array(range(100)))
    plt.plot(0.01*np.array(range(100)),0.5*np.ones_like(range(100)))
    plt.plot(0.5*np.ones_like(range(100)),0.01*np.array(range(100)))
    plt.title(str(i))
    plt.show()
pred = pd.DataFrame(pred,columns = ['amount','close','count','high','low','open','vol'])
target = pd.DataFrame(target,columns = ['amount','close','count','high','low','open','vol'])

"""