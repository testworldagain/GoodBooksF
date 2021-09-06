# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:16:10 2019

@author: testworld
"""



from hbapi import HuobiServices as hs
import pandas as pd
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from numba import jit
class DataProc():
    def __init__(self,sourPath = 'C:/Files/Datas/SumLogs',resultPath = 'C:/Files/Datas/PreProc',fillna = 'line'):
        """
        fillna:'line' 'prev' 'next' 'mean' // 分别表示取缺省值的线性插值，前一个有效值，后一个有效值，前后有效值的均值
        """
        self.sourPath   = sourPath
        self.resultPath = resultPath
        self.fea = ['amount', 'close', 'count', 'high', 'low', 'open','vol', 'avg']
        self.feaName = ['amount','close','count','high','low','open','vol']
        self.SampleType = 0
        pass
    def __GetDataFn(self,Symbol,Period):#获取数据的文件路径
        return self.sourPath + '/' + Symbol + '-' + Period + '.csv'
        pass
    def __IsLine(self,data,fea = 'id',stype = 1):#判断是否存在缺失值
        if stype == 1:
            if len(data[fea].diff().unique()) > 2:
                return False
        elif stype == 2:
            if len(pd.DataFrame(data.index).diff()['id'].unique()) > 2:
                return False
            
            pass
        return True
    def GetData(self,Symbol,Period):#获取数据
        Data = pd.read_csv(self.__GetDataFn(Symbol,Period))
        if not self.__IsLine(Data):
            print('Current load data:\t' + Symbol +"::" + Period +"::" + 'data error\n')
            raise 'data error'
            return None
        print('Current load data:\t' + Symbol +"\t" + Period+'\t success!')
        
        
        #tdata = np.exp(np.array(Data))
        tdata = np.array(Data) + 1
        tdata = tdata[:-1,:]/(tdata[1:,:] + tdata[:-1,:])
        tData = pd.DataFrame(tdata)
        tData.columns = Data.columns
        return tData
    def GetData2(self,Symbol,Period):#获取数据
        print(self.__GetDataFn(Symbol,Period))
        Data = pd.read_csv(self.__GetDataFn(Symbol,Period),index_col = 5)[self.feaName]
        if not self.__IsLine(Data,stype = 2):
            print('Current load data:\t' + Symbol +"::" + Period +"::" + 'data error\n')
            raise 'data error'
            return None
        print('Current load data:\t' + Symbol +"\t" + Period+'\t success!')
        return Data
    
    def __FeatureData_Type1(self,data):#第一类特征处理
        #增加一个均价
        data['avg'] = data['vol']/data['amount']
        data = data.fillna(method = 'ffill')
        return data
    def __FeatureData_Type2(self,data,Symbol):#第一类特征处理
        #增加一个均价
        data['avg'] = data['vol']/data['amount']
        data = data.fillna(method = 'ffill')
        data.columns = [Symbol+'_'+i for i in data.columns]
        return data
    #@jit
    def GenSampleType1(self,Symbol = 'btmusdt',Period = '5min'):#生成样本数据
        self.SampleType = 1        
        Data = self.GetData(Symbol,Period)
        feaData = self.__FeatureData_Type1(Data)
        return feaData
    def GenSampleType2(self,AimSymbol = 'usdt',TranSymbol = ['btm','btc','eth','eos','ada'],Period = '60min'):#生成样本数据
        self.SampleType = 2
        Symbols = [i+AimSymbol for i in TranSymbol]
        feaData = [self.__FeatureData_Type2(self.GetData2(i,Period),i) for i in Symbols]
        self.feaName = self.feaName + ['avg']
        return feaData,Symbols
    def GenSampleType3(self,AimSymbol = 'usdt',TranSymbol = 'btc',Period = '15min'):#生成样本数据
        Symbols = TranSymbol+AimSymbol
        data = self.GetData2(Symbols,Period).values
        self.keepData = pd.read_csv(self.__GetDataFn(Symbols,Period)).id%(7*24*3600)/(24*3600)
        #return data
        sumdata = data[:-1,:] + data[1:,:]
        sumdata += np.ones_like(sumdata)
        sumdata = data[1:,:]/sumdata
        sumdata = np.concatenate([sumdata,self.keepData.values[1:].reshape((-1,1))],axis = 1)
        self.SampleType = 3
        self.curPos = 0
        return sumdata
        pass
    def SampleAddPointFlag(self,Data,symbols,step = 1,minGain = 0.005,maxRisk = -0.005):
        closeValue = Data[[i+'_close' for i in symbols]]
        highValue = Data[[i+'_high' for i in symbols]]
        lowValue = Data[[i+'_low' for i in symbols]]
        gain = (highValue.iloc[step:].values - closeValue.iloc[:-step].values)/closeValue.iloc[:-step].values
        risk = (lowValue.iloc[step:].values - closeValue.iloc[:-step].values)/closeValue.iloc[:-step].values
        
        sortgain = np.fliplr(np.c_[gain,[minGain]*gain.shape[0]].argsort(axis = 1))# 收益反转，越前，收益概率越大
        sortrisk = np.c_[risk,[maxRisk]*risk.shape[0]].argsort(axis = 1)
        
        gainisk  = np.c_[Data.values[:-step,:],sortgain,sortrisk]
        newColumns = list(Data.columns) + ['gain'+str(i) for i in range(len(symbols)+1)]+ ['risk'+str(i) for i in range(len(symbols)+1)]
        trainData = pd.DataFrame(gainisk,columns = newColumns,index = Data.index[step:])
        self.symbols = symbols
        return trainData
    
    def NextBatch(self,data,batchsize = 10,EncodeStep = 7,DecodeStep = 7,DateLth = 30):
        if self.SampleType == 1:
            startIdx = np.random.randint(0,data.shape[0] - EncodeStep - DecodeStep,batchsize)
            InputX = []
            TargetY = []
            TargetY_t = []
            InputTarget = []
            for i in startIdx:
                InputX.append(data.iloc[range(i,i+EncodeStep,1)][self.fea])
                InputTarget.append(np.array(data.iloc[i+EncodeStep][self.fea]))
                TargetY.append(data.iloc[range(i+EncodeStep+1,i+EncodeStep+DecodeStep+1,1)][self.fea])
                TargetY_t.append(data.iloc[range(i+EncodeStep+1,i+EncodeStep+DecodeStep+1,1)]['id'])
            #print(startIdx)
            _InputX = []
            _TargetY = []
            for i in range(EncodeStep):
                
                _InputX.append(np.array([j.iloc[i] for j in InputX]))
            for i in range(DecodeStep):
                _TargetY.append(np.array([j.iloc[i] for j in TargetY]))
            return _InputX,np.array(InputTarget),_TargetY,TargetY_t
        elif self.SampleType == 2:
            startIdx = np.random.randint(0,data.shape[0]-DateLth,batchsize)
            feaColname = [[symbol+'_'+fea for fea in self.feaName] for symbol in self.symbols]
            tarGainCol = ['gain'+str(ifea) for ifea in range(len(self.symbols)+1)]
            tarRiskCol = ['risk'+str(ifea) for ifea in range(len(self.symbols)+1)]            
            Samples = []
            Targets = []
            for i in startIdx:
                tmpInput = []
                tmpTarget = []
                for symfea in feaColname:
                    tmpInput.append(data[symfea].iloc[i:(i+DateLth)].values)
                    pass
                tmpTarget.append(data[tarGainCol].iloc[i+DateLth-1].values)
                tmpTarget.append(data[tarRiskCol].iloc[i+DateLth-1].values)
                
                
                Samples.append(np.array(tmpInput))
                Targets.append(np.array(tmpTarget))
                pass
            return np.array(Samples),np.array(Targets)#[batchsize,seqLth,inputSeqLth,inputsize]#[batchsize,targetNum,targetPoints]
            pass
        elif self.SampleType == 3:
            traindata = []
            target = []
            nextCurPos = self.curPos + 1
            for i in range(batchsize):
                traindata.append(np.reshape(data[self.curPos:(self.curPos+DateLth),:],(1,DateLth,data.shape[1])))
                target.append(data[self.curPos+DateLth:self.curPos+DateLth+1,:])
                self.curPos += 1
                pass
            self.curPos = nextCurPos
            
            np.concatenate(traindata,axis = 0)
            np.concatenate(target,axis = 0)
            return np.concatenate(traindata,axis = 0), np.concatenate(target,axis = 0)
            pass
        
        
        pass


if __name__ == '__main__':
    ### type1 sample test
    #dd = DataProc()
    #Data = dd.GenSampleType1()
    #InputX,InputTarget,TargetY,TargetY_t = dd.NextBatch(Data,batchsize = 10,EncodeStep = 7,DecodeStep = 7)
    
    ### type2 sample test
    #dd = DataProc()
    #Data,symbols = dd.GenSampleType2()
    #tData = pd.concat(Data,axis = 1).dropna(axis = 0)
    #sample = dd.SampleAddPointFlag(tData,symbols,step = 1)
    #traindata,target = dd.NextBatch(sample)
    ### type3 sample test
    dd = DataProc()
    Data = dd.GenSampleType3()
    traindata,target = dd.NextBatch(Data)

"""
[197]: a = np.array([sample.values[:10,:],sample.values[:10,:]])

a = np.array([sample.values[:10,1:4],sample.values[:10,2:5]])
"""



