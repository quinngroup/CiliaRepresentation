from pyoptflow import HornSchunck as hs
import pyoptflow
import cv2
import numpy as np
import os
from pyflow_module import pyflow as pyflow

def horn_schunck(arr,frames):
    '''
    frames: number of frames to compute (truncation)
    arr: original video array 

    computes traditional implementation of Horn-Schunck optical flow
    '''
    
    optflow = np.empty([frames,2,arr.shape[1],arr.shape[2]]).astype('float16')
    for i in range(frames):
            u,v = hs(arr[i],arr[i+1])
            optflow[i] = [u,v]
    return optflow

def pyflow(arr,frames):
    '''
    arr: video array
    frames: number of frames to compute (truncation)
    
    Computes the pyflow pased implementation of coarse to fine optical flow 
    '''

    optflow = np.empty([frames,2,arr.shape[1],arr.shape[2]]).astype('float16')
    for i in range(frames):
        u,v,im2W = pyflow.coarse2fine_flow(arr[i], arr[i+1], alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7,nInnerFPIterations=1, nSORIterations=30,colType=1)
        optflow[i] = [u,v]
    return optflow

def farneback(arr,frames):
    '''
    arr: og vid array
    frames: number of frames to compute (truncation)
    
    Computes the farneback pased implementation of coarse to fine optical flow 
    '''
    
    optflow = np.empty([frames,2,arr.shape[1],arr.shape[2]]).astype('float16')
    for i in range(frames):
        of = cv2.calcOpticalFlowFarneback(arr[i],arr[i+1],None,0.5,3,15,3,5,1.2,0)
        optflow[i] = [of[...,0],of[...,1]]
    return optflow

def main():
    dpath = 'data/'
    optpath = 'flows/'
    for vid in os.listdir(data_path):
        if not os.path.exists(out_path+vid):
            arr = np.load(data_path+vid)
#            optflow = #method 
            np.save(out_path+vid,optflow)
            print("saved vector to: ",out_path)
