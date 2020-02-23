from pyoptflow import HornSchunck as hs
import pyoptflow
import cv2
import numpy as np
import os
from of_utils import *
#from pyflow_module import pyflow as pyflow

def horn_schunck(arr,frames):
    '''
    frames: number of frames to compute (truncation)
    arr: original video array 

    computes traditional implementation of Horn-Schunck optical flow
    '''
    
    optflow = np.empty([frames,2,arr.shape[1],arr.shape[2]]).astype('float16')
    curl_arr = np.empty([frames,arr.shape[1],arr.shape[2]]).astype('float16')
    def_arr = np.empty([frames,arr.shape[1],arr.shape[2]]).astype('float16')
    for i in range(frames):
            u,v = hs(arr[i],arr[i+1])
            curl_arr[i] = curl(u,v)
            def_arr[i] = curl(u,v)
            optflow[i] = [u,v]
    return optflow, curl

def pyflow(arr,frames):
    '''
    arr: video array
    frames: number of frames to compute (truncation)
    
    Computes the pyflow pased implementation of coarse to fine optical flow 
    '''

    optflow = np.empty([frames,2,arr.shape[1],arr.shape[2]]).astype('float16')
    curl_arr = np.empty([frames,arr.shape[1],arr.shape[2]]).astype('float16')
    def_arr = np.empty([frames,arr.shape[1],arr.shape[2]]).astype('float16')
    for i in range(frames):
        u,v,im2W = pyflow.coarse2fine_flow(arr[i], arr[i+1], alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7,nInnerFPIterations=1, nSORIterations=30,colType=1)
        curl_arr[i] = curl(u,v)
        def_arr[i] = curl(u,v)
        optflow[i] = [u,v]
    return optflow,curl_arr

def farneback(arr,frames):
    '''
    arr: og vid array
    frames: number of frames to compute (truncation)
    
    Computes the farneback pased implementation of coarse to fine optical flow 
    '''
    
    optflow = np.empty([frames,2,arr.shape[1],arr.shape[2]]).astype('float16')
    curl_arr = np.empty([frames,arr.shape[1],arr.shape[2]]).astype('float16')
    def_arr = np.empty([frames,arr.shape[1],arr.shape[2]]).astype('float16')
    for i in range(frames):
        of = cv2.calcOpticalFlowFarneback(arr[i],arr[i+1],None,0.5,3,15,3,5,1.2,0)
        optflow[i] = [of[...,0],of[...,1]]
        curl_arr[i] = curl(of[...,0],of[...,1])
        def_arr[i] = curl(u,v)
    return optflow,curl_arr
