import skvideo.io
import numpy as np
import argparse
import os
import re

'''
Converts numpy arrays to videos

@author Meekail Zain
'''
parser = argparse.ArgumentParser(description='array2vid')
parser.add_argument('--source', type=str, default='', metavar='s',help='Source of video')
parser.add_argument('--dest', type=str, default='', metavar='d',help='Destination of numpy array')
args = parser.parse_args()

assert args.source != '','Please specify video directory'

for subdir, dirs, files in os.walk(args.source):
    for file in files:
        if file[-4:]=='.avi':
            location=os.path.join(subdir, file)
            print(location)
            destName=args.dest+file[:-4]
            destName = re.sub('[^0-9a-zA-Z/-]+', '_', destName)
            print(destName)
            if not os.path.exists(destName+'.npy'):
                videodata = skvideo.io.vread(location)
                videodata=videodata[:,:,:,0]
                print(videodata.shape)
                np.save(destName,videodata)
