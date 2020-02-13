import skvideo.io
import numpy as np
import argparse
import os

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
        if not file[-4:]=='.avi'
        location=os.path.join(subdir, file)
        print(location)
        videodata = skvideo.io.vread(location)
        print(videodata.shape)
        destName=args.dest+file[:-4]
        while os.path.exists(destName):
            destName+="_"+subdir
        np.save(destName,videodata.astype(np.uint8))
        