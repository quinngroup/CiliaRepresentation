import argparse
import numpy as np
from numpy.random import default_rng
from random import randint
import os
'''
Generates a clipped dataset from a video dataset focusing on a clip of n frames of each video
Treats 0th axis as time, 1st axis as height, 2nd as width

@author Quinn Wyner
@author Meekail Zain
'''

def clipper(filename, clipLength,mmap,rng):

    if mmap:
        data = np.load(args.source+filename, mmap_mode='r')
    else:
        data = np.load(args.source+filename)
    assert data.shape[0]>=clipLength, "Requested clip would be longer than source video"

    randomFrames = rng.choice(data.shape[0], size=clipLength, replace=False)
    output=np.zeros((clipLength,data.shape[1],data.shape[2]))
    count=0
    for i in randomFrames:
        output[count] = data[i]
        count++
        
    return output
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clippedDataGen')
    parser.add_argument('--source', type=str, default='./', metavar='s',
                        help = 'Name of numpy array file to load (default=\'.\')')
    parser.add_argument('--clipLength', type=int, default=40, metavar='cl',
                        help = 'Length of a single clip (default=40)')
    parser.add_argument('--dest', type=str, default='clipped/', metavar='d',
                        help = 'Directory in which to save files (default=\'clipped/\')')
    parser.add_argument('--mmap', type=bool, default=True, metavar='m',
                        help = 'Flag for whether or not to use read-only memory mapping')
    
    args = parser.parse_args()

    assert args.source != '','Please specify video directory'
    
    rng = default_rng()

    for file in os.listdir(args.source):
        if file.endswith('.npy'):
            np.save(args.dest + file, clipper(file, args.clipLength,args.mmap,rng))