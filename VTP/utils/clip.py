import argparse
import numpy as np
from random import randint
import os
'''
Generates a clipped dataset from a video dataset focusing on a clip of n frames of each video
Treats 0th axis as time, 1st axis as height, 2nd as width

@author Quinn Wyner
'''

def clipper(filename, clipLength):
    data = np.load(args.source+filename, mmap_mode='r')
    
    clipStart = randint(0, data.shape[0] - clipLength)
    
    return data[clipStart:clipStart+clipLength]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clippedDataGen')
    parser.add_argument('--source', type=str, default='./', metavar='s',
                        help = 'Name of numpy array file to load (default=\'.\')')
    parser.add_argument('--clipLength', type=int, default=40, metavar='cl',
                        help = 'Length of a single clip (default=40)')
    parser.add_argument('--dest', type=str, default='clipped/', metavar='d',
                        help = 'Directory in which to save files (default=\'clipped/\')')
    args = parser.parse_args()
    
    for file in os.listdir(args.source):
        if file.endswith('.npy'):
            np.save(args.dest + file, clipper(file, args.clipLength))