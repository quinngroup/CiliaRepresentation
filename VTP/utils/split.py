import numpy as np
from skimage.measure import block_reduce
import argparse
import plotly.express as px
import os


parser = argparse.ArgumentParser(description='array2vid')
parser.add_argument('--source', type=str, default='', metavar='s',help='Source of video')
parser.add_argument('--dest', type=str, default='', metavar='d',help='Destination of numpy array')
args = parser.parse_args()

assert args.source != '','Please specify video directory'

def split(x):
    y_1,y_2=np.split(x,2,axis=1)
    Y=np.split(y_1,2,axis=2)+np.split(y_2,2,axis=2)
    return Y


for subdir, dirs, files in os.walk(args.source):
    for file in files:
        location=os.path.join(subdir, file)
        vid=np.load(location)
        destName=args.dest+file[:-4]
        destName = re.sub('[^0-9a-zA-Z/-]+', '_', destName)
        print(destName)
        for i,v in enumerate(split(downsample(vid))):
            for j,u in enumerate(split(v)):
                np.save(args.dest+"_"+str(i)+str(j),u)