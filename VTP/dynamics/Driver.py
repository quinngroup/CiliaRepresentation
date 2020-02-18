from argparse import ArgumentParser
from VTP.dynamics.module import VTP_D

#Parses arguments passed from command line
parser = ArgumentParser(description='DynamicsDriver')
parser.add_argument('--encoder', type=str, default=None, 
    help='choose the desired encoder architecture')
parser.add_argument('--decoder', type=str, default=None, 
    help='choose the desired decoder architecture')
args = parser.parse_args()

model = VTP_D()

def train():
    pass

def test():
    pass
