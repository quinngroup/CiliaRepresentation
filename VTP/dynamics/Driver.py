from argparse import ArgumentParser

#Parses arguments passed from command line
parser = ArgumentParser(description='DynamicsDriver')
parser.add_argument('--encoder', type=str, default=None, 
    help='choose the desired encoder architecture')
parser.add_argument('--decoder', type=str, default=None, 
    help='choose the desired decoder architecture')
