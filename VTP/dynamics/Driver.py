from argparse import ArgumentParser
from VTP.dynamics.module import VTP_D

#Parses arguments passed from command line
parser = ArgumentParser(description='DynamicsDriver')
parser.add_argument('--encoder', type=str, default=None, 
    help='choose the desired encoder architecture')
parser.add_argument('--decoder', type=str, default=None, 
    help='choose the desired decoder architecture')
args = parser.parse_args()

model = VTP_D(encoder=args.encoder, decoder=args.decoder)

train_loader = None
test_loader = None
optimizer = None

def train():
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_function(output, data)
        train_loss += loss.item()
        optimizer.step()

def test():
    model.eval()
