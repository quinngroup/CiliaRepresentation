from argparse import ArgumentParser
from math import ceil
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from NVP import NVP
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import transforms
from utils import Datasets
import apex
from apex import amp
import os
'''
Appearance Module Driver

@author Quinn Wyner
'''

startTime = time.time()

#Parses arguments passed from command line
parser = ArgumentParser(description='AppearanceDriver')
parser.add_argument('--batch_size', type=int, default=128, metavar='b',
                    help='number of elements per minibatch')
parser.add_argument('--beta', type=float, default=1.0, metavar='b',
                    help='sets the value of beta for a beta-vae implementation')
parser.add_argument('--dataset', type=str, default='nwd', metavar='D',
                    help='dataset selection for training and testing')
parser.add_argument('--dbscan', action='store_true', default= False,
                    help='to run dbscan clustering')      
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--failCount', type=str, default='r', metavar='fc',
                    help='determines how to reset early-stopping failed epoch counter. Options are \'r\' for reset and \'c\' for cumulative')
parser.add_argument('--gamma', type = float, default=.05, metavar='g',
                    help='Pseudo-loss weight')
parser.add_argument('--graph', action='store_true', default= False,
                    help='flag to determine whether or not to run automatic graphing')      
parser.add_argument('--input_height', type=int, default=128, metavar='ih',
                    help='height of each patch')
parser.add_argument('--input_length', type=int, default=128, metavar='il',
                    help='length of each patch')
parser.add_argument('--load', type=str, default='', metavar='l',
                    help='loads the weights from a given filepath')
parser.add_argument("--local_rank", type=int, default=0,
                    help='parameter for distributed training provided by launch script')
parser.add_argument('--log', type=str, default='!', metavar='lg',
                    help='flag to determine whether to use tensorboard for logging. Default \'!\' is read to mean no logging')      
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--logvar_bound', type=float, default=-1.0, metavar='lb',
                    help='Lower bound on logvar (default: -1.0)')
parser.add_argument('--lr', type = float, default=1e-4, metavar='lr',
                    help='learning rate')
parser.add_argument('--lsdim', type = int, default=10, metavar='ld',
                    help='sets the number of dimensions in the latent space. should be >1. If  <3, will generate graphical representation of latent without TSNE projection')
                    #current implementation may not be optimal for dims above 4
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--noEarlyStop', action='store_true', default=False,
                    help='disables early stopping')
parser.add_argument('--patience', type = int, default = 10, metavar='pat',
                    help='patience value for early stopping')
parser.add_argument('--plr', type = float, default=4e-6, metavar='plr',
                    help='pseudoinput learning rate')
parser.add_argument('--print_pseudos', type = int, default=0, metavar='pp',
                    help='Plot pseudos. Controls the number of pseudo inputs to be displayed')
parser.add_argument('--pseudos', type=int, default=10, metavar='p',
                    help='Number of pseudo-inputs (default: 10)')
parser.add_argument('--reg2', type = float, default=0, metavar='rg2',
                    help='coefficient for L2 weight decay')
parser.add_argument('--repeat', action='store_true', default=False,
                    help='determines whether to enact further training after loading weights')
parser.add_argument('--save', type=str, default='', metavar='s',
                    help='saves the weights to a given filepath')
parser.add_argument('--schedule', type = int, default=-1, metavar='sp',
                    help='use learning rate scheduler on loss stagnation with input patience')
parser.add_argument('--seed', type=int, default=None, metavar='s',
                    help='manual random seed (default: None)')
parser.add_argument('--source', type=str, default='data', metavar='S',
                    help='directory containing source files')
parser.add_argument('--test_split', type=float, default=.2, metavar='ts',
                    help='portion of data reserved for testing')
parser.add_argument('--tolerance', type = float, default=.1, metavar='tol',
                    help='tolerance value for early stopping')
parser.add_argument('--tsne', action='store_true', default=False,
                    help='Uses TSNE projection instead of UMAP.')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
                    
args = parser.parse_args()






#Runs model on GPU by default, but on CPU if GPU not available
cuda = not args.no_cuda and torch.cuda.is_available()
if cuda:
    device = 'cuda'
    kwargs = {'num_workers': 1, 'pin_memory': True}
    #Tests CUDA compatibility before instantiating dataset
    with torch.cuda.device(0):
        torch.tensor([1.]).cuda()
else:
    device = 'cpu'
    kwargs = {}
    
#Determine whether the process is being run as distributed
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

torch.backends.cudnn.benchmark = True

#Manually sets random seed if requested
if args.seed:
    torch.manual_seed(args.seed)
    
writer=None
if(args.log!='!'):
    if(args.log=='$'):
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(log_dir=args.log)


model = NVP(args.input_length, args.batch_size, args.lsdim, args.pseudos, args.beta, args.gamma, device, args.logvar_bound).cuda()
optimizer = torch.optim.Adam([{'params': model.vae.parameters()},
                        {'params': model.pseudoGen.parameters(), 'lr': args.plr}],
                        lr=args.lr, weight_decay=args.reg2)

#Amp optional fp optimization
model, optimizer = amp.initialize(model, optimizer, opt_level="O0")

if args.distributed:
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank)

#Construct datasets
if args.dataset == 'od':
    data = Datasets.overlapDataset(args.source, args.clip_length)
elif args.dataset == 'nd':
    data = Datasets.nonOverlapDataset(args.source, args.clip_length)
elif args.dataset == 'frd':
    data = Datasets.frameDataset(args.source, transforms.ToTensor())
elif args.dataset == 'nwd':
    data = Datasets.nonOverlapWindowDataset(args.source, args.input_height, args.input_length, transforms.ToTensor())
elif args.dataset == 'owd':
    data = Datasets.overlapWindowDataset(args.source, args.input_height, args.input_length, transforms.ToTensor())
elif args.dataset == 'ncd':
    data = Datasets.nonOverlapClipDataset(args.source, args.clip_length, args.input_height, args.input_length)
elif args.dataset == 'ocd':
    data = Datasets.overlapClipDataset(args.source, args.clip_length, args.input_height, args.input_length)
else:
    data = None

testSize = ceil(args.test_split * len(data))
trainSize = len(data) - testSize
trainSet, testSet = random_split(data, [trainSize, testSize])
train_sampler=None
test_sampler=None

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainSet)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testSet)

train_loader = DataLoader(
    trainSet, 
    batch_size=args.batch_size, 
    shuffle=(train_sampler is None),
    num_workers=args.workers, 
    pin_memory=True, 
    sampler=train_sampler)

test_loader = DataLoader(
    testSet,
    batch_size=args.batch_size, 
    shuffle=False,
    num_workers=args.workers, 
    pin_memory=True,
    sampler=test_sampler)




stopEarly = False
failedEpochs=0
lastLoss = 0

scheduler=None
if(args.schedule>0):
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=args.schedule)


scale=1
if args.distributed:
    scale=2


def train(epoch):
    if args.distributed:
        train_sampler.set_epoch(epoch)
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)

        #For model module access, must reference model.module 
        #if distributed to get through the distributed wrapper class
        if args.distributed:
            MODEL=model.module
        else:
            MODEL=model
            
        pseudos=MODEL.pseudoGen.forward(MODEL.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
        recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
        loss = MODEL.loss_function(recon_batch, data, mu, logvar, z,pseudos,recon_pseudos, p_mu, p_logvar, p_z)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        train_loss += loss.item()
        genLoss = MODEL.loss_function(recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, gamma=0).item() / len(data)
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.local_rank==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tGenLoss: {:.6f}'.format(
                epoch, scale*batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                genLoss))
        step=epoch*len(train_loader)+batch_idx
        if(args.log!='!'):
            per_item_loss=loss.item()/len(data)
            writer.add_scalar('item_loss',per_item_loss,global_step=step)

    '''Note in the "average loss" section we multiply by the constant scale 
    since if the model is run in distributed mode, each singular gpu will only 
    process *half* the data but would otherwise normalize by the length of the entire dataset    
    '''
    if args.local_rank==0:
        print('====> Epoch: {} Average loss (main gpu): {:.4f}'.format(
              epoch, scale*train_loss / len(train_loader.dataset)))
    if(args.schedule>0):
          scheduler.step(scale*train_loss / len(train_loader.dataset))

def test(epoch, max, startTime):
    model.eval()
    test_loss = 0
    gen_loss = 0
    global lastLoss
    global failedEpochs
    global stopEarly
    zTensor = torch.empty(0,args.lsdim).to(device)
    if args.distributed:
        MODEL=model.module
    else:
        MODEL=model
    pseudos=MODEL.pseudoGen.forward(MODEL.idle_input).view(-1,1,args.input_length,args.input_length).to(device)             
    recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += MODEL.loss_function(recon_batch, data, mu, logvar,z,pseudos,recon_pseudos, p_mu, p_logvar, p_z).item()
            gen_loss += MODEL.loss_function(recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, gamma=0).item()
            zTensor = torch.cat((zTensor, z), 0)
    if (args.dbscan == True) :
        zScaled = StandardScaler().fit_transform((torch.Tensor.cpu(zTensor).numpy())) #re-add StandardScaler().fit_transform
        db = DBSCAN(eps= 0.7, min_samples= 3).fit(zScaled)
        print(db.labels_)
        labelTensor = db.labels_
    test_loss /= len(test_loader.dataset)/scale
    gen_loss /= len(test_loader.dataset)/scale
    if args.local_rank==0:
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Generation loss: {:.4f}'.format(gen_loss))
    if(epoch == 1):
        lastLoss = test_loss
    elif not args.noEarlyStop:
        if lastLoss-test_loss < args.tolerance:
            failedEpochs += 1
            if failedEpochs >= args.patience:
                stopEarly = True
                
                epoch = max
        elif args.failCount == 'r':
            failedEpochs = 0
    if(epoch == max and args.local_rank==0):
        if(args.save != ''):
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()
                        }, args.save)
        print("--- %s seconds ---" % (time.time() - startTime))
        cmap = colors.ListedColormap(['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe'])
        
        #Handling different dimensionalities
        if(args.graph):
            if (args.lsdim < 3) :
                z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
                z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
                scatterPlot = plt.scatter(z1, z2, s = 4) #Regular 2dim plot, RE-ADD CMAP = CMAP
            elif (args.lsdim == 3) :
                fig=plt.figure()
                ax=fig.gca(projection='3d')
                z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
                z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
                z3 = torch.Tensor.cpu(zTensor[:, 2]).numpy()
                scatterPlot = ax.scatter(z1, z2, z3, s = 4) #Regular 3dim plot
            elif args.tsne:    
                Z_embedded = TSNE(n_components=2, verbose=1).fit_transform(zTensor.cpu())        
                z1 = Z_embedded[:, 0]
                z2 = Z_embedded[:, 1]
                scatterPlot = plt.scatter(z1, z2, s = 4) #TSNE projection for >3dim 
            else:
                reducer = umap.UMAP()
                Z_embedded = reducer.fit_transform(zTensor.cpu())
                scatterPlot = plt.scatter(Z_embedded[:, 0], Z_embedded[:, 1], s = 4)

            plt.show()
        if(args.print_pseudos>0):
            t=min(args.print_pseudos,args.pseudos)
            temp = model.means(model.idle_input).view(-1,args.input_length,args.input_length).detach().cpu()
            for x in range(t):
                plt.matshow(temp[x].numpy())
                plt.show()
                
if not args.distributed:
    summary(model,(1,args.input_length,args.input_length))
if(args.load == ''):
    for epoch in range(1, args.epochs + 1):
        if(not stopEarly):
            train(epoch)
            test(epoch, args.epochs, startTime)
else:
    checkpoint=torch.load(args.load)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    test(args.epochs, args.epochs, startTime)
    if(args.repeat==True):
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch, args.epochs, startTime)
            
if(args.log!='!'):
    #res = torch.autograd.Variable(torch.Tensor(1,1,128,128), requires_grad=True).to(device)
    #writer.add_graph(model,res,verbose=True)
    writer.close()