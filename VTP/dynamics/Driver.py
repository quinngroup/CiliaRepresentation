from argparse import ArgumentParser
from VTP.dynamics.module import VTP_D

#Parses arguments passed from command line
parser = ArgumentParser(description='DynamicsDriver')
parser.add_argument('--encoder', type=str, default=None, 
    help='choose the desired encoder architecture')
parser.add_argument('--decoder', type=str, default=None, 
    help='choose the desired decoder architecture')
parser.add_argument('--trainsource', type=str, default=None,
    help='directory containing source training files')
parser.add_argument('--testsource', type=str, default=None,
    help='directory containing source testing files')
parser.add_argument('--optimizer', type=str, default=None, #ADD DEFAULT OPTIMIZER
    help='directory containing source testing files')
parser.add_argument('--no_cuda', action='store_true', default=False,
    help='enables CUDA training')
args = parser.parse_args()

model = VTP_D(encoder=args.encoder, decoder=args.decoder)

# DISTRIBUTED PROGRAMMING STUFF BELOW

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
    worldCount=int(os.environ['WORLD_SIZE'])
    args.distributed = worldCount > 1

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
if(args.log!='!' and args.local_rank==0):
    if(args.log=='$'):
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(log_dir=args.log)

arguments=[args.input_length, args.lsdim, args.pseudos, args.beta, args.gamma, device, args.logvar_bound]
if args.model=='nvp':
    model = NVP(*arguments)
elif args.model=='nvp1':
    model = NVP_1.NVP_1(*arguments)
elif args.model=='nvp4':
    model = NVP_4.NVP_4(*arguments)

model.cuda()
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


imagePace=None
if args.log_image>0 and args.log!='!':
    imagePace=len(train_loader.dataset)//(args.batch_size*args.log_image)


stopEarly = False
failedEpochs=0
lastLoss = 0

scheduler=None
if(args.schedule>0):
            scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, min_lr = args.min_lr, verbose=True, patience=args.schedule)


scale=1
if args.distributed:
    scale=2

def printLoss(phase, loss, epoch=None, batch_idx=None, data_length=None, genLoss=None):
    if phase == 'train':
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tGenLoss: {:.6f}'.format(
            epoch, scale*batch_idx * data_length, len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item() / data_length,
            genLoss))
    elif phase == 'average':
        print('====> Epoch: {} Average loss (main gpu): {:.4f}'.format(
            epoch, scale*loss / len(train_loader.dataset)))
    elif phase=='test':
        print('====> Test set loss: {:.4f}'.format(loss))
        print('====> Generation loss: {:.4f}'.format(genLoss))
    else:
        print('Loss printing error')
    
    
    

# NEED TO FIGURE OUT HOW TO GO FROM INPUT STRING TO FILE
train_loader = args.trainsource
test_loader = args.testsource

# NEED TO FIGURE OUT HOW TO LOAD IN OPTIMIZER
optimizer = args.optimizer

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
