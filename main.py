from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./test', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

# -----------------------------------------------------------------------
# Integrate the idea of IncReg to network slimming
TARGET_REG = 1.0
AA = 0.5e-4
PR = 0.7
Reg = {}
IF_chl_alive = {}
num_pruned_chl = {}
NUM_SHOW = 20
IF_layer_finished = {}
pruned_ratio = {}

def punish_func(r, num_g, thre_rank):
  if r <= thre_rank:
    return AA - AA/thre_rank * r
  else:
    return -AA / (num_g - 1 - thre_rank) * (r - thre_rank)

def updateBN_IncReg():
  bn_cnt = -1
  for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
      bn_cnt += 1; layer_ix = str(bn_cnt)
      num_chl = m.weight.data.shape[0]
      
      # initialization/registration
      if layer_ix not in Reg:
        Reg[layer_ix] = [0] * num_chl
        IF_chl_alive[layer_ix] = [1] * num_chl
        num_pruned_chl[layer_ix] = 0
        pruned_ratio[layer_ix] = 0
        IF_layer_finished[layer_ix] = 0
        
      num_pruned_chl_ = num_pruned_chl[layer_ix]
      num_chl_ = num_chl - num_pruned_chl_
      num_chl_to_prune_ = int(num_chl * PR) - num_pruned_chl_
      
      if num_chl_to_prune_ > 0:
        # sort
        abs_scale = m.weight.data.abs()
        sort_index = abs_scale.sort()[1]

        for i in range(num_chl_):
          chl_of_rank_i = sort_index[i + num_pruned_chl_]
          Delta = punish_func(i, num_chl_, num_chl_to_prune_)
          Reg[layer_ix][chl_of_rank_i] = max(Reg[layer_ix][chl_of_rank_i] + Delta, 0)
          if Reg[layer_ix][chl_of_rank_i] >= TARGET_REG:
            IF_chl_alive[layer_ix][chl_of_rank_i] = 0
            num_pruned_chl[layer_ix] += 1
            pruned_ratio[layer_ix] = num_pruned_chl[layer_ix] / num_chl
            IF_layer_finished[layer_ix] = 1
          
        # apply reg to bn weights
        reg_new = torch.from_numpy(np.array(Reg[layer_ix])).float().cuda()
        m.weight.grad.data.add_(reg_new * m.weight.data) # use L2
      
      # mask out the gradient
      tmp = torch.from_numpy(np.array(IF_chl_alive[layer_ix])).float().cuda()
      m.weight.grad.data.mul_(tmp)
      m.weight.data.mul_(tmp)
# -----------------------------------------------------------------------      
  

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        if args.sr:
            updateBN_IncReg()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # print bn reg
            for i in range(len(Reg.keys())):
              print("the bn reg of layer %2d:" % i, end=" ")
              for k in range(NUM_SHOW):
                if k >= len(Reg[str(i)]): break
                v = Reg[str(i)][k]
                print("%.4f" % v, end=" ")
              print("pruned_ratio: %.4f" % pruned_ratio[str(i)])
            
            # print bn weight
            cnt = -1
            for m in model.modules():
              if isinstance(m, nn.BatchNorm2d):
                cnt += 1; layer_ix = str(cnt)
                print("the bn weight of layer %2d:" % cnt, end=" ")
                for i in range(NUM_SHOW):
                  print("%.4f (%s)" % (abs(m.weight.data[i].item()), IF_chl_alive[layer_ix][i]), end=" ")
                print("")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)
print("Best accuracy: "+str(best_prec1))