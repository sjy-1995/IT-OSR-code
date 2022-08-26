# coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
# import datasets.utils_version3 as dataHelper
import datasets_unified.utils_version3 as dataHelper
from utils_try3 import progress_bar
import os
import torchvision.transforms as tf
import torchvision
from torchvision import models
import numpy as np

# swin transformer as the backbone
from swin_transformer import SwinTransformer   # the more complex file


parser = argparse.ArgumentParser(description='Open Set Classifier Training')
# parser.add_argument('--dataset', required=True, type=str, help='Dataset for training',
#                     choices=['MNIST', 'SVHN', 'CIFAR10', 'CIFAR+10', 'CIFAR+50', 'TinyImageNet'])
# parser.add_argument('--trial', required=True, type=int, help='Trial number, 0-4 provided')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
parser.add_argument('--alpha', default=10, type=int, help='Magnitude of the anchor point')
parser.add_argument('--lbda', default=0.1, type=float, help='Weighting of Anchor loss component')
parser.add_argument('--tensorboard', '-t', action='store_true', help='Plot on tensorboardX')
parser.add_argument('--name', default="backbone20210816_", type=str, help='Optional name for saving and tensorboard')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters useful when resuming and finetuning
best_acc = 0
best_loss = 10000
start_epoch = 0

# Create dataloaders for training
print('==> Preparing data..')
# with open('datasets/config_F.json') as config_file:   # 224, 224
with open('datasets/config_swin_vit.json') as config_file:   # 224, 224
    cfg = json.load(config_file)['CIFAR10']


# class Net(nn.Module):
#     def __init__(self, model, num_class):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
#         super(Net, self).__init__()
#         # # 取掉model的后两层
#         self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
#         # self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
#         # self.pool_layer = nn.MaxPool2d(32)
#         # self.Linear_layer = nn.Linear(2048, 8)
#         self.ln1 = nn.Linear(512, 4096)
#         self.ln2 = nn.Linear(4096, 4096)
#         self.ln3 = nn.Linear(4096, num_class)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # print(x.shape)
#         x = self.resnet_layer(x)
#         # print(x.shape)
#         x1 = x.view(x.shape[0], -1)
#         x2 = self.relu(self.ln1(x1))
#         x2 = self.relu(self.ln2(x2))
#         x2 = self.ln3(x2)
#         return x1, x2   # feature and output


class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(1024, 4096)
        self.ln2 = nn.Linear(4096, 4096)
        self.ln3 = nn.Linear(4096, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.ln1(x))
        x1 = self.relu(self.ln2(x1))
        x1 = self.ln3(x1)
        return x1


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def load_datasets(cfg):

	# controls data transforms
	flip = cfg['data_transforms']['flip']
	rotate = cfg['data_transforms']['rotate']
	scale_min = cfg['data_transforms']['scale_min']
	means_imagenet = [0.5, 0.5, 0.5]
	stds_imagenet = [0.5, 0.5, 0.5]

	transforms = {
		'train': tf.Compose([
			tf.Resize(cfg['im_size']),
			tf.RandomResizedCrop(cfg['im_size'], scale = (scale_min, 1.0)),
			tf.RandomHorizontalFlip(flip),
			tf.RandomRotation(rotate),
			tf.ToTensor(),
			# tf.Normalize(means, stds)
			# tf.Normalize(means_imagenet, stds_imagenet)
		]),
		'val': tf.Compose([
			tf.Resize(cfg['im_size']),
			tf.ToTensor(),
			# tf.Normalize(means, stds)
			# tf.Normalize(means_imagenet, stds_imagenet)
		]),
		'test': tf.Compose([
			tf.Resize(cfg['im_size']),
			tf.ToTensor(),
			# tf.Normalize(means, stds)
			# tf.Normalize(means_imagenet, stds_imagenet)
		])
	}

	trainSet = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['train'], download = True)
	valSet = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['val'])
	testSet = torchvision.datasets.CIFAR10('datasets/data', train = False, transform = transforms['test'])
	trainSet_noaug = torchvision.datasets.CIFAR10('datasets/data', transform=transforms['test'], download=True)

	return trainSet, valSet, testSet, trainSet_noaug


def create_dataSubsets(dataset, idxs_to_use=None):

	# ##################################### for pytorch 1.X ##############################################
	# get class label for dataset images. svhn has different syntax as .labels
	targets = dataset.targets
	# ####################################################################################################
	subset_idxs = []
	if idxs_to_use == None:
		for i, lbl in enumerate(targets):
			subset_idxs += [i]
	else:
		for class_num in idxs_to_use.keys():
			subset_idxs += idxs_to_use[class_num]

	dataSubset = torch.utils.data.Subset(dataset, subset_idxs)
	return dataSubset


def get_train_loaders(cfg):

	trainSet, valSet, testSet, trainSet_noaug = load_datasets(cfg)

	with open("datasets_unified/{}/trainval_idxs.json".format('CIFAR10')) as f:
		trainValIdxs = json.load(f)
		train_idxs = trainValIdxs['Train']
		val_idxs = trainValIdxs['Val']

	trainSubset = create_dataSubsets(trainSet, train_idxs)
	valSubset = create_dataSubsets(valSet, val_idxs)
	testSubset = create_dataSubsets(testSet)
	trainSubset_noaug = create_dataSubsets(trainSet_noaug, train_idxs)

	batch_size = cfg['batch_size']

	trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=True, num_workers=cfg['dataloader_workers'])
	valloader = torch.utils.data.DataLoader(valSubset, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=True)

	return trainloader, valloader, testloader, trainSubset, testSet, trainSubset_noaug


def get_mean_std(dataset, ratio=0.01):
	"""
	Get mean and std by sample ratio
	"""
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), shuffle=True, num_workers=2)
	train = iter(dataloader).next()[0]   # the data on one batch
	mean = np.mean(train.numpy(), axis=(0, 2, 3))
	std = np.std(train.numpy(), axis=(0, 2, 3))
	return mean, std


trainloader, valloader, _, _, _, _ = get_train_loaders(cfg)
# trainloader, valloader, _, mapping = dataHelper.get_train_loaders(args.dataset, args.trial, cfg)
print('==> Building network..')

F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False
                           )   # the feature dim is 1024

F.apply(inplace_relu)
F.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)   # use the pretrained model trained on ImageNet
F = F.to(device)
C = Classifier(10)
C.apply(inplace_relu)
C = C.to(device)

training_iter = int(args.resume)

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('networks/weights'), 'Error: no checkpoint directory found!'
#     # checkpoint = torch.load('networks/weights/{}/{}_{}_try_version_variousK_5CACclassifierAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
#     # checkpoint = torch.load('networks/weights/{}/{}_{}_try_version5_2CACclassifierAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
#     checkpoint = torch.load('networks/weights/{}/{}_{}_try_version5_2_finetuning1CACclassifierAccuracy.pth'.format(args.dataset, args.dataset, args.trial))
#     start_epoch = checkpoint['epoch']
#     net_dict = net.state_dict()
#     pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
#     # net.load_state_dict(pretrained_dict)
#     net_dict.update(pretrained_dict)
#     net.load_state_dict(net_dict)

F.train()
C.train()
#
# optimizer_F = optim.AdamW(F.parameters(), lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer_C = optim.AdamW(C.parameters(), lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
optimizer_F = optim.SGD(F.parameters(), lr=0.1)
optimizer_C = optim.SGD(C.parameters(), lr=0.1)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # if epoch < 200:
    # # if epoch < 50:
    #     optimizer = optim.SGD(F.parameters(), lr=0.01, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
    # elif epoch < 300:
    # # elif epoch < 100:
    #     optimizer = optim.SGD(F.parameters(), lr=0.001, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])
    # else:
    #     optimizer = optim.SGD(F.parameters(), lr=0.0001, momentum=0.9, weight_decay=cfg['openset_training']['weight_decay'])

    F.train()
    C.train()

    train_loss = 0
    correctDist = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print(batch_idx)
        if inputs.shape[1] == 3:
            pass
        else:
            inputs = inputs.repeat(1, 3, 1, 1)
        inputs, targets = inputs.to(device), targets.to(device)
        # convert from original dataset label to known class label
        # targets = torch.Tensor([mapping[x] for x in targets]).long()
        targets = targets.long()
        targets = targets.to(device)
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        outputs = F(inputs)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = C(outputs)
        CEloss = nn.CrossEntropyLoss()
        celoss = CEloss(outputs, targets)
        total_loss = celoss
        total_loss.backward()
        optimizer_F.step()
        optimizer_C.step()
        train_loss = train_loss + total_loss.detach().cpu().numpy()
        _, predicted = outputs.max(1)
        total = total + targets.size(0)
        correctDist = correctDist + predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Total_Loss: %.3f|CE_Loss: %.3f|Acc: %.3f%% (%d/%d)'
                     % (total_loss, celoss, 100. * correctDist / total, correctDist, total))


def val(epoch):
    global best_loss
    global best_acc
    F.eval()
    C.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if inputs.shape[1] == 3:
                pass
            else:
                inputs = inputs.repeat(1, 3, 1, 1)
            inputs = inputs.to(device)
            # targets = torch.Tensor([mapping[x] for x in targets]).long()
            targets = targets.long()
            targets = targets.to(device)
            outputs = F(inputs)
            outputs = outputs.view(outputs.shape[0], -1)
            outputs = C(outputs)
            CEloss = nn.CrossEntropyLoss()
            celoss = CEloss(outputs, targets)
            total_loss = celoss
            total_loss = total_loss.detach().cpu().numpy()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(valloader), 'Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

    total_loss /= len(valloader)
    print(total_loss)
    acc = 100. * correct / total
    print(acc)

    print('Saving..')
    torch.save(F, 'networks/weights/{}/{}_{}F_'.format('CIFAR10', 'CIFAR10', args.name) + 'Newest.pth')
    torch.save(C, 'networks/weights/{}/{}_{}C_'.format('CIFAR10', 'CIFAR10', args.name) + 'Newest.pth')

    if acc >= best_acc:
        print('Saving..')
        torch.save(F, 'networks/weights/{}/{}_{}F_'.format('CIFAR10', 'CIFAR10', args.name) + 'Accuracy.pth')
        torch.save(C, 'networks/weights/{}/{}_{}C_'.format('CIFAR10', 'CIFAR10', args.name) + 'Accuracy.pth')
        best_acc = acc

max_epoch = 400 + start_epoch


for epoch in range(start_epoch, max_epoch):
    train(epoch)
    val(epoch)

