# coding: utf-8
import os
import sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
# import sklearn.datasets
from models.conwgan_try3_2 import *
# from models.conwgan_try3_2_exp6_20210906_wgan import *
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets, models
import torch.nn.init as init
# import datasets.utils_version3_trytransductive_3 as dataHelper1
# import datasets.utils_version3 as dataHelper2
# import datasets_unified.utils_version3_trytransductive_3 as dataHelper1
import datasets_unified.utils_version3 as dataHelper2
import json
from itertools import cycle
import torchvision.transforms as tf
from utils_try3 import progress_bar
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from math import log
import scipy.io as sio
import copy
import skimage
import skimage.io
import skimage.transform

# swin transformer as the backbone
from swin_transformer import SwinTransformer   # the more complex file

# 载入sklearn中的k近邻分类器方法
from sklearn.neighbors import KNeighborsClassifier


parser = argparse.ArgumentParser(description='Open Set Classifier Training')
# parser.add_argument('--dataset', required=True, type=str, help='Dataset for training', choices=['CIFAR10'])
# parser.add_argument('--trial', required=True, type=int, help='Trial number, 0-4 provided')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
parser.add_argument('--tensorboard', '-t', action='store_true', help='Plot on tensorboardX')
args = parser.parse_args()


class Classifier(nn.Module):
	def __init__(self, num_class):   # 从头开始训练，不加载预训练模型的模型
		super(Classifier, self).__init__()
		# self.ln1 = nn.Linear(512, 4096)
		self.ln1 = nn.Linear(1024, 4096)
		self.ln2 = nn.Linear(4096, 4096)
		self.ln3 = nn.Linear(4096, num_class)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.ln1(x))
		x = self.relu(self.ln2(x))
		x = self.ln3(x)
		return x


class Bi(nn.Module):
	def __init__(self):   # 从头开始训练，不加载预训练模型的模型
		super(Bi, self).__init__()
		# self.ln1 = nn.Linear(512, 4096)
		self.ln1 = nn.Linear(1024, 4096)
		self.ln2 = nn.Linear(4096, 4096)
		self.ln3 = nn.Linear(4096, 2)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.ln1(x))
		x = self.relu(self.ln2(x))
		x = self.ln3(x)
		return x


def my_gather_outputs(netF, netC, dataloader):
	X = []
	for i, data in enumerate(dataloader):
		images, labels = data
		if images.shape[1] == 3:
			pass
		else:
			images = images.repeat(1, 3, 1, 1)
		images = images.cuda()
		# features = netF(images).squeeze(-1).squeeze(-1)
		features = netF(images)
		features = features.view(features.shape[0], -1)
		outputs = netC(features)
		X += outputs.cpu().detach().tolist()
	X = np.asarray(X)
	return X


def my_gather_outputs_with_label(netF, netC, dataloader):
	X = []
	Y = []
	for i, data in enumerate(dataloader):
		# if i > 1000:
		#     break
		images, labels = data
		if images.shape[1] == 3:
			pass
		else:
			images = images.repeat(1, 3, 1, 1)
		# labels = torch.Tensor([mapping[x] for x in labels])
		images = images.cuda()
		# features = netF(images).squeeze(-1).squeeze(-1)
		features = netF(images)
		features = features.view(features.shape[0], -1)
		outputs = netC(features)
		X += outputs.cpu().detach().tolist()
		Y += labels.tolist()
	X = np.asarray(X)
	Y = np.asarray(Y)
	return X, Y


def my_gather_features_with_label(netF, dataloader):
	F = []
	Y = []
	for i, data in enumerate(dataloader):
		# if i > 1000:
		#     break
		images, labels = data
		if images.shape[1] == 3:
			pass
		else:
			images = images.repeat(1, 3, 1, 1)
		# labels = torch.Tensor([mapping[x] for x in labels])
		images = images.cuda()
		# features = netF(images).squeeze(-1).squeeze(-1)
		features = netF(images)
		features = features.view(features.shape[0], -1)
		F += features.cpu().detach().tolist()
		Y += labels.tolist()
	F = np.asarray(F)
	Y = np.asarray(Y)
	return F, Y


with open('datasets/config_swin_vit_GAN20210820_smaller3.json') as config_file:   # batch size = 16
	cfg = json.load(config_file)['CIFAR10']


def my_gather_outputs_with_unknown_label(netF, netC, dataloader):
	X = []
	Y = []
	for i, data in enumerate(dataloader):
		# if i > 1000:
		#     break
		images, labels = data

		if images.shape[1] == 3:
			pass
		else:
			images = images.repeat(1, 3, 1, 1)

		labels = 10 * torch.ones_like(labels)
		images = images.cuda()
		# features = netF(images).squeeze(-1).squeeze(-1)
		features = netF(images)
		features = features.view(features.shape[0], -1)
		outputs = netC(features)
		X += outputs.cpu().detach().tolist()
		Y += labels.tolist()
	X = np.asarray(X)
	Y = np.asarray(Y)
	return X, Y


def my_gather_features_with_unknown_label(netF, dataloader):
	F = []
	Y = []
	for i, data in enumerate(dataloader):
		# if i > 1000:
		#     break
		images, labels = data
		if images.shape[1] == 3:
			pass
		else:
			images = images.repeat(1, 3, 1, 1)

		labels = 10 * torch.ones_like(labels)
		images = images.cuda()
		# features = netF(images).squeeze(-1).squeeze(-1)
		features = netF(images)
		features = features.view(features.shape[0], -1)
		F += features.cpu().detach().tolist()
		Y += labels.tolist()
	F = np.asarray(F)
	Y = np.asarray(Y)
	return F, Y


def my_gather_labels(netF, netC, data):
	# features = netF(data).squeeze(-1).squeeze(-1)
	features = netF(data)
	features = features.view(features.shape[0], -1)
	outputs = netC(features)
	Y = torch.argmax(outputs, 1)
	return Y


def my_gather_labels_mix(netF, netC1, netC2, data):
	# features = netF(data).squeeze(-1).squeeze(-1)
	features = netF(data)
	features = features.view(features.shape[0], -1)
	outputs1 = netC1(features)
	outputs2 = netC2(features)
	outputs = outputs1 + outputs2[:, :-1]
	Y = torch.argmax(outputs, 1)
	return Y


def inplace_relu(m):
	classname = m.__class__.__name__
	if classname.find('ReLU') != -1:
		m.inplace = True


def make_one_hot(input, num_classes):
	"""Convert class index tensor to one hot encoding tensor.
	Args:
		 input: A tensor of shape [bs, 1, *]
		 num_classes: An int of number of class
	Returns:
		A tensor of shape [bs, num_classes, *]
	"""
	input = input.unsqueeze(1)
	input = input.long()
	shape = np.array(input.shape)
	shape[1] = num_classes
	shape = tuple(shape)
	result = torch.zeros(shape).cuda()
	result = result.scatter_(1, input, 1)
	result = result.half()

	return result


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
	testtransSet = torchvision.datasets.CIFAR10('datasets/data', train = False, transform = transforms['train'])
	trainSet_noaug = torchvision.datasets.CIFAR10('datasets/data', transform=transforms['test'], download=True)

	return trainSet, valSet, testSet, testtransSet, trainSet_noaug


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

	trainSet, valSet, testSet, testtransSet, trainSet_noaug = load_datasets(cfg)

	with open("datasets_unified/{}/trainval_idxs.json".format('CIFAR10')) as f:
		trainValIdxs = json.load(f)
		train_idxs = trainValIdxs['Train']
		val_idxs = trainValIdxs['Val']

	trainSubset = create_dataSubsets(trainSet, train_idxs)
	valSubset = create_dataSubsets(valSet, val_idxs)
	testSubset = create_dataSubsets(testSet)
	testtransSubset = create_dataSubsets(testtransSet)
	trainSubset_noaug = create_dataSubsets(trainSet_noaug, train_idxs)

	batch_size = cfg['batch_size']

	trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=True, num_workers=cfg['dataloader_workers'])
	valloader = torch.utils.data.DataLoader(valSubset, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=True)

	return trainloader, valloader, testloader, trainSubset, testSubset, testtransSubset, trainSubset_noaug


def get_mean_std(dataset, ratio=0.01):
	"""
	Get mean and std by sample ratio
	"""
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), shuffle=True, num_workers=2)
	train = iter(dataloader).next()[0]   # the data on one batch
	mean = np.mean(train.numpy(), axis=(0, 2, 3))
	std = np.std(train.numpy(), axis=(0, 2, 3))
	return mean, std


class CreateData(Dataset):  # 继承Dataset
	def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
		self.root_dir = root_dir  # 文件目录
		self.transform = transform  # 变换
		self.images = os.listdir(self.root_dir)  # 目录里的所有文件

	def __len__(self):  # 返回整个数据集的大小
		return len(self.images)

	def __getitem__(self, index):  # 根据索引index返回dataset[index]
		image_index = self.images[index]  # 根据索引index获取该图片
		img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
		img = skimage.io.imread(img_path)  # 读取该图片
		label = 10   # OOD datasets
		img = self.transform(img)
		# label = self.transform(label)
		# sample = {'image': img, 'label': label}  # 根据图片和标签创建字典
		# sample = {'image': img, 'label': label}  # 根据图片和标签创建字典
		# if self.transform:
		# 	sample = self.transform(sample)  # 对样本进行变换
		# return sample  # 返回该样本
		return img, label  # 返回该样本


# trainloader, valloader, testloader, trainSet, trainSet_noaug = get_train_loaders(cfg)
# trainloader, _, _, trainSet, testSet, trainSet_noaug = get_train_loaders(cfg)
dataloader, _, _, trainSet, testSet, testtransSet, trainSet_noaug = get_train_loaders(cfg)
dataloader_eval = torch.utils.data.DataLoader(trainSet_noaug, batch_size=16, shuffle=False, num_workers=cfg['dataloader_workers'])
# CIFAR10_mean, CIFAR10_std = get_mean_std(trainSet)
testout_transform = tf.Compose([
		tf.ToPILImage(),
		tf.Resize(cfg['im_size']),
		tf.ToTensor(),
		# tf.Normalize((CIFAR10_mean, CIFAR10_std)),
	])

flip = cfg['data_transforms']['flip']
rotate = cfg['data_transforms']['rotate']
scale_min = cfg['data_transforms']['scale_min']
testout_transform2 = tf.Compose([
		tf.ToPILImage(),
		tf.Resize(cfg['im_size']),
		tf.RandomResizedCrop(cfg['im_size'], scale=(scale_min, 1.0)),
		tf.RandomHorizontalFlip(flip),
		tf.RandomRotation(rotate),
		tf.ToTensor(),
		# tf.Normalize((CIFAR10_mean, CIFAR10_std)),
	])

OODdataset = CreateData('../cac_openset/cac-openset-master/OOD_datasets/Imagenet_crop/test', transform=testout_transform)
OODtransdataset = CreateData('../cac_openset/cac-openset-master/OOD_datasets/Imagenet_crop/test', transform=testout_transform2)
# unknownloader = torchvision.datasets.ImageFolder("../cac_openset/cac-openset-master/OOD_datasets/Imagenet_crop", transform=testout_transform)
# testout_dataloader = torch.utils.data.DataLoader(testout_dataset, batch_size=50, shuffle=False, num_workers=2)

F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
					embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
					window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
					drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
					norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
					# use_checkpoint=False
					use_checkpoint=True
					)  # the feature dim is 1024
# F = torch.load('networks/weights/{}/{}_{}_backbone20210816_F_Accuracy.pth'.format(args.dataset, args.dataset, args.trial))
F = torch.load('networks/weights/{}/{}_backbone20210816_F_Newest.pth'.format('CIFAR10', 'CIFAR10'))
F.apply(inplace_relu)
F = F.cuda()
F.eval()
for p in F.parameters():
	p.requires_grad_(False)  # freeze F

C = Classifier(cfg['num_known_classes'])
# C = torch.load('networks/weights/{}/{}_{}_backbone20210816_C_Accuracy.pth'.format(args.dataset, args.dataset, args.trial))
C = torch.load('networks/weights/{}/{}_backbone20210816_C_Newest.pth'.format('CIFAR10', 'CIFAR10'))
C.apply(inplace_relu)
C = C.cuda()
C.eval()
for p in C.parameters():
	p.requires_grad_(False)  # freeze C

# ############################################################################################################################
# 根据已训练好的k类分类器（SoftMax输出的负熵作为分数，训练集的平均分数作为阈值）、训练初始GAN得到的k+1类分类器对测试集进行初步筛选
# 用k+1类分类器筛选：
# dataloader, dataloader_eval, mapping = dataHelper1.get_train_loaders(args.dataset, args.trial, cfg)
# knownloader, unknownloader, alltestset, alltesttransset, mapping = dataHelper1.get_eval_loaders(args.dataset, args.trial, cfg)   # 未打乱数据集顺序

knownloader = torch.utils.data.DataLoader(testSet, batch_size=16, shuffle=False, num_workers=2)
unknownloader = torch.utils.data.DataLoader(OODdataset, batch_size=16, shuffle=False, num_workers=2)
alltestset = ConcatDataset([testSet, OODdataset])
alltesttransset = ConcatDataset([testtransSet, OODtransdataset])

# ################################################################
# 用k类分类器筛选：
with torch.no_grad():
	# xval = my_gather_outputs(F, C, dataloader_eval)
	xval, yval = my_gather_outputs_with_label(F, C, dataloader_eval)
	xK, yK = my_gather_outputs_with_label(F, C, knownloader)
	FK, _ = my_gather_features_with_label(F, knownloader)
	# xU = my_gather_outputs(F, C, unknownloader)
	xU, yU = my_gather_outputs_with_unknown_label(F, C, unknownloader)
	FU, _ = my_gather_features_with_unknown_label(F, unknownloader)
xtest = np.concatenate((xK, xU), 0)
Ftest = np.concatenate((FK, FU), 0)
ypred_rough = np.argmax(xtest, 1)
ytest = np.concatenate((yK, yU))

xtest_softmax = np.exp(xtest - np.max(xtest, 1)[:, np.newaxis].repeat(xtest.shape[1], 1)) / np.sum(np.exp(xtest - np.max(xtest, 1)[:, np.newaxis].repeat(xtest.shape[1], 1)), 1)[:, np.newaxis].repeat(xtest.shape[1], 1)
xval_softmax = np.exp(xval - np.max(xval, 1)[:, np.newaxis].repeat(xval.shape[1], 1)) / np.sum(np.exp(xval - np.max(xval, 1)[:, np.newaxis].repeat(xval.shape[1], 1)), 1)[:, np.newaxis].repeat(xval.shape[1], 1)

score1 = np.sum(xtest_softmax * np.log(xtest_softmax + 1e-6), 1)  # the minus of the entropy of the outputs
score1_val = np.sum(xval_softmax * np.log(xval_softmax + 1e-6), 1)  # the minus of the entropy of the outputs
threshold1_known_classes = []
threshold1_unknown_classes = []
known_score1_list = []
unknown_score1_list = []
for i in range(cfg['num_known_classes']):
	this_class_score1_val = score1_val[np.where(yval == i)]
	this_class_thresh_known = np.mean(this_class_score1_val).tolist()
	threshold1_known_classes += [this_class_thresh_known]
	this_class_thresh_unknown = np.mean(this_class_score1_val).tolist() - 1.0 * np.std(this_class_score1_val).tolist()
	threshold1_unknown_classes += [this_class_thresh_unknown]
	index_this_class = np.where(ypred_rough == i)[0].tolist()
	score1_classes = score1[index_this_class]
	known_score1_list += np.asarray(index_this_class)[np.where(score1_classes > this_class_thresh_known)].tolist()
	unknown_score1_list += np.asarray(index_this_class)[np.where(score1_classes < this_class_thresh_unknown)].tolist()

score2 = np.sort(xtest, axis=1)[:, -1] - np.sort(xtest, axis=1)[:, -2]  # a-b, where a is the max value, and b is the second max value
score2_val = np.sort(xval, axis=1)[:, -1] - np.sort(xval, axis=1)[:, -2]  # a-b, where a is the max value, and b is the second max value
threshold2_known_classes = []
threshold2_unknown_classes = []
known_score2_list = []
unknown_score2_list = []
for i in range(cfg['num_known_classes']):
	this_class_score2_val = score2_val[np.where(yval == i)]
	this_class_thresh_known = np.mean(this_class_score2_val).tolist()
	threshold2_known_classes += [this_class_thresh_known]
	this_class_thresh_unknown = np.mean(this_class_score2_val).tolist() - 1.0 * np.std(this_class_score2_val).tolist()
	threshold2_unknown_classes += [this_class_thresh_unknown]
	index_this_class = np.where(ypred_rough == i)[0].tolist()
	score2_classes = score2[index_this_class]
	known_score2_list += np.asarray(index_this_class)[np.where(score2_classes > this_class_thresh_known)].tolist()
	unknown_score2_list += np.asarray(index_this_class)[np.where(score2_classes < this_class_thresh_unknown)].tolist()

score3 = np.sort(xtest, axis=1)[:, -1]  # the max value of the output
score3_val = np.sort(xval, axis=1)[:, -1]  # the max value of the output
threshold3_known_classes = []
threshold3_unknown_classes = []
known_score3_list = []
unknown_score3_list = []
for i in range(cfg['num_known_classes']):
	this_class_score3_val = score3_val[np.where(yval == i)]
	this_class_thresh_known = np.mean(this_class_score3_val).tolist()
	threshold3_known_classes += [this_class_thresh_known]
	this_class_thresh_unknown = np.mean(this_class_score3_val).tolist() - 1.0 * np.std(this_class_score3_val).tolist()
	threshold3_unknown_classes += [this_class_thresh_unknown]
	index_this_class = np.where(ypred_rough == i)[0].tolist()
	score3_classes = score3[index_this_class]
	known_score3_list += np.asarray(index_this_class)[np.where(score3_classes > this_class_thresh_known)].tolist()
	unknown_score3_list += np.asarray(index_this_class)[np.where(score3_classes < this_class_thresh_unknown)].tolist()

known_test_list = list(set(known_score1_list).intersection(set(known_score2_list)).intersection(set(known_score3_list)))
unknown_test_list = list(set(unknown_score1_list).intersection(set(unknown_score2_list)).intersection(set(unknown_score3_list)))

# ########################################### 若按照之前的方法，整体计算分数和阈值进行初始筛选 ###########################################
# score1 = np.sum(xtest_softmax * np.log(xtest_softmax + 1e-6), 1)  # the minus of the entropy of the outputs
# threshold_1_known = np.mean(np.sum(xval_softmax * np.log(xval_softmax + 1e-6), 1))
# threshold_1_unknown = np.mean(np.sum(xval_softmax * np.log(xval_softmax + 1e-6), 1)) - 1 * np.std(np.sum(xval_softmax * np.log(xval_softmax + 1e-6), 1))
#
# score2 = np.sort(xtest, axis=1)[:, -1] - np.sort(xtest, axis=1)[:, -2]  # a-b, where a is the max value, and b is the second max value
# threshold_2_known = np.mean(np.sort(xval, axis=1)[:, -1] - np.sort(xval, axis=1)[:, -2]) + 0 * np.std(np.sort(xval, axis=1)[:, -1] - np.sort(xval, axis=1)[:, -2])
# threshold_2_unknown = np.mean(np.sort(xval, axis=1)[:, -1] - np.sort(xval, axis=1)[:, -2]) - 1 * np.std(np.sort(xval, axis=1)[:, -1] - np.sort(xval, axis=1)[:, -2])
#
# score3 = np.sort(xtest, axis=1)[:, -1]  # the max value of the output
# threshold_3_known = np.mean(np.sort(xval, axis=1)[:, -1]) + 0 * np.std(np.sort(xval, axis=1)[:, -1])
# threshold_3_unknown = np.mean(np.sort(xval, axis=1)[:, -1]) - 1 * np.std(np.sort(xval, axis=1)[:, -1])
# known_test_list = np.where((score1 > threshold_1_known) * (score2 > threshold_2_known) * (score3 > threshold_3_known))[0].tolist()
# unknown_test_list = np.where((score1 < threshold_1_unknown) * (score2 < threshold_2_unknown) * (score3 < threshold_3_unknown))[0].tolist()

# ##############################################################################################################################

# known_list_save = np.asarray(known_test_list)
# unknown_list_save = np.asarray(unknown_test_list)
# np.save('known_list_try2.npy', known_list_save)
# np.save('unknown_list_try2.npy', unknown_list_save)

print('{} samples have been selected for known classes / the whole {} known-class samples'.format(len(known_test_list), xK.shape[0]))
print('{} samples have been selected for unknown classes / the whole {} unknown-class samples'.format(len(unknown_test_list), xU.shape[0]))

num_true_known = sum(i < xK.shape[0] for i in known_test_list)
num_true_unknown = sum(i >= xK.shape[0] for i in unknown_test_list)
# 已知类别的伪标签是两个分类器对应位置的加和的最大的位置
# pred_known = np.argmax(xK + xK_aD3_init[:, :-1], 1)
pred_known = np.argmax(xK, 1)
list_correct_known = np.where(pred_known == yK)[0].tolist()
num_true_classified_known = len(list(set(list_correct_known).intersection(set(known_test_list))))
print('the ratio of the known-class samples correctly selected is : ', num_true_known / len(known_test_list))   # 筛选出的已知类别确实是已知类别的概率
print('the ratio of the unknown-class samples correctly selected is : ', num_true_unknown / len(unknown_test_list))   # 筛选出的未知类别确实是未知类别的概率
print('the ratio of the correctly classified samples is : ', num_true_classified_known / len(known_test_list))   # 筛选出的已知类别样本中分类用的伪标签正确的概率

# 保存logit向量和真实标签，用于之后的查看与分析
# selected_known = xtest[known_test_list]
# selected_unknown = xtest[unknown_test_list]
# selected_known_labels = ytest[known_test_list]
# selected_unknown_labels = ytest[unknown_test_list]

# sio.savemat('selected_known_try2.mat', {'s_k': selected_known})
# sio.savemat('selected_unknown_try2.mat', {'s_unk': selected_unknown})
# sio.savemat('selected_known_labels_try2.mat', {'s_k_l': selected_known_labels})
# sio.savemat('selected_unknown_labels_try2.mat', {'s_unk_l': selected_unknown_labels})

# 这里通过F0特征空间对以筛选出的已知类别样本和未知类别样本进行“校正”
# 在F0的特征空间中进行蔓延
F0 = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
					embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
					window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
					drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
					norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
					# use_checkpoint=False
					use_checkpoint=True
					)  # the feature dim is 1024
F0.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)   # use the pretrained model trained on ImageNet
F0.apply(inplace_relu)
F0 = F0.cuda()
F0.eval()
for p in F0.parameters():
	p.requires_grad_(False)  # freeze F0

with torch.no_grad():
	F0K, _ = my_gather_features_with_label(F0, knownloader)
	F0U, _ = my_gather_features_with_unknown_label(F0, unknownloader)
F0test = np.concatenate((F0K, F0U), 0)

F0test_selected_known = F0test[known_test_list]
F0test_selected_unknown = F0test[unknown_test_list]
# F0test_selected = np.concatenate((F0test_selected_known, F0test_selected_unknown), 0)
ytest_process = copy.deepcopy(ypred_rough)
# ytest_process = ytest_process[known_test_list + unknown_test_list]
rest_list = list(set(list(range(xK.shape[0]+xU.shape[0]))) - set(known_test_list) - set(unknown_test_list))
ytest_process[known_test_list] = 0   # 筛选出的已知类别样本标签为第0类
ytest_process[unknown_test_list] = 1   # 筛选出的未知类别样本标签为第1类
ytest_process[rest_list] = 2   # 筛选出的未知类别样本标签为第1类
# ytest_process = ytest_process[known_test_list + unknown_test_list]

# k近邻分类器
# 直接用sklearn库会导致自己与自己的距离最小，如何去除自己再计算距离？？？
knn = KNeighborsClassifier(n_neighbors=11, p=2)   # p=2表示欧氏距离
knn.fit(F0test, ytest_process)
pred_selected_known = knn.predict(F0test_selected_known)
pred_selected_unknown = knn.predict(F0test_selected_unknown)
# neighborpoint = knn.kneighbors(F_candidate, n_neighbors=11, return_distance=True)
dist_11_known, indices_11_known = knn.kneighbors(F0test_selected_known, n_neighbors=11, return_distance=True)
dist_11_unknown, indices_11_unknown = knn.kneighbors(F0test_selected_unknown, n_neighbors=11, return_distance=True)
dist_10_known = dist_11_known[:, 1:11]
dist_10_unknown = dist_11_unknown[:, 1:11]
indices_10_known = indices_11_known[:, 1:11]
indices_10_unknown = indices_11_unknown[:, 1:11]
pred_10_known = ytest_process[indices_10_known]
pred_10_unknown = ytest_process[indices_10_unknown]

# 开始判断，判断准则：先数量，后距离，同时计算置信度
known_test_list_final = copy.deepcopy(known_test_list)
unknown_test_list_final = copy.deepcopy(unknown_test_list)
for i in range(F0test_selected_known.shape[0]):
	# print(i)
	pred_this_sample = pred_10_known[i, :]
	the_count = np.bincount(pred_this_sample)
	# sorted_value, sorted_index = torch.sort(the_count)   # 从小到大排序，返回排序后的值和索引
	sorted_value = np.sort(the_count)   # 从小到大排序，返回排序后的值和索引
	sorted_index = np.argsort(the_count)   # 从小到大排序，返回排序后的值和索引
	sorted_value = sorted_value.tolist()
	sorted_index = sorted_index.tolist()
	if len(sorted_value) > 1:
		if sorted_value[-1] != sorted_value[-2]:   # 邻域内数量最多的类别
			pred_label = sorted_index[-1]
		else:   # 假设邻域内有多个类别数量相同且最多，则分为最近的类别
			# print(sorted_index)
			# print(sorted_value)
			most_indices = sorted_index[sorted_index.index(sorted_value == sorted_value[-1]):]
			# print(dist_10_known)
			# print(most_indices)
			their_dists = dist_10_known[i, :][most_indices]
			pred_label = most_indices[np.argmin(their_dists)]
	else:
		pred_label = sorted_index[-1]
	if pred_label == 0:   # 若仍判别为已知类别
		pass
	elif pred_label == 1:   # 若筛选出的已知类别样本在这里被判别为未知类别
		known_test_list_final.remove(known_test_list[i])   # 从筛选出的已知类别列表中删除这个样本
		# unknown_test_list_final.append(known_test_list[i])  # 从筛选出的未知类别列表中添加这个样本
	else:
		known_test_list_final.remove(known_test_list[i])  # 从筛选出的已知类别列表中删除这个样本

for i in range(F0test_selected_unknown.shape[0]):
	# print(i)
	pred_this_sample = pred_10_unknown[i, :]
	the_count = np.bincount(pred_this_sample)
	# sorted_value, sorted_index = torch.sort(the_count)   # 从小到大排序，返回排序后的值和索引
	sorted_value = np.sort(the_count)   # 从小到大排序，返回排序后的值和索引
	sorted_index = np.argsort(the_count)   # 从小到大排序，返回排序后的值和索引
	sorted_value = sorted_value.tolist()
	sorted_index = sorted_index.tolist()
	if len(sorted_value) > 1:
		if sorted_value[-1] != sorted_value[-2]:   # 邻域内数量最多的类别
			pred_label = sorted_index[-1]
		else:   # 假设邻域内有多个类别数量相同且最多，则分为最近的类别
			most_indices = sorted_index[sorted_index.index(sorted_value == sorted_value[-1]):]
			their_dists = dist_10_unknown[i, :][most_indices]
			pred_label = most_indices[np.argmin(their_dists)]
	else:
		pred_label = sorted_index[-1]
	if pred_label == 1:   # 若仍判别为未知类别
		pass
	elif pred_label == 0:   # 若筛选出的未知类别样本在这里被判别为已知类别
		unknown_test_list_final.remove(unknown_test_list[i])   # 从筛选出的未知类别列表中删除这个样本
		# known_test_list_final.append(unknown_test_list[i])  # 从筛选出的已知类别列表中添加这个样本
	else:
		unknown_test_list_final.remove(unknown_test_list[i])  # 从筛选出的未知类别列表中删除这个样本

known_test_list = known_test_list_final
unknown_test_list = unknown_test_list_final


# ################################ 最终所有经过初始筛选选出的已知类别、未知类别样本的数量和准确率 ########################################
print('{} samples have been totally selected for known classes / the whole {} known-class samples'.format(len(known_test_list), xK.shape[0]))
print('{} samples have been totally selected for unknown classes / the whole {} unknown-class samples'.format(len(unknown_test_list), xU.shape[0]))
num_true_test_known = sum(i < xK.shape[0] for i in known_test_list)
num_true_test_unknown = sum(i >= xK.shape[0] for i in unknown_test_list)
num_true_classified_known = len(list(set(list_correct_known).intersection(set(known_test_list))))
print('the ratio of the known-class samples correctly selected finally is : ', num_true_test_known / len(known_test_list))   # 蔓延出的已知类别确实是已知类别的概率
print('the ratio of the unknown-class samples correctly selected finally is : ', num_true_test_unknown / len(unknown_test_list))   # 蔓延出的未知类别确实是未知类别的概率
print('the ratio of the correctly classified samples is : ', num_true_classified_known / len(known_test_list))   # 蔓延出的已知类别样本中分类用的伪标签正确的概率
# ############################################################################################################################

# sys.exit(0)

# alltestset = MyDataset20210827(alltestset)
# alltesttransset = MyDataset20210827(alltesttransset)

availabletestset_known = torch.utils.data.Subset(alltestset, known_test_list)
availabletestset_unknown = torch.utils.data.Subset(alltestset, unknown_test_list)
availabletesttransset_known = torch.utils.data.Subset(alltesttransset, known_test_list)
availabletesttransset_unknown = torch.utils.data.Subset(alltesttransset, unknown_test_list)

# availabletestset_known = MyDataset20210827(availabletestset_known)
# availabletestset_unknown = MyDataset20210827(availabletestset_unknown)
# availabletesttransset_known = MyDataset20210827(availabletesttransset_known)
# availabletesttransset_unknown = MyDataset20210827(availabletesttransset_unknown)

# newtrainloader_known = torch.utils.data.DataLoader(availabletesttransset_known, batch_size=64, shuffle=True, num_workers=cfg['dataloader_workers'])
# newtrainloader_unknown = torch.utils.data.DataLoader(availabletesttransset_unknown, batch_size=128, shuffle=True, num_workers=cfg['dataloader_workers'])
# newtrainloader_known = torch.utils.data.DataLoader(availabletesttransset_known, batch_size=50, shuffle=True, num_workers=cfg['dataloader_workers'])
# newtrainloader_unknown = torch.utils.data.DataLoader(availabletesttransset_unknown, batch_size=100, shuffle=True, num_workers=cfg['dataloader_workers'])
newtrainloader_known = torch.utils.data.DataLoader(availabletesttransset_known, batch_size=16, shuffle=True, num_workers=cfg['dataloader_workers'])
newtrainloader_unknown = torch.utils.data.DataLoader(availabletesttransset_unknown, batch_size=32, shuffle=True, num_workers=cfg['dataloader_workers'])

del F0, knn, pred_selected_known, pred_selected_unknown, dist_11_known, indices_11_known, dist_11_unknown, indices_11_unknown, dist_10_known
del dist_10_unknown, indices_10_known, indices_10_unknown, pred_10_known, pred_10_unknown
del xtest, Ftest, F0test

# 接下来不再采用之前的无放回抽样，而是采用有放回采样，每次都在整个测试集中进行筛选，迭代次数作为一个超参数经验选定或实验讨论敏感度
# ####################  训练一种新的GAN，其中GAN的判别器有4个——区分真假的D1（对抗训练）、区分开集闭集的D2（对抗训练）、以及进行2类分类的C1、进行（k+1）类分类的C2 #############################
NUM_CLASSES = 10
RESTORE_MODE = False  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
# OUTPUT_PATH = '../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210827_unified/'
# OUTPUT_PATH = '../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210909_unified/'
OUTPUT_PATH = '../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210909_unified_CIFAR10/'
CRITIC_ITERS = 1  # How many iterations to train the discriminator
GENER_ITERS = 1  # How many iterations to train the generator
N_GPUS = 1  # Number of GPUs
# BATCH_SIZE = 50  # Batch size. Must be a multiple of N_GPUS
BATCH_SIZE = 16  # Batch size. Must be a multiple of N_GPUS
# END_ITER = 500  # How many iterations to train for
# END_ITER = 50  # How many iterations to train for
# END_ITER = 20  # How many iterations to train for
# END_ITER = 12  # How many iterations to train for
END_ITER = 4  # How many iterations to train for
# END_ITER = 10  # How many iterations to train for
LAMBDA = 10  # Gradient penalty lambda hyperparameter
ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1.  # How to scale generator's ACGAN loss relative to WGAN loss

# for SGD optimizers
# LR = 0.01
LR = 0.02
# LR = 0.05

# for RMSprop optimizers
# LR = 0.00005
# LR = 0.0001
# switch_iter = 10
# if args.dataset == 'TinyImageNet'
switch_iter = 2


# 网络权重初始化
def weights_init(m):
	if isinstance(m, nn.Linear):
		if m.weight is not None:
			init.xavier_uniform_(m.weight)
		if m.bias is not None:
			init.constant_(m.bias, 0.0)


# 用插值数据计算梯度惩罚
def calc_gradient_penalty(netD, real_features, fake_features, batch_size):   # real/fake data: [64, 1000 ]
	alpha = torch.rand(batch_size, 1)
	alpha = alpha.expand(batch_size, int(real_features.nelement()/batch_size)).contiguous()
	alpha = alpha.view(batch_size, -1)
	alpha = alpha.cuda().half()
	fake_data = fake_features.view(batch_size, -1)
	interpolates = alpha * real_features.detach() + ((1 - alpha) * fake_data.detach())
	interpolates = interpolates.cuda().half()
	interpolates.requires_grad_(True)
	disc_interpolates = netD(interpolates)
	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty


# 生成带标签的随机噪声
def gen_rand_noise_with_label1(label=None, batch_size=BATCH_SIZE):
	if label is None:
		label = np.random.randint(0, NUM_CLASSES + 1, batch_size)
	noise = np.random.normal(0, 1, (batch_size, 128))
	prefix = np.zeros((batch_size, NUM_CLASSES + 1))
	prefix[np.arange(batch_size), label] = 1
	noise[np.arange(batch_size), :NUM_CLASSES + 1] = prefix[np.arange(batch_size)]
	noise = torch.from_numpy(noise).float()
	noise = noise.cuda().half()
	return noise


F_last = F
for p in F_last.parameters():
	p.requires_grad_(False)  # freeze F_last

F = copy.deepcopy(F_last)


# 开始迭代：
# 设置超参数迭代次数T
T = 10
for T_num in range(T):
	print('The iteration now is : ', T_num)

	if T_num == 0:
		pass
	else:
		aG = GoodGenerator()
		aD1 = GoodDiscriminator_noclass()
		aD2 = GoodDiscriminator_noclass()
		# aD1 = GoodDiscriminator_noclass(NUM_CLASSES + 1)
		# aD2 = GoodDiscriminator_noclass(NUM_CLASSES + 1)
		aD3 = GoodDiscriminator_onlyclass(2)
		aD4 = GoodDiscriminator_onlyclass(NUM_CLASSES + 1)
		aG.apply(weights_init)
		aD1.apply(weights_init)
		aD2.apply(weights_init)
		aD3.apply(weights_init)
		aD4.apply(weights_init)
		aG.apply(inplace_relu)
		aD1.apply(inplace_relu)
		aD2.apply(inplace_relu)
		aD3.apply(inplace_relu)
		aD4.apply(inplace_relu)
		aG = aG.cuda().half()
		aD1 = aD1.cuda().half()
		aD2 = aD2.cuda().half()
		aD3 = aD3.cuda().half()
		aD4 = aD4.cuda().half()
		aG.train()
		aD1.train()
		aD2.train()
		aD3.train()
		aD4.train()
		# F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
		#                     embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
		#                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
		#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
		#                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
		#                     # use_checkpoint=False
		#                     use_checkpoint=True
		#                     )  # the feature dim is 1024
		# F.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)  # use the pretrained model trained on ImageNet
		F = F.cuda().half()
		# F.train()
		optimizer_g = torch.optim.SGD(aG.parameters(), lr=LR)
		optimizer_d1 = torch.optim.SGD(aD1.parameters(), lr=LR)
		optimizer_d2 = torch.optim.SGD(aD2.parameters(), lr=LR)
		# optimizer_g = torch.optim.RMSprop(aG.parameters(), lr=LR)
		# optimizer_d1 = torch.optim.RMSprop(aD1.parameters(), lr=LR)
		# optimizer_d2 = torch.optim.RMSprop(aD2.parameters(), lr=LR)
		optimizer_d3 = torch.optim.SGD(aD3.parameters(), lr=LR)
		optimizer_d4 = torch.optim.SGD(aD4.parameters(), lr=LR)
		optimizer_F = torch.optim.SGD(F.parameters(), lr=0.00001)
		aux_criterion = nn.CrossEntropyLoss()
		# one = torch.FloatTensor([1])  # 1
		# mone = one * -1  # -1
		# one = one.cuda()
		# mone = mone.cuda()

		# 开始训练GAN
		iteration = START_ITER
		while (iteration < END_ITER):
			print(iteration)
			# # 只用筛选出的未知类别样本和训练集训练GAN
			# for batch_idx, data in enumerate(zip(dataloader, cycle(newtrainloader_unknown))):
			# 采用训练集、筛选出的已知类别样本和未知类别样本共同训练GAN中的生成器、判别器和分类器
			# for batch_idx, data in enumerate(zip(dataloader, cycle(newtrainloader_known), cycle(newtrainloader_unknown))):   # batch size分别是64、64、128
			# for batch_idx, data in enumerate(zip(dataloader, cycle(newtrainloader_known), cycle(newtrainloader_unknown))):   # batch size分别是32、32、64
			# for batch_idx, data in enumerate(zip(dataloader, cycle(newtrainloader_known), cycle(newtrainloader_unknown))):   # batch size分别是50、50、100
			for batch_idx, data in enumerate(zip(dataloader, cycle(newtrainloader_known), cycle(newtrainloader_unknown))):   # batch size分别是32、32、64

				# if batch_idx > 2:
				#     break

				inputs_close, targets_close = data[0]
				inputs_close_selected, _ = data[1]
				inputs_open, _ = data[2]

				if inputs_close.shape[1] == 3:
					pass
				else:
					inputs_close = inputs_close.repeat(1, 3, 1, 1)

				if inputs_close_selected.shape[1] == 3:
					pass
				else:
					inputs_close_selected = inputs_close_selected.repeat(1, 3, 1, 1)

				if inputs_open.shape[1] == 3:
					pass
				else:
					inputs_open = inputs_open.repeat(1, 3, 1, 1)

				# ######################################################################################################################################
				# # 由于修改了dataset读取数据，读取的内容新增加了在原始alltestset的索引与权重，可根据索引判断是否为蔓延样本，从而赋予对应索引的权重，这里的加权形式采用的是loss加权
				# inputs_close_selected, _, index_close_selected, confidence_close_selected = data[1]
				# inputs_open, _, index_open, confidence_open = data[2]
				# ######################################################################################################################################

				# targets_close = torch.Tensor([mapping[x] for x in targets_close]).long().cuda().half()
				targets_close = targets_close.long().cuda()
				# targets_close_selected = my_gather_labels_mix(F, C, aD3_init, inputs_close_selected)
				F_last.eval()
				F_last.half()
				with torch.no_grad():
					if T_num == 0:
						C.eval()
						C.half()
						# targets_close_selected = my_gather_labels(F, C, inputs_close_selected.cuda())
						targets_close_selected = my_gather_labels(F_last, C, inputs_close_selected.cuda().half())
					else:
						C_last.eval()
						C_last.half()
						targets_close_selected = my_gather_labels(F_last, C_last, inputs_close_selected.cuda().half())
				targets_open = (NUM_CLASSES * torch.ones(inputs_open.shape[0])).long().cuda().half()  # 把筛选出的未知类别样本作为第k+1类
				batch_size = inputs_close.shape[0] + inputs_close_selected.shape[0] + inputs_open.shape[0]  # 整体的batch_size: 64+64+128=256

				# if T_num == 0:
				#     confidence_real = torch.cat((torch.ones(inputs_close.shape[0]), confidence_close_selected, confidence_open))
				# else:
				#     confidence_real = torch.ones(batch_size)
				# confidence_real = confidence_real.cuda()

				# ---------------------TRAIN G------------------------
				# 训练生成器
				# 冻结判别器参数
				for p in aD1.parameters():
					p.requires_grad_(False)  # freeze D1
				for p in aD2.parameters():
					p.requires_grad_(False)  # freeze D2
				for p in aD3.parameters():
					p.requires_grad_(False)  # freeze D3
				for p in aD4.parameters():
					p.requires_grad_(False)  # freeze D4
				#
				if iteration < switch_iter:
					F.eval()
					aG.train()
					for p in F.parameters():
						p.requires_grad_(False)
					for p in aG.parameters():
						p.requires_grad_(True)

				else:
					F.train()
					F.half()
					aG.eval()
					for p in F.parameters():
						p.requires_grad_(True)
					for p in aG.parameters():
						p.requires_grad_(False)

				# # 冻结特征提取网络的参数
				# for p in C.parameters():
				#     p.requires_grad_(False)  # freeze C
				# for p in aD3_init.parameters():
				#     p.requires_grad_(False)  # freeze C

				gen_cost = None

				if iteration < switch_iter:

					for i in range(GENER_ITERS):
						print("Generator iters: " + str(i))
						aG.zero_grad()

						# 产生一组随机整数标签
						# f_label = np.random.randint(0, NUM_CLASSES + 1, batch_size)
						f_label_known = np.random.randint(0, NUM_CLASSES, int(batch_size / 2))
						f_label_unknown = np.random.randint(NUM_CLASSES, NUM_CLASSES + 1, batch_size - int(batch_size / 2))
						f_label = np.concatenate((f_label_known, f_label_unknown), 0)
						# f_label_bi = f_label
						f_label_bi = copy.deepcopy(f_label)
						f_label_bi[np.where(f_label_bi < NUM_CLASSES)] = 0   # 对于二分类问题，已知类别标签为0
						f_label_bi[np.where(f_label_bi == NUM_CLASSES)] = 1   # 对于二分类问题，未知类别标签为1
						# print(f_label)
						# print(f_label_bi)
						# 根据标签产生随机噪声
						noise = gen_rand_noise_with_label1(f_label, batch_size)
						# 噪声需要梯度
						noise.requires_grad_(True)
						# 生成假数据
						fake_features = aG(noise)
						fake_label = torch.from_numpy(f_label).long().cuda()

						# 判别器输出生成损失和生成辅助输出
						# gen_cost1 = aD1(fake_features, make_one_hot(fake_label, NUM_CLASSES + 1))
						gen_cost1 = aD1(fake_features)
						gen_cost1 = -gen_cost1.mean()
						# gen_cost1 = -(confidence_real * gen_cost1).mean()

						# gen_cost2 = aD2(fake_features, make_one_hot(fake_label, NUM_CLASSES + 1))
						gen_cost2 = aD2(fake_features)
						gen_cost2_known = gen_cost2[np.where(f_label < NUM_CLASSES)]
						gen_cost2_unknown = gen_cost2[np.where(f_label == NUM_CLASSES)]
						gen_cost2 = gen_cost2_known.mean() - gen_cost2_unknown.mean()
						# gen_cost2 = (confidence_real * gen_cost2_known).mean() - (confidence_real * gen_cost2_unknown).mean()

						gen_aux_output_bi = aD3(fake_features)
						aux_label_bi = torch.from_numpy(f_label_bi).long()
						aux_label_bi = aux_label_bi.cuda().half()
						aux_errG_bi = aux_criterion(gen_aux_output_bi, aux_label_bi.long()).mean()
						# aux_errG_bi = (confidence_real * aux_criterion(gen_aux_output_bi, aux_label_bi)).mean()

						gen_aux_output = aD4(fake_features)
						aux_label = torch.from_numpy(f_label).long()
						aux_label = aux_label.cuda().half()
						aux_errG = aux_criterion(gen_aux_output, aux_label.long()).mean()
						# aux_errG = (confidence_real * aux_criterion(gen_aux_output, aux_label)).mean()

						# 计算分类准确率
						total1 = 0
						correctDist1 = 0
						total2 = 0
						correctDist2 = 0
						total3 = 0
						correctDist3 = 0
						gen_aux_output1 = gen_aux_output[np.where(f_label < NUM_CLASSES)]
						_, predicted = gen_aux_output1.max(1)
						aux_label1 = torch.from_numpy(f_label[np.where(f_label < NUM_CLASSES)]).long().cuda().half()
						total1 = total1 + aux_label1.shape[0]
						correctDist1 = correctDist1 + predicted.eq(aux_label1).sum().item()
						print('Acc: {%.3f%%}' % (100. * correctDist1 / total1))
						_, predicted = gen_aux_output.max(1)
						aux_label = torch.from_numpy(f_label).long().cuda().half()
						total2 = total2 + aux_label.shape[0]
						correctDist2 = correctDist2 + predicted.eq(aux_label).sum().item()
						print('Acc: {%.3f%%}' % (100. * correctDist2 / total2))
						_, predicted = gen_aux_output_bi.max(1)
						total3 = total3 + aux_label_bi.shape[0]
						correctDist3 = correctDist3 + predicted.eq(aux_label_bi).sum().item()
						print('Acc: {%.3f%%}' % (100. * correctDist3 / total3))

						# print(aux_errG)
						# print(aux_errG_bi)
						# print(gen_cost1)
						# print(gen_cost2)

						# 生成器总的损失函数
						# g_cost = ACGAN_SCALE_G * aux_errG + gen_cost1 + gen_cost2  # 训练生成器时不填加附加损失函数
						# g_cost = ACGAN_SCALE_G * aux_errG + gen_cost1 + 0.1 * gen_cost2  # 训练生成器时不填加附加损失函数
						g_cost = ACGAN_SCALE_G * (aux_errG + aux_errG_bi) + gen_cost1 + 0.1 * gen_cost2  # 训练生成器时不填加附加损失函数
						# g_cost = ACGAN_SCALE_G * (aux_errG + aux_errG_bi) + gen_cost1 + 1 * gen_cost2  # 训练生成器时不填加附加损失函数
						# g_cost = ACGAN_SCALE_G * aux_errG + 0.1 * gen_cost2  # 训练生成器时不填加附加损失函数
						# g_cost = ACGAN_SCALE_G * aux_errG + gen_cost1 + 0.01 * gen_cost2  # 训练生成器时不填加附加损失函数
						print('g_cost is: ', g_cost)
						g_cost.backward()
						# torch.nn.utils.clip_grad_norm_(aG.parameters(), max_norm=1, norm_type=2)  # 直接做梯度裁剪
						torch.nn.utils.clip_grad_value_(aG.parameters(), 1)  # 直接做梯度裁剪
						# torch.nn.utils.clip_grad_value_(aD1.parameters(), 1)  # 直接做梯度裁剪
						# torch.nn.utils.clip_grad_value_(aD2.parameters(), 1)  # 直接做梯度裁剪
						# torch.nn.utils.clip_grad_value_(aD3.parameters(), 1)  # 直接做梯度裁剪
						# torch.nn.utils.clip_grad_value_(aD4.parameters(), 1)  # 直接做梯度裁剪
						optimizer_g.step()

						# 查看网络各层梯度
						# for name, param in aG.named_parameters():
						#     print('层:', name, param.size())
						#     print('权值梯度', param.grad)
						#     print('权值', param)

					# ---------------------TRAIN D------------------------
				# 训练判别器
				# 取消冻结判别器的参数
				if iteration < switch_iter:
					for p in aD1.parameters():  # reset requires_grad
						p.requires_grad_(True)  # they are set to False before in training G
					for p in aD2.parameters():  # reset requires_grad
						p.requires_grad_(True)  # they are set to False before in training G

					#     # weight clipping
					# for p in aD1.parameters():
					#     p.data.clamp_(-0.01, 0.01)
					# for p in aD2.parameters():
					#     p.data.clamp_(-0.01, 0.01)

				else:
					for p in aD1.parameters():
						p.requires_grad_(False)
					for p in aD2.parameters():
						p.requires_grad_(False)

				for p in aD3.parameters():  # reset requires_grad
					p.requires_grad_(True)  # they are set to False before in training G
				for p in aD4.parameters():  # reset requires_grad
					p.requires_grad_(True)  # they are set to False before in training G
				#     # weight clipping
				# for p in aD3.parameters():
				#     p.data.clamp_(-0.01, 0.01)
				# for p in aD4.parameters():
				#     p.data.clamp_(-0.01, 0.01)

				for i in range(CRITIC_ITERS):
					print("Critic iter: " + str(i))

					aD1.zero_grad()
					aD2.zero_grad()
					aD3.zero_grad()
					aD4.zero_grad()
					F.zero_grad()

					# gen fake data and load real data
					# 生成随机整数标签
					# f_label = np.random.randint(0, NUM_CLASSES + 1, batch_size)
					f_label_known = np.random.randint(0, NUM_CLASSES, int(batch_size / 2))
					f_label_unknown = np.random.randint(NUM_CLASSES, NUM_CLASSES + 1, batch_size - int(batch_size / 2))
					f_label = np.concatenate((f_label_known, f_label_unknown), 0)
					fake_label = torch.from_numpy(f_label).long().cuda()
					# f_label_bi = f_label
					f_label_bi = copy.deepcopy(f_label)
					f_label_bi[np.where(f_label_bi < NUM_CLASSES)] = 0   # 对于二分类问题，已知类别的标签为0
					f_label_bi[np.where(f_label_bi == NUM_CLASSES)] = 1   # 对于二分类问题，未知类别的标签为1

					# 产生带标签的随机噪声
					noise = gen_rand_noise_with_label1(f_label, batch_size)
					with torch.no_grad():
						# 赋值给noisev
						noisev = noise  # totally freeze G, training D
					# 通过noisev生成假数据并与梯度图分离开
					fake_features = aG(noisev).detach()

					# 取真实图像和真实标签
					# real_data = torch.cat((inputs_close, inputs_open), 0)
					real_data = torch.cat((inputs_close, inputs_close_selected, inputs_open), 0)
					# real_label = torch.cat((targets_close, targets_open), 0)
					real_label = torch.cat((targets_close, targets_close_selected, targets_open), 0)
					# real_label_bi = real_label
					real_label_bi = copy.deepcopy(real_label)
					real_label_bi[torch.where(real_label_bi < NUM_CLASSES)] = 0   # 对于二分类问题，已知类别的标签为0
					real_label_bi[torch.where(real_label_bi == NUM_CLASSES)] = 1   # 对于二分类问题，未知类别的标签为1
					real_data = real_data.cuda().half()

					# 提取真实图像的特征
					# if hasattr(torch.cuda, 'empty_cache'):
					#     torch.cuda.empty_cache()
					real_features = F(real_data)
					# if hasattr(torch.cuda, 'empty_cache'):
					#     torch.cuda.empty_cache()
					real_features = real_features.view(batch_size, -1)

					# train with real data
					# 用真实数据训练

					# 判别器真实损失函数是辅助输出和真实标签的分类交叉熵损失函数
					aux_output_bi = aD3(real_features)
					aux_errD_real_bi = aux_criterion(aux_output_bi, real_label_bi.long())
					errD_real_bi = aux_errD_real_bi.mean()
					# errD_real_bi = (confidence_real * aux_errD_real_bi).mean()
					aux_output = aD4(real_features)
					aux_errD_real = aux_criterion(aux_output, real_label.long())
					errD_real = aux_errD_real.mean()
					# errD_real = (confidence_real * aux_errD_real).mean()
					# 生成样本
					aux_output_bi_fake = aD3(fake_features)
					aux_errD_bi_fake = aux_criterion(aux_output_bi_fake, torch.from_numpy(f_label_bi).long().cuda())
					errD_fake_bi = aux_errD_bi_fake.mean()
					# errD_fake_bi = (confidence_real * aux_errD_bi_fake).mean()
					aux_output_fake = aD4(fake_features)
					aux_errD_fake = aux_criterion(aux_output_fake, torch.from_numpy(f_label).long().cuda())
					errD_fake = aux_errD_fake.mean()
					# errD_fake = (confidence_real * aux_errD_fake).mean()
					# disc_acgan = errD_real + errD_real_bi  # + errD_fake
					disc_acgan = errD_real + errD_real_bi + errD_fake + errD_fake_bi  # + errD_fake   # 真实特征与生成特征的分类损失
					# disc_acgan = errD_real + errD_real_bi + ((iteration)/(END_ITER)) * (errD_fake + errD_fake_bi)  # + errD_fake   # 真实特征与生成特征的分类损失
					print('disc_acgan is: ', disc_acgan)

					# disc_acgan.backward()
					# optimizer_d3.step()
					# optimizer_d4.step()

					# 计算分类准确率
					total1 = 0
					correctDist1 = 0
					total2 = 0
					correctDist2 = 0
					total3 = 0
					correctDist3 = 0
					# _, predicted = aux_output[np.where(real_label < NUM_CLASSES)].max(1)
					_, predicted = aux_output[torch.where(real_label < NUM_CLASSES)].max(1)
					# total1 = total1 + real_label[np.where(real_label < NUM_CLASSES)].size(0)
					total1 = total1 + real_label[torch.where(real_label < NUM_CLASSES)].size(0)
					# correctDist1 = correctDist1 + predicted.eq(real_label[np.where(real_label < NUM_CLASSES)]).sum().item()
					correctDist1 = correctDist1 + predicted.eq(real_label[torch.where(real_label < NUM_CLASSES)]).sum().item()
					print('Acc: {%.3f%%}' % (100. * correctDist1 / total1))
					_, predicted = aux_output.max(1)
					total2 = total2 + real_label.size(0)
					correctDist2 = correctDist2 + predicted.eq(real_label).sum().item()
					print('Acc: {%.3f%%}' % (100. * correctDist2 / total2))
					_, predicted = aux_output_bi.max(1)
					total3 = total3 + real_label_bi.size(0)
					correctDist3 = correctDist3 + predicted.eq(real_label_bi).sum().item()
					print('Acc: {%.3f%%}' % (100. * correctDist3 / total3))

					if iteration < switch_iter:
						# 判别器输出对真实数据的输出损失和辅助输出
						# 对于D1
						# disc_real1 = aD1(real_features, make_one_hot(real_label, NUM_CLASSES + 1))
						disc_real1 = aD1(real_features)
						disc_real1 = disc_real1.mean()
						# disc_real1 = (confidence_real * disc_real1).mean()

						# 对于D2
						# disc_cost2_r = aD2(real_features, make_one_hot(real_label, NUM_CLASSES + 1))
						disc_cost2_r = aD2(real_features)
						# disc_cost2_r_known = disc_cost2_r[np.where(real_label < NUM_CLASSES)]
						disc_cost2_r_known = disc_cost2_r[torch.where(real_label < NUM_CLASSES)]
						# disc_cost2_r_unknown = disc_cost2_r[np.where(real_label == NUM_CLASSES)]
						disc_cost2_r_unknown = disc_cost2_r[torch.where(real_label == NUM_CLASSES)]
						disc_cost2_r = disc_cost2_r_unknown.mean() - disc_cost2_r_known.mean()
						# disc_cost2_r = (confidence_real * disc_cost2_r_unknown).mean() - (confidence_real * disc_cost2_r_known).mean()

						# train with fake data
						# 用假数据训练
						# 判别器输出对假数据的输出损失和辅助输出
						# disc_fake1 = aD1(fake_features, make_one_hot(fake_label, NUM_CLASSES + 1))
						disc_fake1 = aD1(fake_features)
						disc_fake1 = disc_fake1.mean()
						# disc_fake1 = (confidence_real * disc_fake1).mean()

						# 对于D2
						# disc_cost2_f = aD2(fake_features, make_one_hot(fake_label, NUM_CLASSES + 1))
						disc_cost2_f = aD2(fake_features)
						disc_cost2_f_known = disc_cost2_f[np.where(f_label < NUM_CLASSES)]
						disc_cost2_f_unknown = disc_cost2_f[np.where(f_label == NUM_CLASSES)]
						disc_cost2_f = disc_cost2_f_unknown.mean() - disc_cost2_f_known.mean()
						# disc_cost2_f = (confidence_real * disc_cost2_f_unknown).mean() - (confidence_real * disc_cost2_f_known).mean()

						# # train with interpolates data
						# 用插值数据计算梯度惩罚
						gradient_penalty1 = calc_gradient_penalty(aD1, real_features, fake_features, batch_size)
						# gradient_penalty1 = calc_gradient_penalty(aD1, real_features, real_label, fake_features, fake_label, batch_size)

						disc_cost1 = disc_fake1 - disc_real1 + gradient_penalty1
						# disc_cost1 = disc_fake1 - disc_real1
						print('D1_cost is: ', disc_cost1)
						# # disc_cost1.backward()
						w_dist1 = disc_fake1 - disc_real1  # 推土机距离
						print('The wasserstein distance1 is: ', w_dist1)

						# torch.nn.utils.clip_grad_value_(aD1.parameters(), 2)  # 直接做梯度裁剪

						# optimizer_d1.step()

						# print(real_features.shape)
						# print(fake_features.shape)
						# print(int(batch_size/2))

						gradient_penalty2_r = calc_gradient_penalty(aD2, real_features[:int(batch_size/2)], real_features[-int(batch_size/2):], int(batch_size/2))
						# gradient_penalty2_r = calc_gradient_penalty(aD2, real_features[:int(batch_size/2)], real_label[:int(batch_size/2)], real_features[-int(batch_size/2):], real_label[-int(batch_size/2):], int(batch_size/2))
						gradient_penalty2_f = calc_gradient_penalty(aD2, fake_features[:int(batch_size/2)], fake_features[-int(batch_size/2):], int(batch_size/2))
						# gradient_penalty2_f = calc_gradient_penalty(aD2, fake_features[:int(batch_size/2)], fake_label[:int(batch_size/2)], fake_features[-int(batch_size/2):], fake_label[-int(batch_size/2):], int(batch_size/2))

						disc_cost2 = disc_cost2_r + disc_cost2_f + 1 * gradient_penalty2_r + 1 * gradient_penalty2_f
						# disc_cost2 = disc_cost2_r + disc_cost2_f
						print('D2_cost is: ', disc_cost2)
						# disc_cost2.backward()

						# (disc_cost1 + 0.1 * disc_cost2).backward()
						# (disc_cost1 + 0.1 * disc_cost2 + disc_acgan).backward()
						(disc_cost1 + 1 * disc_cost2 + disc_acgan).backward()

					else:
						(disc_acgan).backward()

					w_dist2 = (disc_cost2_r_unknown.mean() + disc_cost2_f_unknown.mean()) - (disc_cost2_r_known.mean() + disc_cost2_f_known.mean())  # 推土机距离
					print('The wasserstein distance2 is: ', w_dist2)
					# # torch.nn.utils.clip_grad_norm_(aD2.parameters(), max_norm=20, norm_type=2)   # 直接做梯度裁剪

					# torch.nn.utils.clip_grad_value_(aG.parameters(), 1)   # 直接做梯度裁剪
					if iteration < switch_iter:
						torch.nn.utils.clip_grad_value_(aD1.parameters(), 1)   # 直接做梯度裁剪
						torch.nn.utils.clip_grad_value_(aD2.parameters(), 1)   # 直接做梯度裁剪
						torch.nn.utils.clip_grad_value_(aD3.parameters(), 1)   # 直接做梯度裁剪
						torch.nn.utils.clip_grad_value_(aD4.parameters(), 1)   # 直接做梯度裁剪
					else:
						torch.nn.utils.clip_grad_value_(aD3.parameters(), 1)  # 直接做梯度裁剪
						torch.nn.utils.clip_grad_value_(aD4.parameters(), 1)  # 直接做梯度裁剪
						torch.nn.utils.clip_grad_value_(F.parameters(), 1)  # 直接做梯度裁剪

					# 查看网络各层梯度
					# for name, param in aD2.named_parameters():
					#     print('层:', name, param.size())
					#     print('权值梯度', param.grad)
					#     print('权值', param)

					if iteration < switch_iter:
						optimizer_d1.step()
						optimizer_d2.step()
						optimizer_d3.step()
						optimizer_d4.step()

					else:
						optimizer_d3.step()
						optimizer_d4.step()
						optimizer_F.step()

				# if iteration % 50 == 49:
				if (iteration % END_ITER) == (END_ITER - 1):
					# ----------------------Save model----------------------
					# 保存整个模型
					if not os.path.isdir('../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210909_unified_CIFAR10/{}'.format('OOD1')):
						os.mkdir('../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210909_unified_CIFAR10/{}'.format('OOD1'))

					torch.save(aG, OUTPUT_PATH + "{}/generator_iter{}_T{}".format('OOD1', str(iteration), str(T_num)) + ".pt")
					torch.save(aD1, OUTPUT_PATH + "{}/discriminator1_iter{}_T{}".format('OOD1', str(iteration), str(T_num)) + ".pt")
					torch.save(aD2, OUTPUT_PATH + "{}/discriminator2_iter{}_T{}".format('OOD1', str(iteration), str(T_num)) + ".pt")
					torch.save(aD3, OUTPUT_PATH + "{}/discriminator3_iter{}_T{}".format('OOD1', str(iteration), str(T_num)) + ".pt")
					torch.save(aD4, OUTPUT_PATH + "{}/discriminator4_iter{}_T{}".format('OOD1', str(iteration), str(T_num)) + ".pt")
					torch.save(F, OUTPUT_PATH + "{}/F_iter{}_T{}".format('OOD1', str(iteration), str(T_num)) + ".pt")

			iteration += 1

		del F, aG, aD1, aD2

	aD3 = GoodDiscriminator_onlyclass(2)  # 2类分类器
	aD3 = torch.load(OUTPUT_PATH + "{}/discriminator3_iter{}_T{}".format('OOD1', str(END_ITER - 1), str(T_num)) + ".pt")
	aD3 = aD3.cuda().half()
	aD3.eval()
	aD4 = GoodDiscriminator_onlyclass(NUM_CLASSES + 1)  # k+1类分类器
	aD4 = torch.load(OUTPUT_PATH + "{}/discriminator4_iter{}_T{}".format('OOD1', str(END_ITER - 1), str(T_num)) + ".pt")
	aD4 = aD4.cuda().half()
	aD4.eval()
	F_last = torch.load(OUTPUT_PATH + "{}/F_iter{}_T{}".format('OOD1', str(END_ITER - 1), str(T_num)) + ".pt")
	F_last = F_last.cuda().half()
	F_last.eval()
	# for p in aG.parameters():  # reset requires_grad
	#     p.requires_grad_(False)  # they are set to False before in training G
	for p in aD3.parameters():  # reset requires_grad
		p.requires_grad_(False)  # they are set to False before in training G
	for p in aD4.parameters():  # reset requires_grad
		p.requires_grad_(False)  # they are set to False before in training G
	for p in F_last.parameters():  # reset requires_grad
		p.requires_grad_(False)  # they are set to False before in training G

	# 对于已知类别测试集
	known_class_list = []
	unknown_class_list = []
	ypred_multi = []
	for i, (inputs, labels) in enumerate(knownloader):  # batch size = 50
		batch_size = inputs.shape[0]
		if inputs.shape[1] == 3:
			pass
		else:
			inputs = inputs.repeat(1, 3, 1, 1)
		features = F_last(inputs.cuda().half()).view(batch_size, -1)
		outputs_bi = aD3(features)
		outputs_multi = aD4(features)
		pred_bi = torch.argmax(outputs_bi, 1)
		pred_multi = torch.argmax(outputs_multi, 1)
		ypred_multi += outputs_multi.cpu().detach().tolist()
		for j in range(batch_size):
			# id = i * 50 + j
			id = i * 16 + j
			if (pred_bi[j] == 0) and (pred_multi[j] < 10):
				# ####################### modified in 2021.10.11 ##################################
				known_class_list.append(id)

				# if id < FK.shape[0]:
				#     known_class_list.append(id)
				# else:
				#     pass
				# ##################################################################################
			elif (pred_bi[j] == 1) and (pred_multi[j] == 10):
				unknown_class_list.append(id)
			else:
				pass

	# 对于未知类别测试集
	for i, (inputs, labels) in enumerate(unknownloader):  # batch size = 64
		batch_size = inputs.shape[0]
		if inputs.shape[1] == 3:
			pass
		else:
			inputs = inputs.repeat(1, 3, 1, 1)
		features = F_last(inputs.cuda().half()).view(batch_size, -1)
		outputs_bi = aD3(features)
		outputs_multi = aD4(features)
		pred_bi = torch.argmax(outputs_bi, 1)
		pred_multi = torch.argmax(outputs_multi, 1)
		ypred_multi += outputs_multi.cpu().detach().tolist()
		for j in range(batch_size):
			# id = i * 50 + j + FK.shape[0]  # 注意这里是个易错点，之前直接计算xU中的ID而忽略了在整体测试集中的ID，导致大量真实已知类别样本被筛选为未知类别！！！！！！！！！！
			id = i * 16 + j + FK.shape[0]  # 注意这里是个易错点，之前直接计算xU中的ID而忽略了在整体测试集中的ID，导致大量真实已知类别样本被筛选为未知类别！！！！！！！！！！
			if (pred_bi[j] == 1) and (pred_multi[j] == 10):
				unknown_class_list.append(id)
			elif (pred_bi[j] == 0) and (pred_multi[j] < 10):
				# ############################# modified in 2021.10.11 ###########################
				known_class_list.append(id)

				# pass
				# ################################################################################
			else:
				pass
	ypred_multi = np.asarray(ypred_multi)
	ypred_multi = np.argmax(ypred_multi, 1)

	print('{} samples have been selected for known classes / the whole {} known-class samples'.format(len(known_class_list),
																									  FK.shape[0]))
	print('{} samples have been selected for unknown classes / the whole {} unknown-class samples'.format(
		len(unknown_class_list), FU.shape[0]))
	num_true_known = sum(i < FK.shape[0] for i in known_class_list)
	num_true_unknown = sum(i >= FK.shape[0] for i in unknown_class_list)
	list_correct_known = np.where(ypred_multi[known_class_list] == ytest[known_class_list])[0].tolist()
	num_true_classified_known = len(list(set(list_correct_known)))  # 去除多余的0
	print('the ratio of the known-class samples correctly selected is : ',
		  num_true_known / len(known_class_list))  # 筛选出的已知类别确实是已知类别的概率
	print('the ratio of the unknown-class samples correctly selected is : ',
		  num_true_unknown / len(unknown_class_list))  # 筛选出的未知类别确实是未知类别的概率
	print('the ratio of the correctly classified samples is : ',
		  num_true_classified_known / len(known_class_list))  # 筛选出的已知类别样本中分类用的伪标签正确的概率

	# 在已经筛选出的样本中做进一步的筛除
	with torch.no_grad():
		F_last = F_last.float()
		FK, _ = my_gather_features_with_label(F_last, knownloader)
		FU, _ = my_gather_features_with_unknown_label(F_last, unknownloader)
	Ftest = np.concatenate((FK, FU), 0)
	Ftest_selected_known = Ftest[known_class_list]
	Ftest_selected_unknown = Ftest[unknown_class_list]
	# F0test_selected = np.concatenate((F0test_selected_known, F0test_selected_unknown), 0)
	ytest_process = copy.deepcopy(ypred_rough)
	# ytest_process = ytest_process[known_test_list + unknown_test_list]
	rest_list = list(set(list(range(FK.shape[0] + FU.shape[0]))) - set(known_class_list) - set(unknown_class_list))
	ytest_process[known_class_list] = 0  # 筛选出的已知类别样本标签为第0类
	ytest_process[unknown_class_list] = 1  # 筛选出的未知类别样本标签为第1类
	ytest_process[rest_list] = 2  # 筛选出的未知类别样本标签为第1类
	# k近邻分类器
	# 直接用sklearn库会导致自己与自己的距离最小，如何去除自己再计算距离？？？
	knn = KNeighborsClassifier(n_neighbors=11, p=2)  # p=2表示欧氏距离
	knn.fit(Ftest, ytest_process)
	pred_selected_known = knn.predict(Ftest_selected_known)
	pred_selected_unknown = knn.predict(Ftest_selected_unknown)
	# neighborpoint = knn.kneighbors(F_candidate, n_neighbors=11, return_distance=True)
	dist_11_known, indices_11_known = knn.kneighbors(Ftest_selected_known, n_neighbors=11, return_distance=True)
	dist_11_unknown, indices_11_unknown = knn.kneighbors(Ftest_selected_unknown, n_neighbors=11, return_distance=True)
	dist_10_known = dist_11_known[:, 1:11]
	dist_10_unknown = dist_11_unknown[:, 1:11]
	indices_10_known = indices_11_known[:, 1:11]
	indices_10_unknown = indices_11_unknown[:, 1:11]
	pred_10_known = ytest_process[indices_10_known]
	pred_10_unknown = ytest_process[indices_10_unknown]
	# 开始判断，判断准则：先数量，后距离，同时计算置信度
	known_test_list_final = copy.deepcopy(known_class_list)
	unknown_test_list_final = copy.deepcopy(unknown_class_list)
	for i in range(Ftest_selected_known.shape[0]):
		# print(i)
		pred_this_sample = pred_10_known[i, :]
		the_count = np.bincount(pred_this_sample)
		# sorted_value, sorted_index = torch.sort(the_count)   # 从小到大排序，返回排序后的值和索引
		sorted_value = np.sort(the_count)  # 从小到大排序，返回排序后的值和索引
		sorted_index = np.argsort(the_count)  # 从小到大排序，返回排序后的值和索引
		sorted_value = sorted_value.tolist()
		sorted_index = sorted_index.tolist()
		if len(sorted_value) > 1:
			if sorted_value[-1] != sorted_value[-2]:  # 邻域内数量最多的类别
				pred_label = sorted_index[-1]
			else:  # 假设邻域内有多个类别数量相同且最多，则分为最近的类别
				# print(sorted_index)
				# print(sorted_value)
				most_indices = sorted_index[sorted_index.index(sorted_value == sorted_value[-1]):]
				# print(dist_10_known)
				# print(most_indices)
				their_dists = dist_10_known[i, :][most_indices]
				pred_label = most_indices[np.argmin(their_dists)]
		else:
			pred_label = sorted_index[-1]
		if pred_label == 0:  # 若仍判别为已知类别
			pass
		elif pred_label == 1:  # 若筛选出的已知类别样本在这里被判别为未知类别
			known_test_list_final.remove(known_class_list[i])  # 从筛选出的已知类别列表中删除这个样本
			# unknown_test_list_final.append(known_test_list[i])  # 从筛选出的未知类别列表中添加这个样本
		else:
			known_test_list_final.remove(known_class_list[i])  # 从筛选出的已知类别列表中删除这个样本

	for i in range(Ftest_selected_unknown.shape[0]):
		# print(i)
		pred_this_sample = pred_10_unknown[i, :]
		the_count = np.bincount(pred_this_sample)
		# sorted_value, sorted_index = torch.sort(the_count)   # 从小到大排序，返回排序后的值和索引
		sorted_value = np.sort(the_count)  # 从小到大排序，返回排序后的值和索引
		sorted_index = np.argsort(the_count)  # 从小到大排序，返回排序后的值和索引
		sorted_value = sorted_value.tolist()
		sorted_index = sorted_index.tolist()
		if len(sorted_value) > 1:
			if sorted_value[-1] != sorted_value[-2]:  # 邻域内数量最多的类别
				pred_label = sorted_index[-1]
			else:  # 假设邻域内有多个类别数量相同且最多，则分为最近的类别
				most_indices = sorted_index[sorted_index.index(sorted_value == sorted_value[-1]):]
				their_dists = dist_10_unknown[i, :][most_indices]
				pred_label = most_indices[np.argmin(their_dists)]
		else:
			pred_label = sorted_index[-1]
		if pred_label == 1:  # 若仍判别为未知类别
			pass
		elif pred_label == 0:  # 若筛选出的未知类别样本在这里被判别为已知类别
			unknown_test_list_final.remove(unknown_class_list[i])  # 从筛选出的未知类别列表中删除这个样本
			# known_test_list_final.append(unknown_test_list[i])  # 从筛选出的已知类别列表中添加这个样本
		else:
			unknown_test_list_final.remove(unknown_class_list[i])  # 从筛选出的未知类别列表中删除这个样本

	known_class_list = known_test_list_final
	unknown_class_list = unknown_test_list_final

	del knn, pred_selected_known, pred_selected_unknown, dist_11_known, indices_11_known, dist_11_unknown, indices_11_unknown, dist_10_known
	del dist_10_unknown, indices_10_known, indices_10_unknown, pred_10_known, pred_10_unknown
	del Ftest

	# ################################ 最终所有经过初始筛选选出的已知类别、未知类别样本的数量和准确率 ########################################
	print('{} samples have been totally selected for known classes / the whole {} known-class samples'.format(
		len(known_class_list), xK.shape[0]))
	print('{} samples have been totally selected for unknown classes / the whole {} unknown-class samples'.format(
		len(unknown_class_list), xU.shape[0]))
	num_true_test_known = sum(i < xK.shape[0] for i in known_class_list)
	num_true_test_unknown = sum(i >= xK.shape[0] for i in unknown_class_list)
	num_true_classified_known = len(list(set(list_correct_known).intersection(set(known_class_list))))
	print('the ratio of the known-class samples correctly selected finally is : ', num_true_test_known / len(known_class_list))  # 蔓延出的已知类别确实是已知类别的概率
	print('the ratio of the unknown-class samples correctly selected finally is : ', num_true_test_unknown / len(unknown_class_list))  # 蔓延出的未知类别确实是未知类别的概率
	print('the ratio of the correctly classified samples is : ', num_true_classified_known / len(known_class_list))  # 蔓延出的已知类别样本中分类用的伪标签正确的概率
	# ############################################################################################################################

	# 重新在测试集中选取下一步迭代要用到的已知类别样本和未知类别样本
	availabletestset_known = torch.utils.data.Subset(alltestset, known_class_list)
	availabletestset_unknown = torch.utils.data.Subset(alltestset, unknown_class_list)
	availabletesttransset_known = torch.utils.data.Subset(alltesttransset, known_class_list)
	availabletesttransset_unknown = torch.utils.data.Subset(alltesttransset, unknown_class_list)

	# newtrainloader_known = torch.utils.data.DataLoader(availabletesttransset_known, batch_size=50, shuffle=True, num_workers=cfg['dataloader_workers'])
	# newtrainloader_unknown = torch.utils.data.DataLoader(availabletesttransset_unknown, batch_size=100, shuffle=True, num_workers=cfg['dataloader_workers'])
	newtrainloader_known = torch.utils.data.DataLoader(availabletesttransset_known, batch_size=16, shuffle=True, num_workers=cfg['dataloader_workers'])
	newtrainloader_unknown = torch.utils.data.DataLoader(availabletesttransset_unknown, batch_size=32, shuffle=True, num_workers=cfg['dataloader_workers'])

	C_last = aD4
	F_last = F_last.half()
	F = copy.deepcopy(F_last)


# #########################################################################################################################

# for i in range(len(allnowtransset_known)):
#     print(i)
#     # print(allnowtransset_known[0])
#     print(allnowtransset_known[i][0])
#     print(allnowtransset_known[i][1])
# print('----------------------------------------------')
# for i in range(len(availabletesttransset_known)):
#     print(i)
# # print(availabletesttransset_known[0])
#     print(availabletesttransset_known[i][0])
#     print(availabletesttransset_known[i][1])
# print('----------------------------------------------')
# for i in range(len(knowntransset)):
#     print(i)
#     # print(knowntransset[0])
#     print(knowntransset[i][0])
#     print(knowntransset[i][1])

# set_name = 'try3_recur_20210816_C_method4_withG_unified'
# save_name = '{}_{}_{}'.format(args.dataset, args.trial, set_name)
# print('Saving..')
# torch.save(C_new2, 'networks/weights/{}/'.format(args.dataset) + save_name + 'Accuracy.pth')

