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
import datasets_unified.utils_version3_trytransductive_3 as dataHelper1
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

# swin transformer as the backbone
from swin_transformer import SwinTransformer   # the more complex file

# 载入sklearn中的k近邻分类器方法
from sklearn.neighbors import KNeighborsClassifier


parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--dataset', required=True, type=str, help='Dataset for training', choices=['MNIST', 'SVHN', 'CIFAR10', 'CIFAR+10', 'CIFAR+50', 'TinyImageNet'])
parser.add_argument('--trial', required=True, type=int, help='Trial number, 0-4 provided')
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
        labels = torch.Tensor([mapping[x] for x in labels])
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
        labels = torch.Tensor([mapping[x] for x in labels])
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
    cfg = json.load(config_file)[args.dataset]


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
        # labels = torch.Tensor([mapping[x] for x in labels])

        # labels = torch.Tensor([unknown_classes.index(x) + 6 for x in labels])
        labels = torch.Tensor([unknown_classes.index(x) + cfg['num_known_classes'] for x in labels])

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
        # labels = torch.Tensor([mapping[x] for x in labels])

        # labels = torch.Tensor([unknown_classes.index(x) + 6 for x in labels])
        labels = torch.Tensor([unknown_classes.index(x) + cfg['num_known_classes'] for x in labels])

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


# with open('datasets/config_swin_vit_GAN20210820.json') as config_file:   # batch size = 128
#     cfg = json.load(config_file)[args.dataset]

with open("../cac_openset/cac-openset-master/datasets_unified/{}/class_splits/{}.json".format(args.dataset, args.trial)) as f:
    class_splits = json.load(f)
    known_classes = class_splits['Known']

# unknown_classes = list(set(list(range(10))) - set(known_classes))   # 已知类与未知类类别数量总共10类

with open("../cac_openset/cac-openset-master/datasets_unified/{}/class_splits/{}.json".format(args.dataset, args.trial)) as f:
    class_splits = json.load(f)
    unknown_classes = class_splits['Unknown']

print('The known classes are: ', known_classes)
print('The unknown classes are: ', unknown_classes)

F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                    embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                    window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    # use_checkpoint=False
                    use_checkpoint=True
                    )  # the feature dim is 1024
# F = torch.load('networks/weights/{}/{}_{}_backbone20210816_F_Accuracy.pth'.format(args.dataset, args.dataset, args.trial))
F = torch.load('networks/weights/{}/{}_{}_backbone20210816_F_Newest.pth'.format(args.dataset, args.dataset, args.trial))
F.apply(inplace_relu)
F = F.cuda()
F.eval()
for p in F.parameters():
    p.requires_grad_(False)  # freeze F

C = Classifier(cfg['num_known_classes'])
# C = torch.load('networks/weights/{}/{}_{}_backbone20210816_C_Accuracy.pth'.format(args.dataset, args.dataset, args.trial))
C = torch.load('networks/weights/{}/{}_{}_backbone20210816_C_Newest.pth'.format(args.dataset, args.dataset, args.trial))
C.apply(inplace_relu)
C = C.cuda()
C.eval()
for p in C.parameters():
    p.requires_grad_(False)  # freeze C


# dataloader, dataloader_eval, mapping = dataHelper1.get_train_loaders(args.dataset, args.trial, cfg)
# knownloader, unknownloader, alltestset, alltesttransset, mapping = dataHelper1.get_eval_loaders(args.dataset, args.trial, cfg)   # 未打乱数据集顺序


# ############################################################################################################################
# 根据已训练好的k类分类器（SoftMax输出的负熵作为分数，训练集的平均分数作为阈值）、训练初始GAN得到的k+1类分类器对测试集进行初步筛选
# 用k+1类分类器筛选：
# ############### 改变batch size大小以适应GPU显存 ###################
# with open('datasets/config_swin_vit_GAN20210820_smaller.json') as config_file:   # batch size = 64
# with open('datasets/config_swin_vit_GAN20210820_smaller2.json') as config_file:   # batch size = 50
# with open('datasets/config_swin_vit_GAN20210820_exp6.json') as config_file:   # batch size = 32
# with open('datasets/config_swin_vit_GAN20210820_smaller3.json') as config_file:   # batch size = 16
#     cfg = json.load(config_file)[args.dataset]
dataloader, dataloader_eval, mapping = dataHelper1.get_train_loaders(args.dataset, args.trial, cfg)
knownloader, unknownloader, alltestset, alltesttransset, mapping = dataHelper1.get_eval_loaders(args.dataset, args.trial, cfg)   # 未打乱数据集顺序
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

# # 这里开始进行数据“蔓延”
# # 在F的特征空间中进行蔓延
# candidate_list = list(set(list(range(xK.shape[0] + xU.shape[0]))) - set(known_test_list) - set(unknown_test_list))
# F_candidate = Ftest[candidate_list]
# ytest_process = copy.deepcopy(ypred_rough)
# # ytest_process[unknown_test_list] = cfg['num_known_classes']   # 筛选出的未知类别样本标签为第k+1类
# # ytest_process[candidate_list] = cfg['num_known_classes'] + 1   # 剩余的待定类别样本标签暂且设为第k+2类
# ytest_process[known_test_list] = 0   # 筛选出的已知类别样本标签为第0类
# ytest_process[unknown_test_list] = 1   # 筛选出的未知类别样本标签为第1类
# ytest_process[candidate_list] = 2   # 剩余的待定类别样本标签暂且设为第2类

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

# 手动，不再依赖于sklearn库，但是距离矩阵的计算导致内存不足
# 计算距离矩阵
# dist_matrix = torch.norm(F_candidate[:, None] - Ftest, dim=2, p=2)  # (, )
# print(dist_matrix.shape)
# # 找到距离最小的11个位置（因为包含自己与自己的距离（0），所以是10+1=11）
# sorted_dist, indices = torch.sort(dist_matrix)   # 默认按行从小到大排序
# # _, indices = torch.sort(dist_matrix)   # 默认按行从小到大排序，暂时不需要求出具体的距离值
# indices_10 = indices[:, 1:11]
# dist_10 = sorted_dist[:, 1:11]
# pred_10 = ytest_process[indices_10]
# print(pred_10)
# print(pred_10.shape)

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


# # spread_known_list = np.asarray(candidate_list)[torch.where(pred_10_ < cfg['num_known_classes'])].tolist()
# # spread_unknown_list = np.asarray(candidate_list)[torch.where(pred_10_ == cfg['num_known_classes'])].tolist()
# # spread_known_confidence_ratio = confidence_ratio[torch.where(pred_10_ < cfg['num_known_classes'])]
# # spread_unknown_confidence_ratio = confidence_ratio[torch.where(pred_10_ == cfg['num_known_classes'])]
# spread_known_list = np.asarray(candidate_list)[torch.where(pred_10_ == 0)].tolist()
# spread_unknown_list = np.asarray(candidate_list)[torch.where(pred_10_ == 1)].tolist()
# spread_known_confidence_ratio = confidence_ratio[torch.where(pred_10_ == 0)]
# spread_unknown_confidence_ratio = confidence_ratio[torch.where(pred_10_ == 1)]

# # 输出蔓延的数据数量和准确率
# print('{} samples have been spread for known classes / the whole {} known-class samples'.format(len(spread_known_list), xK.shape[0]))
# print('{} samples have been spread for unknown classes / the whole {} unknown-class samples'.format(len(spread_unknown_list), xU.shape[0]))
#
# num_true_spread_known = sum(i < xK.shape[0] for i in spread_known_list)
# num_true_spread_unknown = sum(i >= xK.shape[0] for i in spread_unknown_list)
# # pred_spread_known = pred_10_[torch.where(pred_10_ < cfg['num_known_classes'])]
# # pred_spread_known = pred_10_[torch.where(pred_10_ == 0)]
# # list_correct_spread_known = np.where(np.asarray(pred_spread_known) == ytest[spread_known_list])[0].tolist()   # 最近邻分类的分类效果不如直接用分类器的输出进行预测
# list_correct_spread_known = np.where(ypred_rough[spread_known_list] == ytest[spread_known_list])[0].tolist()
# num_true_classified_known = len(list(set(list_correct_spread_known)))
# print('the ratio of the known-class samples correctly spread is : ', num_true_spread_known / len(spread_known_list))   # 蔓延出的已知类别确实是已知类别的概率
# print('the ratio of the unknown-class samples correctly spread is : ', num_true_spread_unknown / len(spread_unknown_list))   # 蔓延出的未知类别确实是未知类别的概率
# print('the ratio of the correctly classified samples is : ', num_true_classified_known / len(spread_known_list))   # 蔓延出的已知类别样本中分类用的伪标签正确的概率


# # 在F0的特征空间中进行蔓延
# F0 = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
#                     embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
#                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                     use_checkpoint=False
#                     )  # the feature dim is 1024
# F0.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)   # use the pretrained model trained on ImageNet
# F0 = F0.cuda()
# F0.eval()
# for p in F0.parameters():
#     p.requires_grad_(False)  # freeze F0
#
# with torch.no_grad():
#     F0K, _ = my_gather_features_with_label(F0, knownloader)
#     F0U, _ = my_gather_features_with_unknown_label(F0, unknownloader)
# F0test = np.concatenate((F0K, F0U), 0)
#
# F0_candidate = F0test[candidate_list]
#
# # k近邻分类器
# # 直接用sklearn库会导致自己与自己的距离最小，如何去除自己再计算距离？？？
# knn = KNeighborsClassifier(n_neighbors=11, p=2)   # p=2表示欧氏距离
# knn.fit(F0test, ytest_process)
# pred0_candidate = knn.predict(F0_candidate)
# # neighborpoint = knn.kneighbors(F_candidate, n_neighbors=11, return_distance=True)
# dist0_11, indices0_11 = knn.kneighbors(F0_candidate, n_neighbors=11, return_distance=True)
# dist0_10 = dist0_11[:, 1:11]
# indices0_10 = indices0_11[:, 1:11]
# pred0_10 = ytest_process[indices0_10]
#
# # 手动，不再依赖于sklearn库，但是距离矩阵的计算导致内存不足
# # 计算距离矩阵
# # dist_matrix = torch.norm(F_candidate[:, None] - Ftest, dim=2, p=2)  # (, )
# # print(dist_matrix.shape)
# # # 找到距离最小的11个位置（因为包含自己与自己的距离（0），所以是10+1=11）
# # sorted_dist, indices = torch.sort(dist_matrix)   # 默认按行从小到大排序
# # # _, indices = torch.sort(dist_matrix)   # 默认按行从小到大排序，暂时不需要求出具体的距离值
# # indices_10 = indices[:, 1:11]
# # dist_10 = sorted_dist[:, 1:11]
# # pred_10 = ytest_process[indices_10]
# # print(pred_10)
# # print(pred_10.shape)
#
# # 开始判断，判断准则：先数量，后距离，同时计算置信度
# pred0_10_ = torch.zeros(F0_candidate.shape[0])
# confidence0_ratio = torch.zeros(F0_candidate.shape[0])
# for i in range(F0_candidate.shape[0]):
#     # print(i)
#     pred_this_sample = pred0_10[i, :]
#     the_count = np.bincount(pred_this_sample)
#     # sorted_value, sorted_index = torch.sort(the_count)   # 从小到大排序，返回排序后的值和索引
#     sorted_value = np.sort(the_count)   # 从小到大排序，返回排序后的值和索引
#     sorted_index = np.argsort(the_count)   # 从小到大排序，返回排序后的值和索引
#     sorted_value = sorted_value.tolist()
#     sorted_index = sorted_index.tolist()
#     if len(sorted_value) > 1:
#         if sorted_value[-1] != sorted_value[-2]:   # 邻域内数量最多的类别
#             pred0_10_[i] = sorted_index[-1]
#         else:   # 假设邻域内有多个类别数量相同且最多，则分为最近的类别
#             most_indices = sorted_index[sorted_index.index(sorted_value == sorted_value[-1]):]
#             their_dists = dist_10[i, :][most_indices]
#             pred0_10_[i] = most_indices[np.argmin(their_dists)]
#     else:
#         pred0_10_[i] = sorted_index[-1]
#     confidence0_ratio[i] = sorted_value[-1] / 10
#
# # spread0_known_list = np.asarray(candidate_list)[torch.where(pred0_10_ < cfg['num_known_classes'])].tolist()
# # spread0_unknown_list = np.asarray(candidate_list)[torch.where(pred0_10_ == cfg['num_known_classes'])].tolist()
# # spread0_known_confidence_ratio = confidence0_ratio[torch.where(pred0_10_ < cfg['num_known_classes'])]
# # spread0_unknown_confidence_ratio = confidence0_ratio[torch.where(pred0_10_ == cfg['num_known_classes'])]
# spread0_known_list = np.asarray(candidate_list)[torch.where(pred0_10_ == 0)].tolist()
# spread0_unknown_list = np.asarray(candidate_list)[torch.where(pred0_10_ == 1)].tolist()
# spread0_known_confidence_ratio = confidence0_ratio[torch.where(pred0_10_ == 0)]
# spread0_unknown_confidence_ratio = confidence0_ratio[torch.where(pred0_10_ == 1)]
#
# # 输出蔓延的数据数量和准确率
# print('{} samples have been spread for known classes / the whole {} known-class samples'.format(len(spread0_known_list), xK.shape[0]))
# print('{} samples have been spread for unknown classes / the whole {} unknown-class samples'.format(len(spread0_unknown_list), xU.shape[0]))
#
# num_true_spread0_known = sum(i < xK.shape[0] for i in spread0_known_list)
# num_true_spread0_unknown = sum(i >= xK.shape[0] for i in spread0_unknown_list)
# # pred_spread0_known = pred0_10_[torch.where(pred0_10_ < cfg['num_known_classes'])]
# # list_correct_spread_known = np.where(np.asarray(pred_spread_known) == ytest[spread_known_list])[0].tolist()   # 最近邻分类的分类效果不如直接用分类器的输出进行预测
# list_correct_spread0_known = np.where(ypred_rough[spread0_known_list] == ytest[spread0_known_list])[0].tolist()
# num_true_classified0_known = len(list(set(list_correct_spread0_known)))
# print('the ratio of the known-class samples correctly spread is : ', num_true_spread0_known / len(spread0_known_list))   # 蔓延出的已知类别确实是已知类别的概率
# print('the ratio of the unknown-class samples correctly spread is : ', num_true_spread0_unknown / len(spread0_unknown_list))   # 蔓延出的未知类别确实是未知类别的概率
# print('the ratio of the correctly classified samples is : ', num_true_classified0_known / len(spread0_known_list))   # 蔓延出的已知类别样本中分类用的伪标签正确的概率
#
# # 根据F特征空间和F0特征空间结合选出最终蔓延的已知类别与未知类别样本
# final_spread_known_list = list(set(spread_known_list).intersection(set(spread0_known_list)))
# final_spread_unknown_list = list(set(spread_unknown_list).intersection(set(spread0_unknown_list)))
#
# print('{} samples have been finally spread for known classes / the whole {} known-class samples'.format(len(final_spread_known_list), xK.shape[0]))
# print('{} samples have been finally spread for unknown classes / the whole {} unknown-class samples'.format(len(final_spread_unknown_list), xK.shape[0]))
#
# # 蔓延样本的置信度是两个特征空间计算出的置信度的平均值
# indices_spread = [spread_known_list.index(x) for x in final_spread_known_list]
# indices_spread0 = [spread0_known_list.index(x) for x in final_spread_known_list]
# confidence_ratio_inter = confidence_ratio[indices_spread]
# confidence0_ratio_inter = confidence0_ratio[indices_spread0]
# final_spread_known_confidence = (confidence_ratio_inter + confidence0_ratio_inter) / 2
# final_spread_known_confidence = final_spread_known_confidence.tolist()   # 全都转化成列表形式
#
# indices_spread = [spread_unknown_list.index(x) for x in final_spread_unknown_list]
# indices_spread0 = [spread0_unknown_list.index(x) for x in final_spread_unknown_list]
# confidence_ratio_inter = confidence_ratio[indices_spread]
# confidence0_ratio_inter = confidence0_ratio[indices_spread0]
# final_spread_unknown_confidence = (confidence_ratio_inter + confidence0_ratio_inter) / 2
# final_spread_unknown_confidence = final_spread_unknown_confidence.tolist()   # 全都转化成列表形式

# sys.exit(0)

# # 自定义一个数据集加载方式，将数据集的索引也作为数据加载的一部分，便于之后从每个batch中筛选出“蔓延”样本从而赋予loss对应的权重
# class MyDataset20210827(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#
#     def __getitem__(self, index):
#         data, target = self.dataset[index]
#         if index in final_spread_known_list:   # 对于蔓延出的已知类别样本
#             confidence = final_spread_known_confidence[final_spread_known_list.index(index)]
#         elif index in final_spread_unknown_list:   # 对于蔓延出的未知类别样本
#             confidence = final_spread_unknown_confidence[final_spread_unknown_list.index(index)]
#         else:
#             confidence = 1
#         return data, target, index, confidence
#
#     def __len__(self):
#         return len(self.dataset)


# known_test_list = known_test_list + final_spread_known_list
# unknown_test_list = unknown_test_list + final_spread_unknown_list

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
NUM_CLASSES = cfg['num_known_classes']
RESTORE_MODE = False  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
# OUTPUT_PATH = '../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210827_unified/'
OUTPUT_PATH = '../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210909_unified/'
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


# def calc_gradient_penalty(netD, real_features, real_labels, fake_features, fake_labels, batch_size):   # real/fake data: [64, 1000 ]
#     alpha = torch.rand(batch_size, 1)
#     alpha_labels = alpha.expand(batch_size, NUM_CLASSES + 1).contiguous()
#     alpha = alpha.expand(batch_size, int(real_features.nelement()/batch_size)).contiguous()
#     alpha = alpha.view(batch_size, -1)
#     alpha_labels = alpha_labels.view(batch_size, -1)
#     alpha = alpha.cuda()
#     alpha_labels = alpha_labels.cuda()
#     fake_data = fake_features.view(batch_size, -1)
#     real_labels = make_one_hot(real_labels, NUM_CLASSES + 1)
#     fake_labels = make_one_hot(fake_labels, NUM_CLASSES + 1)
#     interpolates = alpha * real_features.detach() + (1 - alpha) * fake_data.detach()
#     interpolates_labels = alpha_labels * real_labels.detach() + (1 - alpha_labels) * fake_labels.detach()
#     interpolates = interpolates.cuda()
#     interpolates_labels = interpolates_labels.cuda()
#     interpolates.requires_grad_(True)
#     interpolates_labels.requires_grad_(True)
#     # disc_interpolates = netD(interpolates)
#     disc_interpolates = netD(interpolates.half(), interpolates_labels.half())
#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
#     return gradient_penalty


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
# T = 10
T = 2
for T_num in range(T):
# for T_num in [7, 8, 9]:
# for T_num in [6, 7, 8, 9]:
    print('The iteration now is : ', T_num)

    # if T_num == 1:
        # sys.exit(0)

    if T_num == 0:
    # if T_num == 7:
    # if T_num == 6:
    # if (T_num == 0) or (T_num == 1):
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
        # optimizer_F = torch.optim.SGD(F.parameters(), lr=0.0001)
        optimizer_F = torch.optim.SGD(F.parameters(), lr=0.00001)
        # optimizer_F = torch.optim.SGD(F.parameters(), lr=LR*0.1)
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
                # ######################################## modified in 2021.10.11 #######################################
                inputs_close_selected, _ = data[1]

                # inputs_close_selected, targets_close_selected = data[1]
                # targets_close_selected = torch.Tensor([mapping[x] for x in targets_close_selected]).long().cuda()
                # #######################################################################################################
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

                targets_close = torch.Tensor([mapping[x] for x in targets_close]).long().cuda().half()
                # targets_close_selected = my_gather_labels_mix(F, C, aD3_init, inputs_close_selected)
                F_last.eval()
                F_last.half()

                # ################################### modified in 2021.10.11 ############################################
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
                # #######################################################################################################

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

                        # # 添加于20210913,使生成特征与训练集特征在距离上尽可能接近
                        # loss_dis = torch.norm(real_features - fake_features)
                        # print('loss_dis is : ', loss_dis)
                        # (disc_cost1 + 0.1 * disc_cost2 + disc_acgan).backward()
                        (disc_cost1 + 1 * disc_cost2 + disc_acgan).backward()
                        # (disc_cost1 + 1 * disc_cost2 + disc_acgan + 0.1 * loss_dis).backward()

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
                    if not os.path.isdir('../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210909_unified/{}'.format(args.dataset)):
                        os.mkdir('../../newdisk/sunjiayin/GAN_transductive/outputs_try3_method4_20210909_unified/{}'.format(args.dataset))

                    torch.save(aG, OUTPUT_PATH + "{}/trial{}_generator_iter{}_T{}".format(args.dataset, args.trial, str(iteration), str(T_num)) + ".pt")
                    torch.save(aD1, OUTPUT_PATH + "{}/trial{}_discriminator1_iter{}_T{}".format(args.dataset, args.trial, str(iteration), str(T_num)) + ".pt")
                    torch.save(aD2, OUTPUT_PATH + "{}/trial{}_discriminator2_iter{}_T{}".format(args.dataset, args.trial, str(iteration), str(T_num)) + ".pt")
                    torch.save(aD3, OUTPUT_PATH + "{}/trial{}_discriminator3_iter{}_T{}".format(args.dataset, args.trial, str(iteration), str(T_num)) + ".pt")
                    torch.save(aD4, OUTPUT_PATH + "{}/trial{}_discriminator4_iter{}_T{}".format(args.dataset, args.trial, str(iteration), str(T_num)) + ".pt")
                    torch.save(F, OUTPUT_PATH + "{}/trial{}_F_iter{}_T{}".format(args.dataset, args.trial, str(iteration), str(T_num)) + ".pt")

            iteration += 1


        # ################################### 根据训练好的GAN中两个分类器对整个测试集再次进行筛选（有放回抽样） #################################################
        del F, aG, aD1, aD2

    # aG = GoodGenerator()
    # aG = torch.load(OUTPUT_PATH + "{}/trial{}_generator_iter49_T{}".format(args.dataset, args.trial, str(T)) + ".pt")
    # aG = aG.cuda()
    # aG.eval()
    aD3 = GoodDiscriminator_onlyclass(2)   # 2类分类器
    aD3 = torch.load(OUTPUT_PATH + "{}/trial{}_discriminator3_iter{}_T{}".format(args.dataset, args.trial, str(END_ITER - 1), str(T_num)) + ".pt")
    aD3 = aD3.cuda().half()
    aD3.eval()
    aD4 = GoodDiscriminator_onlyclass(NUM_CLASSES + 1)   # k+1类分类器
    aD4 = torch.load(OUTPUT_PATH + "{}/trial{}_discriminator4_iter{}_T{}".format(args.dataset, args.trial, str(END_ITER - 1), str(T_num)) + ".pt")
    aD4 = aD4.cuda().half()
    aD4.eval()
    F_last = torch.load(OUTPUT_PATH + "{}/trial{}_F_iter{}_T{}".format(args.dataset, args.trial, str(END_ITER - 1), str(T_num)) + ".pt")
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
            if (pred_bi[j] == 0) and (pred_multi[j] < cfg['num_known_classes']):
                # ####################### modified in 2021.10.11 ##################################
                # known_class_list.append(id)

                if id < FK.shape[0]:
                    known_class_list.append(id)
                else:
                    pass
                # ##################################################################################
            elif (pred_bi[j] == 1) and (pred_multi[j] == cfg['num_known_classes']):
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
            if (pred_bi[j] == 1) and (pred_multi[j] == cfg['num_known_classes']):
                unknown_class_list.append(id)
            elif (pred_bi[j] == 0) and (pred_multi[j] < cfg['num_known_classes']):
                # ############################# modified in 2021.10.11 ###########################
                # known_class_list.append(id)

                pass
                # ################################################################################
            else:
                pass
    ypred_multi = np.asarray(ypred_multi)
    ypred_multi = np.argmax(ypred_multi, 1)

    print('{} samples have been selected for known classes / the whole {} known-class samples'.format(len(known_class_list), FK.shape[0]))
    print('{} samples have been selected for unknown classes / the whole {} unknown-class samples'.format(len(unknown_class_list), FU.shape[0]))
    num_true_known = sum(i < FK.shape[0] for i in known_class_list)
    num_true_unknown = sum(i >= FK.shape[0] for i in unknown_class_list)
    list_correct_known = np.where(ypred_multi[known_class_list] == ytest[known_class_list])[0].tolist()
    num_true_classified_known = len(list(set(list_correct_known)))  # 去除多余的0
    print('the ratio of the known-class samples correctly selected is : ', num_true_known / len(known_class_list))  # 筛选出的已知类别确实是已知类别的概率
    print('the ratio of the unknown-class samples correctly selected is : ', num_true_unknown / len(unknown_class_list))  # 筛选出的未知类别确实是未知类别的概率
    print('the ratio of the correctly classified samples is : ', num_true_classified_known / len(known_class_list))  # 筛选出的已知类别样本中分类用的伪标签正确的概率

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
    print('{} samples have been totally selected for known classes / the whole {} known-class samples'.format(len(known_class_list), xK.shape[0]))
    print('{} samples have been totally selected for unknown classes / the whole {} unknown-class samples'.format(len(unknown_class_list), xU.shape[0]))
    num_true_test_known = sum(i < xK.shape[0] for i in known_class_list)
    num_true_test_unknown = sum(i >= xK.shape[0] for i in unknown_class_list)
    num_true_classified_known = len(list(set(list_correct_known).intersection(set(known_class_list))))
    print('the ratio of the known-class samples correctly selected finally is : ', num_true_test_known / len(known_class_list))   # 蔓延出的已知类别确实是已知类别的概率
    print('the ratio of the unknown-class samples correctly selected finally is : ', num_true_test_unknown / len(unknown_class_list))   # 蔓延出的未知类别确实是未知类别的概率
    print('the ratio of the correctly classified samples is : ', num_true_classified_known / len(known_class_list))   # 蔓延出的已知类别样本中分类用的伪标签正确的概率
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

