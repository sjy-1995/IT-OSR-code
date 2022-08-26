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
from torchvision import models

# swin transformer as the backbone
from swin_transformer import SwinTransformer   # the more complex file


parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--dataset', required=True, type=str, help='Dataset for training',
                    choices=['MNIST', 'SVHN', 'CIFAR10', 'CIFAR+10', 'CIFAR+50', 'TinyImageNet'])
parser.add_argument('--trial', required=True, type=int, help='Trial number, 0-4 provided')
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
    cfg = json.load(config_file)[args.dataset]


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


trainloader, valloader, _, mapping = dataHelper.get_train_loaders(args.dataset, args.trial, cfg)
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
C = Classifier(cfg['num_known_classes'])
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


optimizer_F = optim.AdamW(F.parameters(), lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
optimizer_C = optim.AdamW(C.parameters(), lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)


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
        targets = torch.Tensor([mapping[x] for x in targets]).long()
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
            targets = torch.Tensor([mapping[x] for x in targets]).long()
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

    # Save checkpoint and model
    # state = {
    #     'net': F.state_dict(),
    #     'acc': acc,
    #     'epoch': epoch,
    # }
    if not os.path.isdir('networks/weights/{}'.format(args.dataset)):
        os.mkdir('networks/weights/{}'.format(args.dataset))
    if args.dataset == 'CIFAR+10':
        if not os.path.isdir('networks/weights/CIFAR+50'):
            os.mkdir('networks/weights/CIFAR+50')
    # save_name = '{}_{}_{}F_and_C_unified'.format(args.dataset, args.trial, args.name)

    print('Saving..')
    torch.save(F, 'networks/weights/{}/{}_{}_{}F_'.format(args.dataset, args.dataset, args.trial, args.name) + 'Newest.pth')
    torch.save(C, 'networks/weights/{}/{}_{}_{}C_'.format(args.dataset, args.dataset, args.trial, args.name) + 'Newest.pth')
    if args.dataset == 'CIFAR+10':
        torch.save(F, 'networks/weights/CIFAR+50/{}_{}_{}F_'.format('CIFAR+50', args.trial, args.name) + 'Newest.pth')
        torch.save(C, 'networks/weights/CIFAR+50/{}_{}_{}C_'.format('CIFAR+50', args.trial, args.name) + 'Newest.pth')

    if total_loss <= best_loss:
        print('Saving..')
        torch.save(F, 'networks/weights/{}/{}_{}_{}F_'.format(args.dataset, args.dataset, args.trial, args.name) + 'TotalLoss.pth')
        torch.save(C, 'networks/weights/{}/{}_{}_{}C_'.format(args.dataset, args.dataset, args.trial, args.name) + 'TotalLoss.pth')
        best_loss = total_loss
        if args.dataset == 'CIFAR+10':
            torch.save(F, 'networks/weights/CIFAR+50/{}_{}_{}F_'.format('CIFAR+50', args.trial, args.name) + 'TotalLoss.pth')
            torch.save(C, 'networks/weights/CIFAR+50/{}_{}_{}C_'.format('CIFAR+50', args.trial, args.name) + 'TotalLoss.pth')

    if acc >= best_acc:
        print('Saving..')
        torch.save(F, 'networks/weights/{}/{}_{}_{}F_'.format(args.dataset, args.dataset, args.trial, args.name) + 'Accuracy.pth')
        torch.save(C, 'networks/weights/{}/{}_{}_{}C_'.format(args.dataset, args.dataset, args.trial, args.name) + 'Accuracy.pth')
        best_acc = acc

        if args.dataset == 'CIFAR+10':
            torch.save(F, 'networks/weights/CIFAR+50/{}_{}_{}F_'.format('CIFAR+50', args.trial, args.name) + 'Accuracy.pth')
            torch.save(C, 'networks/weights/CIFAR+50/{}_{}_{}C_'.format('CIFAR+50', args.trial, args.name) + 'Accuracy.pth')


max_epoch = 400 + start_epoch


for epoch in range(start_epoch, max_epoch):
    train(epoch)
    val(epoch)

