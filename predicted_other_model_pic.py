import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torchmetrics.functional import dice_score
# from torchmetrics import Dice
# from tensorboardX import SummaryWriter
import dataset
import logging
import model_net
from model_net import *
from dataset import *
from PIL import Image
import pdb
from medpy import metric
from torchvision.datasets import ImageFolder
import os
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torchvision.transforms as TF

from model.segnet import SegNet
from model.unet_model import R2U_Net, AttU_Net, R2AttU_Net, U_Net
# from model.unext import UNext
from model.transunet_model import TransUNet
from model.sknet import SKNet26
from model.nestedUNet import NestedUNet

from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
import argparse



image_files = [os.path.join("/home/deeplearning/lzp/DRCNet_pytorch/pic", filename)
               for filename in os.listdir("/home/deeplearning/lzp/DRCNet_pytorch/pic")
               if filename.endswith(".png")]
            
# 按照文件名排序（假设文件名以“数字_数字_模型名称.png”的格式命名）
# pdb.set_trace()
# image_files = sorted(image_files, key=lambda x: (tuple(map(int, os.path.basename(x).split('_')[:2])), os.path.basename(x).split('_')[-1]))

#model_order = {"DRCNET": 0, "UNet": 1, "SegNet": 2, "R2U_Net": 3, "AttU_Net": 4, "R2AttU_Net": 5, "NestedUNet": 6, "AAUnet": 7}

#image_files = sorted(image_files, key=lambda x: (tuple(map(int, os.path.basename(x).split('_')[:2])), model_order[os.path.basename(x).split('_')[-1]]))


# model_order = {"DRCNET": 0, "UNet": 1, "SegNet": 2, "R2UNet": 3, "AttUNet": 4, "R2AttUNet": 5, "NestedUNet": 6, "AAUnet": 7}

# image_files = sorted(image_files, key=lambda x: (tuple(map(int, os.path.basename(x).split('_')[:-2])), model_order[os.path.basename(x).split('_')[-1].split('.')[0]]))

model_order = {"DRCNET": 0, "UNet": 1, "SegNet": 2, "R2UNet": 3, "AttUNet": 4, "R2AttUNet": 5, "NestedUNet": 6, "AAUnet": 7}

def sort_key(file_name):
    parts = os.path.basename(file_name).split('_')
    num_parts = tuple(map(int, parts[:-1]))
    model_name = parts[-1].split('.')[0]
    model_order_value = model_order[model_name]
    return (num_parts, model_order_value)

image_files = sorted(image_files, key=sort_key)

# pdb.set_trace()


# pdb.set_trace()
# 将图像文件划分为每七个图像的列表
image_lists = [image_files[i:i+8] for i in range(0, len(image_files), 8)]
# pdb.set_trace()
# 遍历每个子列表，并将七个图像放在同一行中
for image_list in image_lists:
    fig, axs = plt.subplots(1, 8, figsize=(20, 5))
    for i, image_file in enumerate(image_list):
        image = plt.imread(image_file)
        axs[i].imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])
        axs[i].axis('off')
        model_name = os.path.basename(image_file).split('_')[2].split('.')[0]  # 从文件名中获取模型名称
        axs[i].text(0.5, -0.1, model_name, size=10, ha="center", transform=axs[i].transAxes)  # 在图像下方添加模型名称
    plt.savefig('/home/deeplearning/lzp/DRCNet_pytorch/pic_row/row_{}.png'.format(image_files.index(image_list[0])))
    plt.close()

