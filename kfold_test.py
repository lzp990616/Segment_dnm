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
from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
import argparse
from sklearn.model_selection import KFold
from model.segnet import SegNet
from model.unet_model import R2U_Net, AttU_Net, R2AttU_Net, U_Net
# from model.unext import UNext
from model.transunet_model import TransUNet
from model.sknet import SKNet26
from model.nestedUNet import NestedUNet

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.RandomResizedCrop(224),#Resizes all images into same dimension
    # transforms.RandomRoation(10),# Rotates the images upto Max of 10 Degrees
    # transforms.RandomHorizontalFlip(p=0.4),#Performs Horizantal Flip over images

    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Coverts into Tensors
    # transforms.Normalize(mean=mean_nums, std=std_nums)  # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    # normalize
])

transform_test = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224), #Performs Crop at Center and resizes it to 224
    transforms.ToTensor(),
    # transforms.Normalize(mean = mean_nums, std=std_nums) # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

filepath_busi = './data/Dataset_BUSI/Dataset_BUSI_with_GT/'
filepath_bus = './data/BUS/BUS/'

imagefilepath = filepath_busi + 'data_mask/images/'
imagefilepath_label = filepath_busi + 'data_mask/masks/'


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# logger = get_logger('./log/log_M10_0dnm_busi_4folder_8batch.log')
logger = get_logger('./log/RRCNet_busi_4folder.log')
logger.info('start training!')
# logging.info(f'''Starting training:
#     Epochs:          {epochs}
#     Batch size:      {batch_size}
#     Learning rate:   {learning_rate}
#     Training size:   {n_train}
#     Validation size: {n_val}
#     Checkpoints:     {save_checkpoint}
#     Device:          {device.type}
#     Images scaling:  {img_scale}
#     Mixed Precision: {amp}
# ''')


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)


# 训练
def train(train_loader, model, criterion, criterion_dice, epoch, args, DEVICE):
    model.train()  # 模型的训练模式
    sum_total_loss_batch = 0
    sum_total_loss = 0
    for batch_idx, (img, label) in tqdm(enumerate(train_loader)):  # 从dataloader中按照dataset.py的顺序进行数据调用，tqdm为循环时间的可视化
        total_loss = 0
        img, label = img.to(DEVICE), label.to(DEVICE)  # 使用gpu
        model.zero_grad()
        output = model(img)
        output = output.float().cpu()
        label = label.cpu()
        loss = criterion_dice(output, label)

        total_loss += loss
        sum_total_loss += total_loss.data.item()
        sum_total_loss_batch += total_loss.data.item()
        total_loss.backward()  # 反向传播当前梯度
        optimizer.step()  # 进行单次优化更新参数
        if batch_idx % 10 == 0 and batch_idx != 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch, batch_idx * len(img), len(train_loader.dataset), sum_total_loss_batch / 10, loss_sum / 10,
                    optimizer.param_groups[0]['lr']))
            sum_total_loss_batch = 0

    print(
        'Train Epoch: {}\tLoss: {:.6f}'.format(epoch, (sum_total_loss * args.batch_size) / (len(train_loader.dataset))))
    logger.info(
        'Train Epoch: {}\tLoss: {:.6f}'.format(epoch, (sum_total_loss * args.batch_size) / (len(train_loader.dataset))))

    return (sum_total_loss * args.batch_size) / len(train_loader.dataset)


def calculate_metric_percase(pred, gt):
    # pdb.set_trace()
    if torch.is_tensor(pred):
        predict = pred.data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    pred = numpy.atleast_1d(predict.astype(numpy.bool))
    gt = numpy.atleast_1d(target.astype(numpy.bool))

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    pre = metric.binary.precision(pred, gt)
    rec = metric.binary.recall(pred, gt)
    spe = metric.binary.specificity(pred, gt)
    return dice, jc, pre, rec, spe


def test(test_loader, model, args):
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, mask_true = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            # mask_pred = (Out1>0.5).float()
            mask_pred = torch.sigmoid(output)
            mask_pred = torch.where(mask_pred > 0.5, 1., 0.)
            mask_true = mask_true.cpu()

            # transforms.ToPILImage()(mask_pred[0].squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(mask_true[0].squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(img[0]).show()  # Alternati
            loss = criterion_dice(output.cpu(), mask_true)
            sum_total_loss += loss.data.item()
            dice, jc, pre, rec, spe = calculate_metric_percase(mask_pred, mask_true)
            # mask_pred.argmax(dim=1)
            if epoch == 90:
                pdb.set_trace()
            dice, jc, pre, rec, spe = calculate_metric_percase(mask_pred, mask_true)
            pre_score += pre
            recall_score += rec
            dice_score += dice
            jaccard_score += jc
            spe_score += spe
    print("Test Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(test_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    # print(jaccard_score/len(test_loader))

    # print('Accuracy of the model on the validation images: %d %%' % (100 * correct / total))
    # sen, spe = (get2Svalues(targets, preds))
    # print(sen, spe)
    # print((sen + spe) / 2)
    # return
    return pre_score / len(test_loader), recall_score / len(test_loader), dice_score / len(
        test_loader), jaccard_score / len(test_loader), spe_score / len(test_loader), sum_total_loss / len(
        test_loader)
    # return pre_score / len(test_loader), recall_score / len(test_loader), dice_score / len(test_loader), jaccard_score / len(test_loader),  spe_score / len(test_loader)


# logger.info('finish training!')

def adjust_learning_rate(optimizer, epoch):  # 学习率自动调整
    if epoch % 60 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=6)
    parse.add_argument("--EPOCH", type=int, default=120)
    parse.add_argument("--LR", type=float, default=0.0001)
    parse.add_argument("--DEVICE", type=int, default=0)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")

    args = parse.parse_args()
    if args.DEVICE == 0:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练
    else:
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练

    total_img = os.listdir(imagefilepath)
    total_label = os.listdir(imagefilepath_label)

    skf = KFold(n_splits=4, shuffle=True)
    # pdb.set_trace()
    for i, (train_idx, val_idx) in enumerate(skf.split(total_img, total_label)):
        print("k_fold training : {} ".format(i))
        logging.info("k_fold training : {} ".format(i))

        trainset, valset = np.array(total_img)[[train_idx]], np.array(total_img)[[val_idx]]
        traintag, valtag = np.array(total_label)[[train_idx]], np.array(total_label)[[val_idx]]
        train_dataset = dataset.K_fold(trainset, traintag, imagefilepath, imagefilepath_label, transform,
                                       transform_test)
        val_dataset = dataset.K_fold(valset, valtag, imagefilepath, imagefilepath_label, transform, transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        criterion = nn.BCELoss()
        criterion_mse = nn.MSELoss()
        criterion_dice = DiceLoss()

        model = NestedUNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.LR)

        for epoch in range(1, args.EPOCH + 1):
            train_loss = train(train_loader, model, criterion, criterion_dice, epoch, args, DEVICE)
            pre, recall, dice, jaccard, spe, test_loss = test(test_loader, model, args)
            logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                        f'pre={pre:.4f}, recall={recall:.4f}, dice={dice:.4f}, jaccard={jaccard:.4f},spe={spe:.4f},test_loss={test_loss:.4f},')
        # for epoch in range(1, args.EPOCH + 1):
        #     train(epoch)  # 调用训练函数
        #     test(epoch)
        # # sen, spe = test()
        # # if epoch > 40:
        # #     # sen, spe = test()  # 调用测试函数
        # #     # if (sen + spe) / 2 > best:  # 断点存储模型方便下次训练
        # #     #     best = (sen + spe) / 2
        # #     checkpoint = {
        # #         "model_static_dict": model.state_dict(),
        # #         "epoch": epoch,
        # #         "optimizer_state_dic": optimizer.state_dict()
        # #     }
        # #     torch.save(checkpoint, 'first_train_%d.pth' % (epoch))
        # adjust_learning_rate(optimizer, epoch)  # 调用学习率自动调整函数
