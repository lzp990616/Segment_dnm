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

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--log_name", type=str, default="./log/test.log")
parse.add_argument("--batch_size", type=int, default=6)
parse.add_argument("--EPOCH", type=int, default=100)
parse.add_argument("--LR", type=float, default=0.0001)
parse.add_argument("--DEVICE", type=int, default=0)
parse.add_argument("--M", type=int, default=10)
parse.add_argument("--DNM1", type=int, default=1)
parse.add_argument("--DNM2", type=int, default=1)
parse.add_argument("--ckpt", type=str, help="the path of model weight file")
args = parse.parse_args()
if args.DEVICE == 0:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练
else:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.RandomResizedCrop(224),#Resizes all images into same dimension
    # transforms.RandomRoation(10),# Rotates the images upto Max of 10 Degrees
    # transforms.RandomHorizontalFlip(p=0.4),#Performs Horizantal Flip over images
    # transforms.RandomVerticalFlip(p=0.4),
    # transforms.RandomRotation(15),
    # transforms.RandomRotation(90, expand=True),
    # transforms.RandomHorizontalFlip(p=1.0),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Coverts into Tensors
    transforms.Normalize(mean=mean_nums, std=std_nums)  # Normalizes
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
filepath_busi_m = './data/Dataset_BUSI_malignant/Dataset_BUSI_with_GT/'
filepath_cloth = './data/archive/'

filepath = filepath_bus

imagefilepath = filepath + 'data/train/'
imagefilepath_label = filepath + 'data/trainannot/'

valfilepath = filepath + 'data/val/'
valfilepath_label = filepath + 'data/valannot/'

train_dataset = dataset.Busi(imagefilepath, imagefilepath_label, transform, transform_test)
test_dataset = dataset.Busi(valfilepath, valfilepath_label, transform, transform_test)
# print(train_dataset.class_to_idx)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


# train_size = len(train_loader.dataset)
# test_size = len(test_loader.dataset) #incorrect
# train_num_batches = len(train_loader)
# test_num_batches = len(test_loader)


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


# 损失函数和模型调用
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_dice = DiceLoss()

model = model_net.RRCNet(DEVICE, args.M, args.DNM1, args.DNM2)
model.to(DEVICE)

# 预训练模型和优化器的选用：
# pretrained_model = "./log_beifen_weight/bus_2dnm_adam1e-4_epoch160_batchSize8.log.pth"
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)
optimizer = optim.Adam(model.parameters(), lr=args.LR)

# 预训练模型加载
pretrained = 0
if pretrained:
    pretrain_model = model_net.RRCNet(DEVICE)
    pre_dic = torch.load(pretrained_model)
    pretrain_model.load_state_dict(pre_dic["model_static_dict"])
    model_dict = model.state_dict()
    pretrained_dict = pretrain_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 选择相同的部分
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.CRITICAL}
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


# logger = get_logger('./log/log_M10_2dnm_busi.log')
# log_name
logger = get_logger(args.log_name)
# logger = get_logger('./log/bus_RRCNet_2dnm_4_adam 1e-4.log')
logger.info('start training!')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)


# 训练
def train(epoch):
    model.train()  # 模型的训练模式
    sum_total_loss_batch = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(8)]
    pdb.set_trace()
    for batch_idx, (img, label) in tqdm(enumerate(train_loader)):  # 从dataloader中按照dataset.py的顺序进行数据调用，tqdm为循环时间的可视化
        loss = [0 for i in range(8)]
        total_loss = 0
        img, label = img.to(DEVICE), label.to(DEVICE)  # 使用gpu
        model.zero_grad()  # 清梯度
        output = model(img)  # 把数据img 传入模型 获得结果output
        loss[0] = criterion_mse(output[0], label)
        loss[1] = criterion_mse(output[1], label)

        for i in range(2, 8):
            loss[i] = 0.5 * criterion_dice(output[i], label)
        for i in range(8):
            loss_sum[i] += loss[i].data.item()
        for i in range(8):
            total_loss += loss[i]

        # total_loss = loss[0] + loss[1] + loss[2] + 0.5 * loss[3] + 0.5 * loss[4] + 0.5 * loss[5] + 0.5 * loss[6] + 0.5 * loss[7]

        sum_total_loss += total_loss.data.item()
        sum_total_loss_batch += total_loss.data.item()
        total_loss.backward()  # 反向传播当前梯度
        optimizer.step()  # 进行单次优化更新参数
        if batch_idx % 10 == 0 and batch_idx != 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss0: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}\tLoss4: {:.6f}\tLoss5: {:.6f}\tLoss6: {:.6f}\tLoss7: {:.6f}\tLR: {}'.format(
                    epoch, batch_idx * len(img), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), sum_total_loss_batch / 10, loss_sum[0] / 10,
                           loss_sum[1] / 10, loss_sum[2] / 10, loss_sum[3] / 10, loss_sum[4] / 10, loss_sum[5] / 10,
                           loss_sum[6] / 10, loss_sum[7] / 10,
                    optimizer.param_groups[0]['lr']))
            # logger.info(
            #     'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss0: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}\tLoss4: {:.6f}\tLoss5: {:.6f}\tLoss6: {:.6f}\tLoss7: {:.6f}\tLR: {}'.format(
            #         epoch, batch_idx * len(img), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), sum_total_loss_batch / 10, loss_sum[0] / 10,
            #                loss_sum[1] / 10, loss_sum[2] / 10, loss_sum[3] / 10, loss_sum[4] / 10, loss_sum[5] / 10,
            #                loss_sum[6] / 10, loss_sum[7] / 10,
            #         optimizer.param_groups[0]['lr']))
            sum_total_loss_batch = 0
            loss_sum = [0 for i in range(8)]
    print(
        'Epoch: {}\tTrain_loss: {:.6f}'.format(epoch, (sum_total_loss * args.batch_size) / (len(train_loader.dataset))))
    # logger.info(
    #     'Epoch: {}\ttrain_loss: {:.6f}'.format(epoch, (sum_total_loss * args.batch_size) / (len(train_loader.dataset))))
    return (sum_total_loss * args.batch_size) / (len(train_loader.dataset))


def recall_fun(predict, target):  # Sensitivity, Recall, true positive rate都一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    fn = numpy.count_nonzero(~predict & target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def precision_fun(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    fp = numpy.count_nonzero(predict & ~target)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def specificity_fun(predict, target):  # Specificity，true negative rate一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tn = numpy.count_nonzero(~predict & ~target)
    fp = numpy.count_nonzero(predict & ~target)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity


def dice_fun(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))  # 转一维数组
    target = numpy.atleast_1d(target.astype(numpy.bool))

    intersection = numpy.count_nonzero(predict & target)  # 计算非零个数

    size_i1 = numpy.count_nonzero(predict)
    size_i2 = numpy.count_nonzero(target)

    try:
        dice = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dice = 0.0

    return dice


def jadcard_fun(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    intersection = numpy.count_nonzero(predict & target)
    union = numpy.count_nonzero(predict | target)

    jac = float(intersection) / float(union)

    return jac


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


def test(epoch):
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(8)]
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, label = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            # mask_pred = (Out1>0.5).float()
            mask_pred = torch.where(output[1] > 0.5, 1., 0.)
            # pdb.set_trace()
            # 可视化
            # plt.imshow(transforms.ToPILImage()(mask_pred.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(output[0].squeeze()).show()  # Alternatively
            # plt.imshow(transforms.ToPILImage()(mask_true.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_true.squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(Out0[0].squeeze()).show()
            mask_pred = mask_pred.float().cpu()
            # mask_pred.argmax(dim=1)
            mask_true = mask_true.to(device=DEVICE, dtype=torch.long).float().cpu()
            # loss = criterion(mask_true, mask_pred)

            # loss[0] = criterion_mse(output[0], label)
            # loss[1] = criterion_mse(output[1], label)
            # for i in range(2, 8):
            #   loss[i] = criterion(output[i], label)
            # for i in range(8):
            #    loss_sum[i] += loss[i].data.item()

            if epoch == 99:
                pdb.set_trace()
            loss = criterion_mse(mask_true, mask_pred)
            sum_total_loss += loss.data.item()
            # precision_fun(mask_pred, mask_true)
            # specificity_fun(mask_pred, mask_true)
            # recall_fun(mask_pred, mask_true)
            # dice_fun(mask_pred, mask_true)
            # jadcard_fun(mask_pred, mask_true)
            # pre, recall = precision_recall(mask_pred, mask_true.int(), average='micro')
            # specificity = Specificity(average='micro')
            # spe_score += specificity(mask_pred, mask_true.int())
            # pdb.set_trace()
            dice, jc, pre, rec, spe = calculate_metric_percase(mask_pred, mask_true)
            pre_score += pre
            recall_score += rec
            dice_score += dice
            jaccard_score += jc
            spe_score += spe
            # pdb.set_trace()
            # dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
    print("Test Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(test_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    print("test_loss: \t{:.4f}".format(sum_total_loss / len(test_loader)))
    # logger.info(
    #     "Epoch: {}".format(epoch))
    # logger.info("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    # logger.info("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    # logger.info("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    # logger.info("jaccard_score: {:.4f}".format(jaccard_score / len(test_loader)))
    # logger.info("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    return pre_score / len(test_loader), recall_score / len(test_loader), dice_score / len(
        test_loader), jaccard_score / len(test_loader), spe_score / len(test_loader), sum_total_loss / len(test_loader)
    # print(jaccard_score/len(test_loader))

    # print('Accuracy of the model on the validation images: %d %%' % (100 * correct / total))
    # sen, spe = (get2Svalues(targets, preds))
    # print(sen, spe)
    # print((sen + spe) / 2)
    # return


# logger.info('finish training!')

def adjust_learning_rate(optimizer, epoch):  # 学习率自动调整
    if epoch % 80 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


# best = 0.9
if __name__ == '__main__':
    min = 0.011
    for epoch in range(1, args.EPOCH + 1):
        train_loss = train(epoch)  # 调用训练函数
        pre, recall, dice, jaccard, spe, test_loss = test(epoch)
        logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                    f'pre={pre:.4f}, recall={recall:.4f}, dice={dice:.4f}, jaccard={jaccard:.4f},spe={spe:.4f},test_loss={test_loss:.4f},')
        if min > test_loss:
            min = test_loss
            checkpoint = {
                "model_static_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dic": optimizer.state_dict()
            }
            torch.save(checkpoint, args.log_name + '.pth')
    # sen, spe = test()
    # if epoch > 40:
    #     # sen, spe = test()  # 调用测试函数
    #     # if (sen + spe) / 2 > best:  # 断点存储模型方便下次训练
    #     #     best = (sen + spe) / 2

    # adjust_learning_rate(optimizer, epoch)  # 调用学习率自动调整函数
