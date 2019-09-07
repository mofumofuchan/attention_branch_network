import time

import torch
import torch.nn as nn

from utils import Bar, AverageMeter, accuracy

from models.imagenet.resnet import resnet50


class ResNet50WithNewAttentionBranch(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super(ResNet50WithNewAttentionBranch, self).__init__()
        self.original_model = original_model
        self.new_model = resnet50(num_classes=num_classes)

    def forward(self, x):
        x = self.original_model.conv1(x)
        x = self.original_model.bn1(x)
        x = self.original_model.relu(x)
        x = self.original_model.maxpool(x)

        x = self.original_model.layer1(x)
        x = self.original_model.layer2(x)
        x = self.original_model.layer3(x)

        fe = x

        ax = self.new_model.bn_att(self.new_model.att_layer4(x))
        ax = self.new_model.relu(self.new_model.bn_att2(
            self.new_model.att_conv(ax)))
        bs, cs, ys, xs = ax.shape
        self.new_model.att = self.new_model.sigmoid(
            self.new_model.bn_att3(self.new_model.att_conv3(ax)))
        # self.new_model.att = self.new_model.att.view(bs, 1, ys, xs)
        ax = self.new_model.att_conv2(ax)
        ax = self.new_model.att_gap(ax)
        ax = ax.view(ax.size(0), -1)

        rx = x * self.new_model.att
        rx = rx + x
        per = rx
        rx = self.new_model.layer4(rx)
        rx = self.new_model.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.new_model.fc(rx)

        return ax, rx, [self.new_model.att, fe, per]


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs),\
                          torch.autograd.Variable(targets)

        # compute output
        att_outputs, outputs, _ = model(inputs)
        att_loss = criterion(att_outputs, targets)
        per_loss = criterion(outputs, targets)
        loss = att_loss + per_loss

        # measure accuracy and record loss
        prec1, = accuracy(outputs.data, targets.data, topk=(1,))
        #losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.data, inputs.size(0))
        #top1.update(prec1[0], inputs.size(0))
        top1.update(prec1, inputs.size(0))
        #top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)
