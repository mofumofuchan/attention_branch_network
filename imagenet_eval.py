import time
import os
from os import path

import cv2

import torch
from torch import nn

from utils import AverageMeter, Bar, accuracy


def test(val_loader, model, criterion, epoch, use_cuda):
    softmax = nn.Softmax()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # fc = open('categorie.txt', 'r')
    # cate_name = []
    # for line in fc:
    #     idx_item, name_item = line[:-2].split(':')
    #     sp_name = name_item.split(',')
    #     name_item = sp_name[0]
    #     cate_name.append(name_item)
    # cate_name = np.asarray(cate_name)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    count = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), \
                            torch.autograd.Variable(targets)

        # compute output
        _, outputs, attention = model(inputs)
        outputs = softmax(outputs)
        loss = criterion(outputs, targets)
        attention, fe, per = attention

        c_att = attention.data.cpu()
        c_att = c_att.numpy()
        d_inputs = inputs.data.cpu()
        d_inputs = d_inputs.numpy()

        in_b, in_c, in_y, in_x = inputs.shape
        for item_img, item_att in zip(d_inputs, c_att):

            v_img = ((item_img.transpose((1, 2, 0)) +
                      0.5 +
                      [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225]) * 256
            v_img = v_img[:, :, ::-1]
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            resize_att *= 255.

            cv2.imwrite('stock1.png', v_img)
            cv2.imwrite('stock2.png', resize_att)
            v_img = cv2.imread('stock1.png')
            vis_map = cv2.imread('stock2.png', 0)
            jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
            jet_map = cv2.add(v_img, jet_map)

            out_dir = path.join('output')
            if not path.exists(out_dir):
                os.mkdir(out_dir)
            out_path = path.join(out_dir,
                                 'attention', '{0:06d}.png'.format(count))
            cv2.imwrite(out_path, jet_map)
            out_path = path.join(out_dir,
                                 'raw', '{0:06d}.png'.format(count))
            cv2.imwrite(out_path, v_img)

            count += 1

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        prec1, = accuracy(outputs.data, targets.data, topk=(1,))

        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | ' + \
                     'Batch: {bt:.3f}s | Total: {total:} | ' + \
                     'ETA: {eta:} | Loss: {loss:.4f} | ' + \
                     'top1: {top1: .4f} | top5: {top5: .4f}'.format(
                         batch=batch_idx + 1,
                         size=len(val_loader),
                         data=data_time.avg,
                         bt=batch_time.avg,
                         total=bar.elapsed_td,
                         eta=bar.eta_td,
                         loss=losses.avg,
                         top1=top1.avg,
                         top5=top5.avg,
                     )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)
