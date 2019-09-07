#!/usr/bin/env python3
# coding: utf-8
import os
from collections import OrderedDict
import argparse

import torch

from models.imagenet.resnet import resnet50


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()

    for k, v in state_dict["state_dict"].items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # load it
    state_dict = torch.load(args.pth)
    model = resnet50()
    model.load_state_dict(fix_model_state_dict(state_dict))

    fname = os.path.splitext(os.path.basename(args.pth))[0] + "_fixed.pth"
    out_path = os.path.join(args.out, fname)

    torch.save(model.state_dict(), out_path)


if __name__ == "__main__":
    main()
