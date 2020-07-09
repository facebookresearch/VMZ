import os
from argparse import ArgumentParser

import torch

from .kinetics import Kinetics
from .ucf101 import UCF

from vmz.common import utils

__all__ = [
    "get_dataset",
    "dataset_asserts",
    "dataset_load_defaults",
    "get_dataset_arguments",
]


def get_dataset(args, transform, split="train"):
    metadata = None
    if split == "val" or split == "validataion":
        if args.val_file and os.path.isfile(args.val_file):
            metadata = torch.load(args.val_file)
        root = args.valdir
        train = False

    elif split == "train":
        if args.train_file and os.path.isfile(args.train_file):
            metadata = torch.load(args.train_file)
        root = args.traindir
        train = True

    if args.dataset == "kinetics400":
        _dataset = Kinetics(
            root, args.num_frames, transform=transform, _precomputed_metadata=metadata
        )
    elif args.dataset == "ucf101":
        _dataset = UCF(
            root,
            args.annotation_path,
            frames_per_clip=args.num_frames,
            train=train,
            transform=transform,
            fold=args.fold,
            _precomputed_metadata=metadata,
        )

    _dataset.video_clips.compute_clips(args.num_frames, 1)
    if args.train_file is None or not os.path.isfile(args.train_file):
        utils.save_on_master(
            _dataset.metadata,
            "{}_{}_{}fms.pth".format(args.dataset, split, args.num_frames),
        )
    return _dataset


def dataset_asserts(args):
    # safety checks
    # assert args.val_file is not None and args.train_file is not None

    if args.dataset == "kinetics400":
        assert args.traindir != args.valdir

    if args.dataset in ["ucf101", "hmdb51"]:
        assert args.annotation_path is not None
        assert args.fold is not None


def dataset_load_defaults(args):
    if args.dataset == "kinetics400":
        args.traindir = "/datasets01_101/kinetics/070618/train_avi-480p"
        args.valdir = "/datasets01_101/kinetics/070618/val_avi-480p"
        args.train_file = "/checkpoint/bkorbar/DATASET_TV/kinetics_val32frms_01_101.pth"
        args.val_file = "/checkpoint/bkorbar/DATASET_TV/kinetics_val32frms_01_101.pth"
    if args.dataset == "ucf101":
        args.traindir = "/private/home/bkorbar/data/video/ucf101/data"
        args.valdir = "/private/home/bkorbar/data/video/ucf101/data"
        args.val_file = "/checkpoint/bkorbar/DATASET_TV/ucf101_train_16fms.pth"
        args.train_file = "/checkpoint/bkorbar/DATASET_TV/ucf101_train_16fms.pth"
        args.annotation_path = (
            "/private/home/bkorbar/data/video/ucf101/orig_annotations/"
        )

    return args


def get_dataset_arguments(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument("--dataset", default="kinetics400", type=str)
    parser.add_argument("--num_classes", default=400, type=int)

    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--train_bs_multiplier", default=5, type=int)
    parser.add_argument("--val_clips_per_video", default=1, type=int)

    parser.add_argument("--traindir", default="", type=str)
    parser.add_argument("--valdir", default="", type=str)
    parser.add_argument("--train_file", default="", type=str)
    parser.add_argument("--val_file", default="", type=str)

    parser.add_argument("--annotation_path", default="", type=str)
    parser.add_argument("--fold", default=1, type=int)

    # Transform parameters
    parser.add_argument(
        "--scale_h",
        default=128,
        type=int,
        metavar="N",
        help="number of frames per clip",
    )
    parser.add_argument(
        "--scale_w",
        default=174,
        type=int,
        metavar="N",
        help="number of frames per clip",
    )
    parser.add_argument(
        "--crop_size",
        default=112,
        type=int,
        metavar="N",
        help="number of frames per clip",
    )

    return parser
