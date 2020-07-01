import os

# TODO: remove unused imports
import torch
import torch.utils.data
from torch import nn

from tqdm import tqdm

import torchvision
import torchvision.datasets.video_utils

from vmz.common import log, utils, transforms as T
from vmz.common.sampler import UniformClipSampler
from vmz.datasets import get_dataset
import vmz.models as models


def extract_feats(model, data_loader, dataset, device, args):
    # TODO: docs and comments
    feats = {}

    model.eval()
    header = "Test:"
    with torch.no_grad():
        t = tqdm(iter(data_loader), leave=False, total=len(data_loader))
        for i, data in enumerate(t):
            video, target, video_idx, clip_idx = data
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)

            for j in range(len(video_idx)):
                vid = video_idx[j].item()
                video_id = dataset.video_clips.video_paths[vid]
                if video_id not in list(feats.keys()):
                    feats[video_id] = {
                        "feature": [],
                        "label": [],
                        "clip_id": [],
                        "pts": [],
                    }
                clip_id = clip_idx[j].item()

                # note, this is not SM but FC layer actually
                sm = output[j]
                label = target[j]
                feats[video_id]["feature"].append(sm.cpu())
                feats[video_id]["label"].append(label.cpu())
                feats[video_id]["clip_id"].append(clip_id)
                feats[video_id]["pts"].append(
                    dataset.video_clips.video_pts[vid][clip_id]
                )

    # gather the stats from all processes
    sp = os.path.join(args.output_dir, "features.pth")
    torch.save(feats, sp)


def ef_main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    transform_test = torchvision.transforms.Compose(
        [
            T.ToTensorVideo(),
            T.Resize((256, 324)),
            T.NormalizeVideo(
                mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
            ),
            T.CenterCropVideo(224),
        ]
    )

    print("Loading validation data")
    if os.path.isfile(args.val_file):
        metadata = torch.load(args.val_file)
        root = args.valdir

    # TODO: add test option fro datasets that support that
    dataset_test = get_dataset(args, transform_test, "val")

    print("by default we're extracting all clips at given fps with 50percent overlap")
    dataset_test.video_clips.compute_clips(
        args.num_frames, args.num_frames // 2, frame_rate=15
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.workers,
    )

    print("Creating model")
    # TODO: model only from our models
    available_models = {**models.__dict__}

    model = available_models[args.model](pretraining=args.pretrained)
    model.to(device)
    model_without_ddp = model

    model = torch.nn.parallel.DataParallel(model)
    model_without_ddp = model.module

    # model pretrained or this
    print(f"Loading the model from {args.resume_from_model}")
    checkpoint = torch.load(args.resume_from_model, map_location="cpu")
    if "model" in checkpoint.keys():
        model_without_ddp.load_state_dict(checkpoint["model"])
    else:
        model_without_ddp.load_state_dict(checkpoint)

    print("Starting feature extraction")
    extract_feats(model, data_loader_test, dataset_test, device, args)


if __name__ == "__main__":
    from vmz.func.opts import parse_args

    args = parse_args()
    ef_main(args)
    exit()
