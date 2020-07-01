import os

# TODO: remove unused imports
import torch
import torch.utils.data
from torch import nn

import torchvision
import torchvision.datasets.video_utils

from vmz.common import log, utils, transforms as T
from vmz.common.sampler import UniformClipSampler
from vmz.datasets import get_dataset
import vmz.models as models


__all__ = ["aggredate_video_accuracy", "test", "test_main"]


def aggredate_video_accuracy(softmaxes, labels, topk=(1,), aggregate="mean"):
    # TODO: docs and comments
    torch.save(labels, "/private/home/bkorbar/torch_projects/VMZ/pt/LABELs.pth")
    torch.save(softmaxes, "/private/home/bkorbar/torch_projects/VMZ/pt/SM.pth")

    maxk = max(topk)
    output_batch = torch.stack(
        [
            torch.mean(torch.stack(softmaxes[sms]), 0, keepdim=False)
            for sms in softmaxes.keys()
        ]
    )
    num_videos = output_batch.size(0)
    output_labels = torch.stack([labels[video_id] for video_id in softmaxes.keys()])

    _, pred = output_batch.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(output_labels.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / num_videos))
    return res


def test(model, criterion, data_loader, device, print_freq, metric_logger):
    # TODO: docs and comments
    softmaxes = {}
    labels = {}

    model.eval()
    header = "Test:"
    i = 1
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, print_freq, header):
            i += 1
            video, target, video_idx, _ = data
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            for j in range(len(video_idx)):
                video_id = video_idx[j].item()
                # note, this is not SM but FC layer actually
                sm = output[j].cpu()
                label = target[j].cpu()

                # append it to video dict
                softmaxes.setdefault(video_id, []).append(sm)
                labels[video_id] = label
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    video_acc1, video_acc5 = aggredate_video_accuracy(softmaxes, labels, topk=(1, 5))

    print(
        " *** TESTING SUMMARY:"
        "Clip Acc@1 {top1.global_avg:.3f}"
        "Clip Acc@5 {top5.global_avg:.3f}\n"
        "*** Video Acc@1 {v_acc1:.3f} Video Acc@5 {v_acc5:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            v_acc1=video_acc1.item(),
            v_acc5=video_acc5.item(),
        )
    )


def test_main(args):
    torchvision.set_video_backend("video_reader")
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
            T.Resize((args.scale_h, args.scale_w)),
            T.NormalizeVideo(
                mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
            ),
            T.CenterCropVideo((args.crop_size, args.crop_size)),
        ]
    )

    print("Loading validation data")
    if os.path.isfile(args.val_file):
        metadata = torch.load(args.val_file)
        root = args.valdir

    # TODO: add test option fro datasets that support that
    dataset_test = get_dataset(args, transform_test, "val")

    dataset_test.video_clips.compute_clips(args.num_frames, 1, frame_rate=15)

    test_sampler = UniformClipSampler(
        dataset_test.video_clips, args.val_clips_per_video
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
    )

    print("Creating model")
    # TODO: model only from our models
    available_models = {**models.__dict__}

    model = available_models[args.model](pretraining=args.pretrained)
    model.to(device)
    model_without_ddp = model

    model = torch.nn.parallel.DataParallel(model)
    model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss()

    # model pretrained or this
    if not args.pretrained:
        print(f"Loading the model from {args.resume_from_model}")
        checkpoint = torch.load(args.resume_from_model, map_location="cpu")
        if "model" in checkpoint.keys():
            model_without_ddp.load_state_dict(checkpoint["model"])
        else:
            model_without_ddp.load_state_dict(checkpoint)

    print("Starting test_only")
    metric_logger = log.MetricLogger(delimiter="  ", writer=None, stat_set="val")
    test(model, criterion, data_loader_test, device, 2, metric_logger)


if __name__ == "__main__":
    from vmz.func.opts import parse_args

    args = parse_args()
    test_main(args)
    exit()
