import datetime
import os
import time
import sys

import torch
import torch.nn as nn
import torchvision

from vmz.common import utils, transforms as T
from vmz.common.log import MetricLogger, setup_tbx, get_default_loggers
from vmz.common.sampler import DistributedSampler, UniformClipSampler, RandomClipSampler
from vmz.common.scheduler import WarmupMultiStepLR
from vmz.datasets import get_dataset
import vmz.models as models

try:
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    data_loader,
    device,
    epoch,
    print_freq,
    metric_logger,
    apex=False,
):
    model.train()

    header = "Epoch: [{}]".format(epoch)
    for data in metric_logger.log_every(data_loader, print_freq, header):
        video, target, _, _ = data
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        output = model(video)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()


def evaluate(model, criterion, data_loader, device, metric_logger):
    # TODO: docs and comments
    model.eval()
    header = "Test:"
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 2, header):
            video, target, _, _ = data
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        " *Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )
    return metric_logger.acc1.global_avg


def train_main(args):
    torchvision.set_video_backend("video_reader")
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError(
                "Failed to import apex. Please install apex "
                "from https://www.github.com/nvidia/apex "
                "to enable mixed-precision training."
            )

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    writer = setup_tbx(args.output_dir)

    # Data loading code
    print("Loading data")

    print("\t Loading datasets")
    st = time.time()

    if not args.eval_only:
        print("\t Loading train data")
        transform_train = torchvision.transforms.Compose(
            [
                T.ToTensorVideo(),
                T.Resize((args.scale_h, args.scale_w)),
                T.RandomHorizontalFlipVideo(),
                T.NormalizeVideo(
                    mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
                ),
                T.RandomCropVideo((args.crop_size, args.crop_size)),
            ]
        )
        dataset = get_dataset(args, transform_train)
        dataset.video_clips.compute_clips(args.num_frames, 1, frame_rate=15)
        train_sampler = RandomClipSampler(dataset.video_clips, args.train_bs_multiplier)
        if args.distributed:
            train_sampler = DistributedSampler(train_sampler)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
        )

    print("\t Loading validation data")
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
    dataset_test = get_dataset(args, transform_test, split="val")
    dataset_test.video_clips.compute_clips(args.num_frames, 1, frame_rate=15)
    test_sampler = UniformClipSampler(
        dataset_test.video_clips, args.val_clips_per_video
    )
    if args.distributed:
        test_sampler = DistributedSampler(test_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
    )

    criterion = nn.CrossEntropyLoss()

    print("Creating model")
    # TODO: model only from our models
    available_models = {**models.__dict__}
    model = available_models[args.model](pretraining=args.pretrained)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume_from_model and not args.resume:
        checkpoint = torch.load(args.resume_from_model, map_location="cpu")
        if "model" in checkpoint.keys():
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    if args.finetune:
        assert args.resume_from_model is not None or args.pretrained
        model.fc = nn.Linear(model.fc.in_features, args.num_finetune_classes)

    lr = args.lr * args.world_size
    if args.finetune:
        params = [
            {"params": model.stem.parameters(), "lr": 0},
            {"params": model.layer1.parameters(), "lr": args.l1_lr * args.world_size},
            {"params": model.layer2.parameters(), "lr": args.l2_lr * args.world_size},
            {"params": model.layer3.parameters(), "lr": args.l3_lr * args.world_size},
            {"params": model.layer4.parameters(), "lr": args.l4_lr * args.world_size},
            {"params": model.fc.parameters(), "lr": args.fc_lr * args.world_size},
        ]
    else:
        params = model.parameters()

    print(params)

    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )

    if args.apex:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.apex_opt_level
        )

    # convert scheduler to be per iteration,
    # not per epoch, for warmup that lasts
    # between different epochs
    if not args.eval_only:
        warmup_iters = args.lr_warmup_epochs * len(data_loader)
        lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=lr_milestones,
            gamma=args.lr_gamma,
            warmup_iters=warmup_iters,
            warmup_factor=1e-5,
        )

    if os.path.isfile(os.path.join(args.output_dir, "checkpoint.pth")):
        args.resume = os.path.join(args.output_dir, "checkpoint.pth")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.eval_only:
        print("Starting test_only")
        metric_logger = MetricLogger(delimiter="  ", writer=writer, stat_set="val")
        evaluate(model, criterion, data_loader_test, device, metric_logger)
        return

    # Get training metric logger
    stat_loggers = get_default_loggers(writer, args.start_epoch)

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            args.print_freq,
            stat_loggers["train"],
            args.apex,
        )
        evaluate(model, criterion, data_loader_test, device, stat_loggers["val"])
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "model_{}.pth".format(epoch))
            )
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

        # reset all meters in the metric logger
        for log in stat_loggers:
            stat_loggers[log].reset_meters()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    from vmz.func.opts import parse_args
    import torchvision

    torchvision.set_video_backend("video_reader")
    args = parse_args()
    train_main(args)
    exit()
