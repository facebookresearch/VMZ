from argparse import ArgumentParser
from vmz.datasets import get_dataset_arguments, dataset_load_defaults, dataset_asserts

__all__ = ["parse_args"]


def parse_args():
    parser = ArgumentParser(description="PyTorch Video Classification Training")
    parser.add_argument("--name", default="video_classification_workflow")
    parser.add_argument("--model", default="r2plus1d_18", help="model")
    parser.add_argument("--device", default="cuda", help="device")

    parser.add_argument("-b", "--batch-size", default=24, type=int)
    parser.add_argument(
        "--epochs",
        default=45,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 10)",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument(
        "--fc_lr", default=0.1, type=float, help="fully connected learning rate"
    )
    parser.add_argument(
        "--l1_lr", default=0.001, type=float, help="first_block learning rate"
    )
    parser.add_argument(
        "--l2_lr", default=0.001, type=float, help="second_block learning rate"
    )
    parser.add_argument(
        "--l3_lr", default=0.001, type=float, help="third_block learning rate"
    )
    parser.add_argument(
        "--l4_lr", default=0.001, type=float, help="last_block learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        default=[20, 30, 40],
        type=int,
        help="decrease lr on milestones",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-warmup-epochs", default=10, type=int, help="number of warmup epochs"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--num_finetune_classes", default=101, type=int)
    parser.add_argument("--output-dir", default=".", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--resume_from_model", default="", help="resume from pretrained model"
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    parser.add_argument(
        "--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true",
    )
    parser.add_argument(
        "--eval_only", help="Only validate the model", action="store_true",
    )
    # parser.add_argument(
    #     "--test_on_cluster",
    #     dest="is_online",
    #     help="Should we run it locally or not",
    #     action="store_true",
    # )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default="",
        help="Use pre-trained models from the modelzoo",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--apex", action="store_true", help="Use apex for mixed precision training"
    )

    parser.add_argument(
        "--apex-opt-level",
        default="O1",
        type=str,
        help="For apex mixed precision training"
        "O0 for FP32 training, O1 for mixed precision training."
        "For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    # slurm args
    parser.add_argument("--partition", default="dev", help="Slurm partition to use?")
    parser.add_argument("--nodes", default=2, type=int, help="number nodes tu use")

    # get dataset arguments
    parser = get_dataset_arguments(parser)

    # parse the args
    args = parser.parse_args()

    # TODO: remove this before publishing
    args = dataset_load_defaults(args)

    # safety checks
    dataset_asserts(args)

    return args
