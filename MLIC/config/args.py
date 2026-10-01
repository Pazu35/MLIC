import argparse


def train_options():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        default="mlicplus0483mse",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/home/npr/dataset/",
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=500,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.045,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--loss-learning-rate",
        default=1e-2,
        help="Loss loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        default=True,
        help="Use cuda"
    )
    parser.add_argument(
        "--save",
        action = "store_true",
        default=False,
        help="Save model to disk"
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=192.1,
        help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="With --checkpoint: continue that run instead of fine-tuning from it. "
             "Restores the optimizers, the LR scheduler, the epoch counter and the "
             "loss weights; --epochs stays the TOTAL epoch count (utils/resume.py)"
    )
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes'
    )
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training'
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of gradient accumulation steps (default: %(default)s)"
    )
    # Remedies for an auxiliary (structure) loss whose gradient opposes the
    # reconstruction's -- report section 14.7. Both off by default.
    parser.add_argument(
        "--grad_projection",
        action="store_true",
        default=False,
        help="PCGrad-style: project the structure-term gradient off the value-term "
             "gradient whenever they conflict (utils/grad_surgery.py)"
    )
    parser.add_argument(
        "--grad_weighting",
        action="store_true",
        default=False,
        help="Weight the structure terms by gradient norm (a fraction of recon's) "
             "instead of by loss value; needs use_factor_weights"
    )
    parser.add_argument(
        "--grad_weight_ratio",
        default=0.2,
        type=float,
        help="Target |grad term| / |grad recon| for --grad_weighting (default: %(default)s)"
    )
    parser.add_argument(
        "--grad_weight_every",
        default=0,
        type=int,
        help="Re-calibrate --grad_weighting every N epochs after warm-up; 0 = once "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "--grad_weight_batches",
        default=4,
        type=int,
        help="Training batches used per --grad_weighting calibration (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose for training"
    )

    args = parser.parse_args()
    return args


def test_options():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        default="swlic_mse_0932_testv3",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/home/npr/dataset/",
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=3,
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        default=True,
        type=bool,
        help="Use cuda"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save model to disk"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose for training"
    )

    args = parser.parse_args()
    return args
