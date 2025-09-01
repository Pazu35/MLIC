import os 
import sys

running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)

import os
import random
import logging
from PIL import ImageFile, Image
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from MLIC.MLIC.utils.logger import setup_logger
from MLIC.MLIC.utils.utils import CustomDataParallel, save_checkpoint
from MLIC.MLIC.utils.optimizers import configure_optimizers
from MLIC.MLIC.utils.training import train_one_epoch
from MLIC.MLIC.utils.testing import test_one_epoch
from MLIC.MLIC.loss.rd_loss import RateDistortionLoss
from MLIC.MLIC.config.args import train_options
from MLIC.MLIC.config.config import model_config
from MLIC.MLIC.models import *
import random
import pickle
import xarray as xr
from datetime import datetime
from FASCINATION.src.autoencoder_datamodule import AutoEncoderDatamodule_3D



def main(dm,config):
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    #config = model_config()

    # Create timestamp for unique checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir_name = f"{args.experiment}_{config['N']}_{config['M']}_lambda{args.lmbda}/{timestamp}"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
    #     seed = args.seed
    # else:
        seed = 100 * random.random()
    torch.manual_seed(seed)
    random.seed(seed)

    if not os.path.exists(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name)):
        os.makedirs(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name))

    # Ensure the logger directory path matches the absolute path
    logger_dir = os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name)
    setup_logger('train', logger_dir, 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', logger_dir, 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='/Odyssey/private/o23gauvr/code/MLIC/tb_logger/' + checkpoint_dir_name)

    if not os.path.exists(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name, 'checkpoints')):
        os.makedirs(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name, 'checkpoints'))

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    if dm==None:
            
        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
        test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )
    
    else:
        train_dataloader = dm["train"]
        test_dataloader = dm["test"]

    net = MLICPlusPlus(config=config)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)

    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        # new_ckpt = modify_checkpoint(checkpoint['state_dict'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1)
        # lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        # lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        # print(lr_scheduler.state_dict())
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        checkpoint = None
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    # start_epoch = 0
    # best_loss = 1e10
    # current_step = 0

    logger_train.info(args)
    logger_train.info(config)
    logger_train.info(net)
    logger_train.info(optimizer)
    optimizer.param_groups[0]['lr'] = args.learning_rate
    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step
        )

        save_dir = os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name, 'val_images', '%03d' % (epoch + 1))
        loss = test_one_epoch(epoch, test_dataloader, net, criterion, save_dir, logger_val, tb_logger)

        lr_scheduler.step()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        net.update(force=True)
        if args.save and is_best:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == '__main__':  
    import argparse
    
    # Parse custom arguments for configuration
    parser = argparse.ArgumentParser(description='Train MLIC++ model')
    parser.add_argument('--cfg_N', type=int, default=192, help='Model config N parameter')
    parser.add_argument('--cfg_M', type=int, default=320, help='Model config M parameter')
    parser.add_argument('--lambda_val', type=float, default=0.0018, help='Lambda value for rate-distortion loss')
    parser.add_argument('--load_dm', action='store_true', help='Load datamodule from pickle file')
    parser.add_argument('--dm_path', type=str, default="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/dm_enatl_mean_std_along_depth_4_157_196_256.pkl", help='Path to datamodule pickle file')
    parser.add_argument('--epochs', type=int, default=34000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU ID')
    
    custom_args, remaining_args = parser.parse_known_args()

    sys.argv = [
        "train.py",
        "--metrics", "mse",
        "--exp", "mlicpp_on_ssp",
        "--gpu_id", custom_args.gpu_id,
        "--epochs", str(custom_args.epochs),
        "--lambda", str(custom_args.lambda_val),
        "-lr", str(custom_args.lr),
        "--num-workers", "16",
        "--clip_max_norm", "1.0",
        "--seed", "42",
        "--batch-size", str(custom_args.batch_size),
        "--patch-size", "196", "256"
    ]

    cfg = model_config()
    cfg["N"] = custom_args.cfg_N
    cfg["M"] = custom_args.cfg_M
    # cfg["slice_num"] = 10
    # cfg["context_window"] = 5

    load_datamodule = custom_args.load_dm
    dm_path = custom_args.dm_path


    if load_datamodule:
        with open(dm_path, 'rb') as f:
            datamodule = pickle.load(f)
            dm={"train": datamodule.train_dataloader(), "test": datamodule.test_dataloader()}

    else:

        data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                    "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}


        datamodule = AutoEncoderDatamodule_3D(
            input_da=xr.open_dataarray(data_path["enatl"]),         # your xarray DataArray
        dl_kw={"batch_size": 4, "num_workers": 2},
        norm_stats={"method": "mean_std_along_depth"}, #, "params": {"mean": None, "std": None}  #"method":"min_max"
        manage_nan="supress_with_max_depth",
        n_profiles=None,
        reshape=None, #["factor_64"], #"RGB"
        dtype_str="float32"
        )


        datamodule.setup(stage="fit")
        train_dataloader = datamodule.train_dataloader()

        datamodule.setup(stage="test")
        test_dataloader = datamodule.test_dataloader()

        dm={"train": train_dataloader, "test": test_dataloader}

        with open(dm_path, 'wb') as f:
            pickle.dump(datamodule, f)

    cfg["in_channels"] = next(iter(dm['train'])).shape[1] # e.g., 3 for RGB, 157 for your current data

    # Save experiment configuration to log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = "mlicpp_on_ssp"  # Extract from sys.argv[4] if needed
    log_dir = f"/Odyssey/private/o23gauvr/code/MLIC/experiments/{experiment_name}_{cfg['N']}_{cfg['M']}_lambda{sys.argv[7]}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    config_log_path = os.path.join(log_dir, "experiment_config.log")
    with open(config_log_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"EXPERIMENT LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        f.write("COMMAND LINE ARGUMENTS:\n")
        f.write("-" * 25 + "\n")
        for i, arg in enumerate(sys.argv):
            f.write(f"argv[{i}]: {arg}\n")
        f.write("\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        for key, value in cfg.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("ADDITIONAL INFO:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Load datamodule: {load_datamodule}\n")
        f.write(f"Datamodule path: {dm_path}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"GPU count: {torch.cuda.device_count()}\n")

    print(f"Experiment configuration saved to: {config_log_path}")

    main(dm,cfg)
