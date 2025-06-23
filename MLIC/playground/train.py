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
from FASCINATION.src.autoencoder_datamodule import AutoEncoderDatamodule_3D



def main(dm,config):
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    #config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
    #     seed = args.seed
    # else:
        seed = 100 * random.random()
    torch.manual_seed(seed)
    random.seed(seed)

    if not os.path.exists(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', args.experiment)):
        os.makedirs(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', args.experiment))

    setup_logger('train', os.path.join('./experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('./experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='/Odyssey/private/o23gauvr/code/MLIC/tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', args.experiment, 'checkpoints'))

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

        save_dir = os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
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
                os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')

if __name__ == '__main__':  


    sys.argv = [
        "train.py",
        "--metrics", "mse",
        "--exp", "mlicpp_on_ssp",
        "--gpu_id", "0",
        "--epochs", "34000",
        "--lambda", "0.0018",
        "-lr", "1e-4",
        "--num-workers", "16",
        "--clip_max_norm", "1.0",
        "--seed", "42",
        "--batch-size", "32",
        "--patch-size", "196", "256"
    ]

    cfg = model_config()
    cfg["N"] = 249
    cfg["M"] = 320
    # cfg["slice_num"] = 10
    # cfg["context_window"] = 5


    load_datamodule = True
    dm_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/natl_dm_4_157_196_256.pkl"


    if load_datamodule:
        with open(dm_path, 'rb') as f:
            datamodule = pickle.load(f)
            dm={"train": datamodule.train_dataloader(), "test": datamodule.test_dataloader()}

    else:

        data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}


        datamodule = AutoEncoderDatamodule_3D(
        input_da=xr.open_dataarray(data_path["natl"]),         # your xarray DataArray
        dl_kw={"batch_size": 4, "num_workers": 2},
        norm_stats={"method": "min_max", "params": {"mean": None, "std": None}},
        pooled_dim="dense",
        depth_pre_treatment={"method": None},
        manage_nan="supress_with_max_depth",
        n_profiles=None,
        reshape=["factor_64"], #"RGB"
        dtype_str="float32"
        )


        datamodule.setup(stage="fit")
        train_dataloader = datamodule.train_dataloader()

        datamodule.setup(stage="test")
        test_dataloader = datamodule.test_dataloader()

        dm={"train": train_dataloader, "test": test_dataloader}

        with open(dm_path, 'wb') as f:
            pickle.dump(datamodule, f)



    main(dm,cfg)
