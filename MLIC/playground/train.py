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



def compute_total_bits(out_net):
    return sum(torch.log(likelihoods).sum() / (-math.log(2))
              for likelihoods in out_net['likelihoods'].values()).item()

def log_compression_rate_before_training(net, test_dataloader, device, logger_train):
    """
    Compute and log compression rate before training starts
    """
    net.eval()
    total_bits = 0
    total_elements = 0
    total_original_bits = 0
    
    logger_train.info("Computing initial compression rate...")
    
    with torch.no_grad():
        # Test on a few batches to get average compression rate
        for i, d in enumerate(test_dataloader):
            if i >= 5:  # Only test on first 5 batches for speed
                break
                
            d = d.to(device)
            
            try:
                rv = net(d)
                
                bits = compute_total_bits(rv)
                numel = rv['x_hat'].numel()
                original_bits = numel * 8
                
                total_bits += bits
                total_elements += numel
                total_original_bits += original_bits
                
            except Exception as e:
                logger_train.warning(f"Error computing compression rate for batch {i}: {e}")
                continue
    
    if total_elements > 0:
        avg_bpe = total_bits / total_elements
        avg_cr = total_original_bits / total_bits
        
        logger_train.info("="*50)
        logger_train.info("INITIAL COMPRESSION METRICS")
        logger_train.info("="*50)
        logger_train.info(f"Average Bits Per Element: {avg_bpe:.6f}")
        logger_train.info(f"Average Compression Rate: {avg_cr:.2f}x")
        logger_train.info(f"Total compressed bits: {total_bits:.2f}")
        logger_train.info(f"Total original bits: {total_original_bits:.2f}")
        logger_train.info("="*50)
        
        return avg_bpe, avg_cr
    else:
        logger_train.warning("Could not compute compression rate - no valid batches processed")
        return None, None

def main(dm,config,checkpoint_dir_name):
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    #config = model_config()

    # Create timestamp for unique checkpoint directory
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #checkpoint_dir_name = f"{args.experiment}_{config['N']}_{config['M']}_{args.lmbda}_{dm.norm_stats['method']}/{timestamp}"

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
    tb_logger = SummaryWriter(log_dir=os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments/', checkpoint_dir_name , "tb_logger"))

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
        train_dataloader = dm.train_dataloader()
        test_dataloader = dm.test_dataloader()

    net = MLICPlusPlus(config=config)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)

    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        
        # # Define layers to skip due to size mismatch
        set_up_layers = [               

        ]
            # 'g_a.analysis_transform.0.conv1.weight',
            # 'g_a.analysis_transform.0.skip.weight', 
            # 'g_s.synthesis_transform.7.0.weight',
            # 'g_s.synthesis_transform.7.0.bias' 

        model_dict = net.state_dict()

        # Initialize the skipped layers
        init_method = "kaiming_normal_"  # Change to "classical" for default initialization
        
        for key in checkpoint['state_dict'].keys():


            if key in set_up_layers:
                
                if init_method == "kaiming_normal_":
                    if 'weight' in key:
                        checkpoint['state_dict'][key] = nn.init.kaiming_normal_(model_dict[key], mode='fan_out', nonlinearity='relu')
                        logger_train.info(f"Initialized {key} with kaiming_normal_")
                    elif 'bias' in key:
                        checkpoint['state_dict'][key] = nn.init.constant_(model_dict[key], 0)
                        logger_train.info(f"Initialized {key} with zeros")

                else:  # classical initialization
                    checkpoint['state_dict'][key] = model_dict[key] 

        net.load_state_dict(checkpoint['state_dict'], strict=False)


        
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1)
        # start_epoch = checkpoint['epoch']
        # best_loss = checkpoint['loss']
        # current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
        # checkpoint = None
    else:
        pass
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
    
    # ADD THIS: Log compression rate before training
    initial_bpe, initial_cr = log_compression_rate_before_training(
        net, test_dataloader, device, logger_train
    )
    
    # Also add to config log file
    config_log_path = os.path.join(checkpoint_dir_name, "experiment_config.log")
    with open(config_log_path, 'a') as f:  # Append mode
        f.write("\nINITIAL COMPRESSION METRICS:\n")
        f.write("-" * 29 + "\n")
        if initial_bpe is not None:
            f.write(f"Initial Bits Per Element: {initial_bpe:.6f}\n")
            f.write(f"Initial Compression Rate: {initial_cr:.2f}x\n")
        else:
            f.write("Could not compute initial compression metrics\n")
        f.write("\n")
    
    # Continue with training loop
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
            current_step,
            args.gradient_accumulation_steps  # Add this parameter
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

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, gradient_accumulation_steps=1
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        # Only zero gradients at the beginning of accumulation
        if i % gradient_accumulation_steps == 0:
            print(f"🔄 Zeroing gradients at batch {i}")
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        
        # Scale loss by accumulation steps
        loss = out_criterion["loss"] / gradient_accumulation_steps
        loss.backward()
        print(f"📈 Accumulated gradients for batch {i}, scaled loss: {loss.item():.6f}")
        
        # Only step optimizer after accumulating gradients
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            print(f"⚡ Stepping optimizer after {gradient_accumulation_steps} accumulations at batch {i}")
            
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss = model.aux_loss() / gradient_accumulation_steps
            aux_loss.backward()
            aux_optimizer.step()

        current_step += 1

    return current_step

if __name__ == '__main__':  

    sys.argv = [
        "train.py",
        "--metrics", "mse",
        "--exp", "mlicpp_on_ssp",
        "--gpu_id", "0",
        "--epochs", "34000",
        "--lambda", "0.0018" , #"0.0250",  #0.0018, 0.0035, 0.0067, 0.0130, 0.0250 , 0.0483
        "-lr", "1e-4",
        "--num-workers", "16",
        "--clip_max_norm", "1.0",
        "--seed", "42",
        "--batch-size", "4",  # Reduced batch size for gradient accumulation
        "--patch-size", "196", "256",
        "--gradient_accumulation_steps", "32",  # Add gradient accumulation
        #"--checkpoint", "/Odyssey/private/o23gauvr/code/MLIC/checkpoints/renamed_mlicpp_mse_q5_2960000.pth.tar"  # Add this line

    ]

    cfg = model_config()
    cfg["N"] = 1600 #64 #1600 #192 #128 #192 #640  #128 
    cfg["M"] = 2400 #96 #2400 #320 #192 #320 #960  #192
    cfg["slice_num"] = 25 #25 #6 #8 #10
    # cfg["context_window"] = 5


    load_datamodule = True
    dm_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_4_157_196_256.pkl" #enatl_dm_4_157_196_256.pkl"
    save_dm = False
    data = "enatl"  # "enatl" or "natl"

    if load_datamodule:
        with open(dm_path, 'rb') as f:
            datamodule = pickle.load(f)
            #dm={"train": datamodule.train_dataloader(), "test": datamodule.test_dataloader()}

    else:

        data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                    "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}


        datamodule = AutoEncoderDatamodule_3D(
            input_da=xr.open_dataarray(data_path[data]),         # your xarray DataArray
        dl_kw={"batch_size": 16, "num_workers": 16},
        norm_stats={"method": "min_max"}, #, "params": {"mean": None, "std": None}  #"method":"min_max"
        manage_nan="supress_with_max_depth",
        n_profiles=None,
        reshape=["factor_64","RGB"], #["factor_64"], #"RGB"
        dtype_str="float32"
        )


        datamodule.setup(stage="fit")
        train_dataloader = datamodule.train_dataloader()

        datamodule.setup(stage="test")
        test_dataloader = datamodule.test_dataloader()

        dm={"train": train_dataloader, "test": test_dataloader}

        if save_dm:
            with open(dm_path, 'wb') as f:
                pickle.dump(datamodule, f)

    cfg["in_channels"] = datamodule.test_shape[1] # e.g., 3 for RGB, 157 for your current data

    # Save experiment configuration to log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir_name = f"/Odyssey/private/o23gauvr/code/MLIC/experiments/{sys.argv[4]}_{cfg['N']}_{cfg['M']}_{float(sys.argv[10])}_{datamodule.norm_stats['method']}/{timestamp}"
    os.makedirs(checkpoint_dir_name, exist_ok=True)
    
    config_log_path = os.path.join(checkpoint_dir_name, "experiment_config.log")
    with open(config_log_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"EXPERIMENT LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        f.write("COMMAND LINE ARGUMENTS:\n")
        f.write("-" * 25 + "\n")
        for i, arg in enumerate(sys.argv):
            f.write(f"argv[{i}]: {arg}\n")
        f.write("\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 22 + "\n")
        f.write(f"Actual batch size: {sys.argv[20]}\n")
        f.write(f"Gradient accumulation steps: {sys.argv[25]}\n")
        f.write(f"Effective batch size: {int(sys.argv[20])*int(sys.argv[25])}\n")
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

    main(datamodule,cfg,checkpoint_dir_name)
