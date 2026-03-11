import os 
import sys

from numpy import astype

running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)

import os
import random
import logging
from PIL import ImageFile, Image
import math
from tqdm import tqdm
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
from MLIC.MLIC.loss.rd_loss import *
from MLIC.MLIC.config.args import train_options
from MLIC.MLIC.config.config import model_config
from MLIC.MLIC.models import *
import random
import pickle
import xarray as xr
from datetime import datetime
from timm.scheduler import CosineLRScheduler
from FASCINATION.src.autoencoder_datamodule_natl_enatl import AEDatamodule as AEDatamodule_enatl_natl, AE_BaseDataset_3D
from FASCINATION.src.autoencoder_datamodule_good_split import AEDatamodule

from FASCINATION.src.utils import load_model 




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
    

def month_to_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall
    


def main(dm,config, loss_params,checkpoint_dir_name):

    # Log loss_params for experiment traceability
    print(f"[INFO] loss_params: {loss_params}")
    # Optionally, log to file if logger_train is not yet available
    loss_params_log_path = os.path.join(checkpoint_dir_name, "loss_params.log")
    with open(loss_params_log_path, 'w') as f:
        f.write("loss_params = " + str(loss_params) + "\n")

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
        dm.dl_kw['batch_size'] = args.test_batch_size
        val_dataloader = dm.val_dataloader()
        test_dataloader = dm.test_dataloader()
        depth_array = dm.depth_array
        train_norm_stats = dm.train_ds.input.attrs.get("norm_stats", None)
        test_norm_stats = dm.test_ds.input.attrs.get("norm_stats", None)

        #test_dataloader.dataset.input.attrs["norm_stats"] = norm_stats
        # test_dataloader.dataset.input.attrs["depth"] = depth_array
        # train_dataloader.dataset.input.attrs["depth"] = depth_array

    net = MLICPlusPlus(config=config)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    net = net.to(device)

    if loss_params["method"] == "original":
        criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)
    elif loss_params["method"] == "homoscedastic":
        criterion = HomoscedasticSSPLoss(**loss_params)
    elif loss_params["method"] == "fixed_weight":
        criterion = FixedWeightSSPLoss(**loss_params)
    elif loss_params["method"] == "dlw":
        criterion = DynamicLossWeightingSSPLoss(**loss_params)
    # elif loss_params["method"] == "factor":
    #     # Factor method: before warmup epochs, only recon=1.0; after, apply factor-based weights
    #     factor_params = loss_params.copy()
    #     factor_params["use_factor_weights"] = True
    #     factor_params["factor_warmup_epochs"] = loss_params.get("factor_warmup_epochs", 150)
    #     criterion = FixedWeightSSPLoss(**factor_params)       

    #criterion = HeteroscedasticSSPLoss(sharpness=15.0, lambda_minmax=2.0, lambda_inflection=1.0)
    #

    optimizer, aux_optimizer, loss_optimizer = configure_optimizers(net, criterion, args)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1) #[30,100,500] [500,1000,5000]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # lr_scheduler = CosineLRScheduler(
    # optimizer,
    # t_initial=100,    # cycle length
    # lr_min=1e-5,
    # warmup_t=5,
    # cycle_limit=100000,  #args.epochs//100,  # number of cycles   
    # warmup_lr_init=optimizer.defaults['lr'],
    # warmup_prefix=True,
    # cycle_decay=0.5   # <-- decays max LR each restart
    # )

    config_log_path = os.path.join(checkpoint_dir_name, "experiment_config.log")
    with open(config_log_path, 'a') as f:
        f.write("CRITERION PARAMETERS:\n")
        f.write("-" * 21 + "\n")
        for k, v in getattr(criterion, '__dict__', {}).items():
            if not k.startswith('_'):
                f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("LR SCHEDULER PARAMETERS:\n")
        f.write("-" * 23 + "\n")
        # Try to log the state_dict and class name for clarity
        f.write(f"Class: {type(lr_scheduler).__name__}\n")
        try:
            for k, v in lr_scheduler.state_dict().items():
                f.write(f"{k}: {v}\n")
        except Exception as e:
            f.write(f"Could not log lr_scheduler state_dict: {e}\n")
        f.write("\n")


    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # # Define layers to skip due to size mismatch
        set_up_layers = [               

        ]
            # 'g_a.analysis_transform.0.conv1.weight',
            # 'g_a.analysis_transform.0.skip.weight', 
            # 'g_s.synthesis_transform.7.0.weight',
            # 'g_s.synthesis_transform.7.0.bias' 

        #model_dict = net.state_dict()

        # Initialize the skipped layers
        # init_method = "kaiming_normal_"  # Change to "classical" for default initialization
        
        # for key in checkpoint['state_dict'].keys():


        #     if key in set_up_layers:
                
        #         if init_method == "kaiming_normal_":
        #             if 'weight' in key:
        #                 checkpoint['state_dict'][key] = nn.init.kaiming_normal_(model_dict[key], mode='fan_out', nonlinearity='relu')
        #                 logger_train.info(f"Initialized {key} with kaiming_normal_")
        #             elif 'bias' in key:
        #                 checkpoint['state_dict'][key] = nn.init.constant_(model_dict[key], 0)
        #                 logger_train.info(f"Initialized {key} with zeros")

        #         else:  # classical initialization
        #             checkpoint['state_dict'][key] = model_dict[key] 

        net.load_state_dict(checkpoint['state_dict'], strict=False)

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.learning_rate = optimizer.param_groups[0]['lr']
        start_epoch = 0 #checkpoint['epoch']
        best_loss =  1e10 #checkpoint.get("loss_dict",checkpoint).get("loss")
        current_step = 0 #start_epoch * (math.ceil(len(train_dataloader.dataset) / args.batch_size))
        checkpoint = None
    else:
        start_epoch = 0
        current_step = 0
        best_loss = 1e10
    

    best_ecs = 1e10
    best_f1 = 0.0
    best_bpp_loss = 1e10
    best_rmse = 1e10

    dir_path = os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name, 'checkpoints')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training Progress", unit="epoch"):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Apply factor-based weights after warmup epochs
        if vars(criterion).get("use_factor_weights",None):
            if epoch == criterion.factor_warmup_epochs and not criterion._factor_weights_applied:
                logger_train.info(f"Applying factor-based weights at epoch {epoch}...")
                criterion.apply_factor_weights(net, train_dataloader, device, logger_train)

        current_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            loss_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step,
            args.gradient_accumulation_steps,  # Add this parameter
            verbose=args.verbose
        )

        # if "log_vars" in dir(criterion):
        #     for i, l in enumerate(criterion.loss_dict.keys()):
        #         tb_logger.add_scalar('{}'.format(f'[train]: {l}_loss weight'), criterion.log_vars[i], epoch + 1)
        
        tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], epoch + 1)
        save_dir = os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name, 'val_images', '%03d' % (epoch + 1))
        loss_dict = test_one_epoch(epoch, val_dataloader, net, criterion, save_dir, logger_val, tb_logger)

        lr_scheduler.step(epoch) #lr_scheduler.step(epoch)

        is_best_loss = loss_dict['loss'] <= best_loss
        is_best_ecs = loss_dict['ecs_loss'] <= best_ecs
        is_best_rmse = loss_dict['rmse_loss'] <= best_rmse
        is_best_f1 = loss_dict['f1_score'] >= best_f1
        is_best_bpp = loss_dict['bpp_loss'] <= best_bpp_loss


        if is_best_loss:
            best_loss = loss_dict['loss']
        if is_best_ecs:
            best_ecs = loss_dict['ecs_loss']
        if is_best_bpp:
            best_bpp_loss = loss_dict['bpp_loss']
        if is_best_rmse:
            best_rmse = loss_dict['rmse_loss']
        if is_best_f1:
            best_f1 = loss_dict['f1_score']

        net.update(force=True)

        if any([is_best_loss, is_best_ecs, is_best_f1, is_best_bpp, is_best_rmse]):
            state = {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "loss_dict": loss_dict,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "loss_optimizer": loss_optimizer.state_dict() if loss_optimizer is not None else None,
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_norm_stats": train_norm_stats,
                "test_norm_stats": test_norm_stats,
            }

        # Save checkpoints for each best metric
        if args.save and is_best_loss:
            save_checkpoint(
                state=state,
                dir_path=dir_path,
                filename="best_checkpoint_loss.pth.tar"
            )
            logger_val.info('best checkpoint (loss) saved.')

        if args.save and is_best_ecs:
            save_checkpoint(
                state= state,
                dir_path=dir_path,
                filename="best_checkpoint_ecs.pth.tar"
            )
            logger_val.info('best checkpoint (ecs) saved.')

        if args.save and is_best_f1:
            save_checkpoint(
                state= state,
                dir_path=dir_path,
                filename="best_checkpoint_f1.pth.tar"
            )
            logger_val.info('best checkpoint (f1) saved.')

        if args.save and is_best_bpp:
            save_checkpoint(
                state= state,
                dir_path=dir_path,
                filename="best_checkpoint_bpp_loss.pth.tar"
            )
            logger_val.info('best checkpoint (bpp_loss) saved.')

        if args.save and is_best_rmse:
            save_checkpoint(
                state= state,
                dir_path=dir_path,
                filename="best_checkpoint_rmse.pth.tar"
            )
            logger_val.info('best checkpoint (rmse) saved.')


    # --- Evaluate on test set after training ---
    logger_train.info("Evaluating on test set after training...")
    test_save_dir = os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name, 'test_images')
    os.makedirs(test_save_dir, exist_ok=True)
    test_metrics = test_one_epoch(
        epoch + 1,  # or args.epochs
        test_dataloader,
        net,
        criterion,
        test_save_dir,
        logger_train,  # log to train logger for test set
        tb_logger,
        validation=False
    )

    logger_train.info(f"Test set metrics after training: {test_metrics}")
    

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, loss_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, gradient_accumulation_steps=1, verbose=False
):
    model.train()
    device = next(model.parameters()).device
    depth_array = train_dataloader.dataset.input.attrs.get("depth", None)
    season_idx_list = train_dataloader.dataset.input.attrs.get("season_idx", None)
    sst_full = train_dataloader.dataset.input.attrs.get("sst", None)

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        season_idx = season_idx_list[i * d.size(0):(i + 1) * d.size(0)] if season_idx_list is not None else None
        sst = sst_full[i * d.size(0):(i + 1) * d.size(0), :] if sst_full is not None else None

        # if cfg["add_embedded_seasons"]["use"]:
        #     if cfg["add_embedded_seasons"]["mode"] == "one_hot":
        #         #[month_to_season(m) for m in pd.DatetimeIndex(self.input["time"]).month.values]
        #         seasons = [month_to_season(m) for m in train_dataloader.dataset.input.time.dt.month]
        #         seasons_one_hot = F.one_hot(torch.tensor(seasons), num_classes=4).float().to(device)
        #         seasons_one_hot = seasons_one_hot.unsqueeze(1).unsqueeze(-1).repeat(1, d.size(1), 1, d.size(3))
        #         d = torch.cat([d, seasons_one_hot], dim=2)


        # Only zero gradients at the beginning of accumulation
        if i % gradient_accumulation_steps == 0:
            if verbose:
                print(f"🔄 Zeroing gradients at batch {i}")
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            if loss_optimizer is not None:
                loss_optimizer.zero_grad()

        out_net = model(d,season_idx, sst)
        #out_net['x_hat'] = out_net['x_hat'][:,-len(depth_array):,:]
        out_criterion = criterion(out_net, d)
        
        # Scale loss by accumulation steps
        loss = out_criterion["loss"] / gradient_accumulation_steps
        loss.backward()
        if verbose:
            print(f"📈 Accumulated gradients for batch {i}, scaled loss: {loss.item():.6f}")
        
        # Only step optimizer after accumulating gradients
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            if verbose:
                print(f"⚡ Stepping optimizer after {gradient_accumulation_steps} accumulations at batch {i}")
            
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss = model.aux_loss() / gradient_accumulation_steps
            aux_loss.backward()

            aux_optimizer.step()
            if loss_optimizer is not None:
                loss_optimizer.step()   


        current_step += 1

    return current_step

if __name__ == '__main__':  

    sys.argv = [
        "train.py",
        #"--metrics", "mse",
        "--exp", "test_loss", #"mlicpp_multi_loss_test_lr_step500",
        "--gpu_id", "0",
        "--epochs", "1500",
        "--lambda", "1.0",
        "-lr", "1e-4",
        "--num-workers", "10",
        "--clip_max_norm", "1.0",
        "--seed", "42",
        "--batch-size", "4",
        "--test-batch-size", "4",
        "--patch-size", "196", "256",
        "--gradient_accumulation_steps", "1",
        #"--checkpoint", "/Odyssey/private/o23gauvr/code/MLIC/checkpoints/mlicpp_mse_q5_2960000.pth.tar",
        "--save",
        #"--verbose", 
    ]

    # Parse dm_type from sys.argv
    # if "--dm-type" in sys.argv:
    #     dm_type_idx = sys.argv.index("--dm-type") + 1
    #     dm_type = sys.argv[dm_type_idx]
    # else:
    #     dm_type = "good_split"

    loss_params = {
        "method": "fixed_weight",  # Options: "dlw" "original", "homoscedastic", "fixed_weight",
        # For "factor" method: before factor_warmup_epochs, only recon=1.0; 
        # after, weights are normalized so that loss_dict values act as relative factors
        # e.g., weighted_recon=5.0 means 5x the weight of recon after normalization
        "loss_dict":{"recon": 1.0,
                    "weighted_recon": 1.0, 
                    "deriv": 1.0,
                    "weighted_deriv": 1.0,  # weighted derivative loss (emphasis on first depth indices)
                    "curvature_recon": 0.0,
                    "soft_peak": 0.0,  # soft peak localization
                    "wasserstein_peak": 0.0,  # Wasserstein peak alignment
                    "max_pos": 100.0,
                    "max_value": 1.0,
                    "extrema_pos": 0.0,
                    "extrema_value": 0.0},  
        "extrema_method": "minmax",
        "dlw_method": "uncertainty", #"uncertainty", "dwa", "gradnorm" #"ruw"
        "cr_treshold": 10000.0,
        "lmbda": float(sys.argv[8]),
        "use_smoothl1": True,
        "auto_normalize": False,
        "use_factor_weights": True,
        "factor_warmup_epochs": 150,  # Only used when method="factor"
    }
 
#        "lambda_deriv": 0.1,
#        "lambda_extrema": 0.5


    cfg = model_config()
    cfg["N"] = 64 #64 #1600 #192 #128 #192 #640  #128 
    cfg["M"] = 96 #96 #2400 #320 #192 #320 #960  #192
    cfg["slice_num"] = 6 #25 #6 #8 #10
    cfg["context_window"] = 5
    cfg['act'] = torch.nn.GELU
    cfg["enable_channel_context"] = True
    cfg["enable_local_context"] = True
    cfg["enable_global_inter_context"] = True
    cfg["enable_global_intra_context"] = True

    cfg["add_seasons"] = {"use":False, "mode":"embed"}
    cfg["add_sst"] = False

    rgb = {"use":False, "method":"depth_layers"} #PCA
    chn = "3" if rgb["use"] else "157"

    # if rgb["use"] and rgb["method"] == "CAE":
    #     cae_ckpt_path = "/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/CAE/CAE/channels_[5000, 3000, 1000, 3]/upsample_mode_trilinear/linear_layer_False/cr_100000/1_conv_per_layer/padding_cubic/interp_size_5/final_upsample_upsample/act_fn_LeakyRelu/use_final_act_fn_True/lr_0.001/normalization_mean_std_along_depth/manage_nan_supress_with_max_depth/n_profiles_None/2025-03-04_06-01/checkpoints/val_loss=0.01-epoch=970.ckpt"
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     rgb_model = load_model(cae_ckpt_path, device=device)
    #     rgb_model.eval()
    #     rgb["model"] = rgb_model



    load_datamodule = False
    save_dm = False
    norm_method = "mean_std_along_depth"
    dm_type = "good_split" #"enatl_natl" #"good_split"
    data_name = "enatl" #enatl, natl, natl_sst
    data_name_dir = data_name if dm_type=="good_split" else ""
    dm_path = f"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/{data_name}_dm_157_196_256_good_split.pkl"
    if load_datamodule and os.path.exists(dm_path):
        with open(dm_path, 'rb') as f:
            datamodule = pickle.load(f)
            #dm={"train": datamodule.train_dataloader(), "test": datamodule.test_dataloader()}

    else:

        data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                    "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc",
                    "natl_sst": "/Odyssey/public/natl60/raw/NATL60GULF-CJM165_degraded_vosaline_regrid.nc"}


        # datamodule = AutoEncoderDatamodule_3D(
        #     input_da=xr.open_dataarray(data_path[data]),         # your xarray DataArray
        # dl_kw={"batch_size": int(sys.argv[18]), "num_workers": int(sys.argv[12])},
        # norm_stats={"method": "min_max"}, #, "params": {"mean": None, "std": None}  #"method":"min_max"
        # manage_nan="supress_with_max_depth",
        # n_profiles=None,
        # reshape=["factor_64"], #["factor_64"], #"RGB"
        # rgb=rgb,
        # dtype_str="float32"
        # )

        if dm_type == "enatl_natl":
            datamodule = AEDatamodule_enatl_natl(
                dl_kw={"batch_size": int(sys.argv[18]), "num_workers": int(sys.argv[12])},
                norm_stats={"method": norm_method},
                test_norm="on_test",
                manage_nan="supress_with_max_depth",
                reshape=["factor_64"],
                rgb=rgb,
                dtype_str="float32"
            )
        else:
            datamodule = AEDatamodule(
                data_name=data_name,
                dl_kw={"batch_size": int(sys.argv[18]), "num_workers": int(sys.argv[12])},
                norm_stats={"method": norm_method},
                manage_nan="supress_with_max_depth",
                reshape=["factor_64"],
                rgb=rgb,
                dtype_str="float32",
                shuffle=True,
            )
        datamodule.setup()
        if save_dm:
            with open(dm_path, 'wb') as f:
                pickle.dump(datamodule, f)

        #dm={"train": train_dataloader, "test": test_dataloader}


                # Interpolate SST and mean-std normalize

    #xr.open_dataset("/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc").sel(time=datamodule.test_dataloader().dataset.input.time.values)

    datamodule.dl_kw['batch_size'] = int(sys.argv[18])
    datamodule.dl_kw['num_workers'] = int(sys.argv[12])

    cfg["in_channels"] = datamodule.test_shape[1] # e.g., 3 for RGB, 157 for your current data

    if cfg["add_seasons"]["use"]==True and cfg["add_seasons"]["mode"]=="embed":
        cfg["in_channels"] += 8  # Add one channel for season embedding
    elif cfg["add_seasons"]["use"]==True and cfg["add_seasons"]["mode"]=="one_hot":
        cfg["in_channels"] += 4  # Add four channels for one-hot season encoding

    if cfg["add_sst"]:
        cfg["in_channels"] += 1  # Add one channel for SST

    # Save experiment configuration to log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir_name = f"/Odyssey/private/o23gauvr/code/MLIC/experiments/{sys.argv[2]}/{loss_params['method']}_loss_{cfg['N']}_{cfg['M']}_{float(sys.argv[8])}_CR_{loss_params['cr_treshold']}_{dm_type}_{data_name_dir}_{norm_method}/{timestamp}"
    os.makedirs(checkpoint_dir_name, exist_ok=True)
    
    config_log_path = os.path.join(checkpoint_dir_name, "experiment_config.log")
    with open(config_log_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"EXPERIMENT LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Datamodule type: {dm_type}\n")
        f.write(f"Data name: {data_name}\n")
        f.write(f"Norm method: {norm_method}\n")
        f.write("\n")
        
        f.write("COMMAND LINE ARGUMENTS:\n")
        f.write("-" * 25 + "\n")
        for i, arg in enumerate(sys.argv):
            f.write(f"argv[{i}]: {arg}\n")
        f.write("\n")

        f.write("LOSS FUNCTION PARAMETERS:\n")
        f.write("-" * 27 + "\n")
        for key, value in loss_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 22 + "\n")
        f.write(f"Actual batch size: {sys.argv[18]}\n")
        f.write(f"Gradient accumulation steps: {sys.argv[25]}\n")
        f.write(f"Effective batch size: {int(sys.argv[18])*int(sys.argv[25])}\n")
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

    main(datamodule,cfg,loss_params,checkpoint_dir_name)
