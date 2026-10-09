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
from MLIC.MLIC.utils.grad_monitor import TermGradientLogger
from MLIC.MLIC.utils.grad_surgery import split_loss, projected_backward
from MLIC.MLIC.utils.resume import criterion_state, restore_criterion_state

# Per-term gradient logging (utils/grad_monitor.py) -> <run dir>/grad_terms.csv.
# One extra backward per active term on each logged batch. 0 disables it.
GRAD_LOG_EVERY_EPOCHS = 10
GRAD_LOG_BATCHES = 2
from MLIC.MLIC.loss.rd_loss import *
from MLIC.MLIC.config.args import train_options
from MLIC.MLIC.config.config import model_config
from MLIC.MLIC.models import *
import random
import pickle
import xarray as xr
from datetime import datetime, timedelta
from timm.scheduler import CosineLRScheduler
from MLIC.MLIC.utils.lr_scheduler import CosineWithFloor, SmoothCosineDecay
from FASCINATION.src.autoencoder_datamodule_natl_enatl import AEDatamodule as AEDatamodule_enatl_natl
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
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    #config = model_config()

    # Create timestamp for unique checkpoint directory
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #checkpoint_dir_name = f"{args.experiment}_{config['N']}_{config['M']}_{args.lmbda}_{dm.norm_stats['method']}/{timestamp}"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = int(100 * random.random())
    torch.manual_seed(seed)
    random.seed(seed)

    if not os.path.exists(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name)):
        os.makedirs(os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name))

    # Ensure the logger directory path matches the absolute path
    logger_dir = os.path.join('/Odyssey/private/o23gauvr/code/MLIC/experiments', checkpoint_dir_name)
    setup_logger('train', logger_dir, 'train' , level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', logger_dir, 'val', level=logging.INFO,
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

    depth_array = None
    native_depth_array = None
    train_norm_stats = None
    test_norm_stats = None

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
        # The levels the field natively has, before any depth_grid resampling. Used
        # as the reference for cr_treshold so the bit budget does not depend on how
        # many levels this particular model runs on.
        native_depth_array = getattr(dm, "original_depth_array", None)
        train_norm_stats = dm.train_ds.input.attrs.get("norm_stats", None)
        test_norm_stats = dm.test_ds.input.attrs.get("norm_stats", None)

        #test_dataloader.dataset.input.attrs["norm_stats"] = norm_stats
        # test_dataloader.dataset.input.attrs["depth"] = depth_array
        # train_dataloader.dataset.input.attrs["depth"] = depth_array

    net = MLICPlusPlus(config=config)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    net = net.to(device)

    if dm.dtype_str == "float64":
        net = net.double()

    # The structure losses (soft_max_pos, matched_extrema_pos, prominence_recall)
    # are defined on physical m/s profiles: with mean_std_along_depth the argmax
    # over depth of the *normalised* profile is the largest anomaly relative to
    # that level's climatology, which is not the sound-speed maximum and not what
    # any evaluation metric looks at. Handing the loss the training stats lets it
    # undo the normalisation internally.
    # `cr_treshold` is a floor on the compression ratio, and its reference used to be
    # the field on the model's own depth grid: at cr_treshold=10000 and float32 that
    # is 0.5024 bits/profile for a 157-level model but 0.2048 for a 64-level one, so
    # three runs asking for "CR 10000" trained at three different rates. Passing the
    # native level count makes cr_treshold target `cr_native` instead, i.e. the same
    # bits per water column whatever grid the model uses. Identical behaviour when the
    # model already runs on the native axis.
    native_levels = len(native_depth_array) if native_depth_array is not None else None
    if native_levels and depth_array is not None and len(depth_array) != native_levels:
        print(f"[rate] model grid has {len(depth_array)} levels, native has "
              f"{native_levels}; cr_treshold targets the native field.", flush=True)
    loss_kwargs = dict(depth_array=depth_array, norm_stats=train_norm_stats,
                       native_depth_levels=native_levels)

    if loss_params["method"] == "original":
        criterion = RateDistortionLoss(
            lmbda=args.lmbda, metrics=args.metrics,
            native_depth_levels=native_levels,
            rate_reference_bits_per_level=loss_params.get("rate_reference_bits_per_level"))
    elif loss_params["method"] == "homoscedastic":
        criterion = HomoscedasticSSPLoss(**loss_kwargs, **loss_params)
    elif loss_params["method"] == "fixed_weight":
        criterion = FixedWeightSSPLoss(**loss_kwargs, **loss_params)
    elif loss_params["method"] == "dlw":
        criterion = DynamicLossWeightingSSPLoss(**loss_kwargs, **loss_params)
    # Say which terms will actually train, first thing in the job output. Three
    # "E1b" runs went 17 h with local_extrema_pos silently at 0 because the launch
    # dict carried the key twice.
    active = {k: v for k, v in (getattr(criterion, "loss_dict", {}) or {}).items()
              if float(v or 0) > 0}
    print(f"[loss] terms with non-zero weight: {active}", flush=True)
    # elif loss_params["method"] == "factor":
    #     # Factor method: before warmup epochs, only recon=1.0; after, apply factor-based weights
    #     factor_params = loss_params.copy()
    #     factor_params["use_factor_weights"] = True
    #     factor_params["factor_warmup_epochs"] = loss_params.get("factor_warmup_epochs", 150)
    #     criterion = FixedWeightSSPLoss(**factor_params)       

    #criterion = HeteroscedasticSSPLoss(sharpness=15.0, lambda_minmax=2.0, lambda_inflection=1.0)
    #

    optimizer, aux_optimizer, loss_optimizer = configure_optimizers(net, criterion, args)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[350, 500], gamma=0.1) #[30,100,500] [500,1000,5000]
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
    #lr_scheduler = CosineWithFloor(optimizer, T=500, eta_min=1e-6)  # <-- Use the custom scheduler with a floor
    # lr_scheduler = CosineLRScheduler(
    # optimizer,
    # t_initial=500,    # cycle length
    # lr_min=1e-6,
    # warmup_t=5,
    # cycle_limit=100000,  #args.epochs//100,  # number of cycles   
    # warmup_lr_init=optimizer.defaults['lr'],
    # warmup_prefix=True,
    # cycle_decay=0.9   # <-- decays max LR each restart
    # )
    lr_scheduler = SmoothCosineDecay(
        optimizer,
        t_initial=750,    # cycle length
        lr_min=1e-6,
        warmup_t=5,
        cycle_limit=100000,  #args.epochs//100,  # number of cycles
        warmup_lr_init=optimizer.defaults['lr'],
        cycle_decay=0.5   # <-- decays max LR each restart
    )

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
        resume_checkpoint = checkpoint if args.resume else None

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
        if args.resume:
            raise ValueError("--resume needs --checkpoint <run>/checkpoints/<file>.pth.tar")
        resume_checkpoint = None
        start_epoch = 0
        current_step = 0
        best_loss = 1e10
    

    best_ecs = 1e10
    best_f1 = 0.0
    best_bpp_loss = 1e10
    best_rmse = 1e10

    # --resume: continue the run instead of fine-tuning from it (utils/resume.py).
    # The weights are already loaded above; restore everything else.
    if resume_checkpoint is not None:
        ck = resume_checkpoint
        # load_state_dict(strict=False) above would silently accept a different model
        # (e.g. the output smoother switched off): a resume must be the same network.
        mismatch = set(net.state_dict()) ^ set(ck["state_dict"])
        if mismatch:
            raise RuntimeError(f"--resume: model and checkpoint differ in {len(mismatch)} "
                               f"keys, e.g. {sorted(mismatch)[:3]}. Use the checkpoint's "
                               "model config (see its experiment_config.log)")
        optimizer.load_state_dict(ck["optimizer"])
        aux_optimizer.load_state_dict(ck["aux_optimizer"])
        if loss_optimizer is not None and ck.get("loss_optimizer") is not None:
            loss_optimizer.load_state_dict(ck["loss_optimizer"])
        lr_scheduler.load_state_dict(ck["lr_scheduler"])
        start_epoch = int(ck["epoch"])           # saved as epoch + 1 = epochs completed
        if start_epoch >= args.epochs:
            raise ValueError(f"checkpoint has {start_epoch} epochs done, --epochs is "
                             f"{args.epochs}: nothing left to train")
        current_step = start_epoch * len(train_dataloader)
        args.learning_rate = optimizer.param_groups[0]['lr']
        restore_criterion_state(criterion, ck, args.checkpoint, start_epoch, logger_train)
        # Best values: restored when the checkpoint carries them (written since 28 Sep);
        # otherwise every best_checkpoint_* is re-established in the new run directory.
        best = ck.get("best") or {}
        best_loss = best.get("loss", best_loss)
        best_ecs = best.get("ecs", best_ecs)
        best_f1 = best.get("f1", best_f1)
        best_bpp_loss = best.get("bpp_loss", best_bpp_loss)
        best_rmse = best.get("rmse", best_rmse)
        msg = (f"[resume] {args.checkpoint}: epoch {start_epoch} -> {args.epochs}, "
               f"lr {args.learning_rate:.4g}, best values "
               f"{'restored' if best else 'reset (legacy checkpoint)'}")
        logger_train.info(msg)
        with open(os.path.join(checkpoint_dir_name, "experiment_config.log"), 'a') as f:
            f.write(f"\nRESUMED FROM: {args.checkpoint}\n{msg}\n\n")
        del ck
        resume_checkpoint = None

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
    
    # Per-term gradient norms -> <run dir>/grad_terms.csv (see utils/grad_monitor.py).
    # Also logged on the two epochs after the factor weights switch on, which is
    # when the non-recon terms start to pull.
    warmup = vars(criterion).get("factor_warmup_epochs", None)
    grad_logger = TermGradientLogger(
        checkpoint_dir_name,
        [p for g in optimizer.param_groups for p in g["params"]],
        every_epochs=GRAD_LOG_EVERY_EPOCHS,
        batches_per_epoch=GRAD_LOG_BATCHES,
        clip_max_norm=args.clip_max_norm,
        extra_epochs=(warmup, warmup + 1) if warmup is not None else (),
    ) if GRAD_LOG_EVERY_EPOCHS else None

    # Gradient remedies for the structure terms (report 14.7), off by default.
    if args.grad_weighting and not (hasattr(criterion, "apply_gradient_weights")
                                    and vars(criterion).get("use_factor_weights")):
        raise ValueError("--grad_weighting needs method 'fixed_weight' with "
                         "use_factor_weights=True: it calibrates at the factor warm-up")
    if args.grad_projection and not hasattr(criterion, "last_weighted_terms"):
        raise ValueError("--grad_projection needs a criterion that records "
                         "last_weighted_terms (method 'fixed_weight')")
    logger_train.info(f"Gradient remedies: projection={args.grad_projection}, "
                      f"weighting={args.grad_weighting} (ratio={args.grad_weight_ratio}, "
                      f"every={args.grad_weight_every}, batches={args.grad_weight_batches})")

    # Continue with training loop
    optimizer.param_groups[0]['lr'] = args.learning_rate
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training Progress", unit="epoch"):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Apply factor-based weights after warmup epochs
        if vars(criterion).get("use_factor_weights",None):
            if epoch == criterion.factor_warmup_epochs and not criterion._factor_weights_applied:
                logger_train.info(f"Applying factor-based weights at epoch {epoch}...")
                criterion.apply_factor_weights(net, train_dataloader, device, logger_train)
                if args.grad_weighting:
                    criterion.apply_gradient_weights(
                        net, train_dataloader, device, ratio=args.grad_weight_ratio,
                        n_batches=args.grad_weight_batches, logger=logger_train)
            elif (args.grad_weighting and args.grad_weight_every > 0
                  and criterion._factor_weights_applied
                  and epoch > criterion.factor_warmup_epochs
                  and (epoch - criterion.factor_warmup_epochs) % args.grad_weight_every == 0):
                logger_train.info(f"Re-calibrating gradient-norm weights at epoch {epoch}...")
                criterion.apply_gradient_weights(
                    net, train_dataloader, device, ratio=args.grad_weight_ratio,
                    n_batches=args.grad_weight_batches, logger=logger_train)

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
            verbose=args.verbose,
            grad_logger=grad_logger,
            grad_projection=args.grad_projection,
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
            best_loss = loss_dict['loss']   ##TODO change numpy scallar to torch scallar as weights_only=True does not trust numpy scalar
        if is_best_ecs:
            best_ecs = loss_dict['ecs_loss']
        if is_best_bpp:
            best_bpp_loss = loss_dict['bpp_loss']
        if is_best_rmse:
            best_rmse = loss_dict['rmse_loss']
        if is_best_f1:
            best_f1 = loss_dict['f1_score']

        net.update(force=True)

        # Built every epoch: last_checkpoint.pth.tar used to carry the state of the
        # last epoch that improved *some* metric, not of the last epoch.
        state = {
            "epoch": epoch + 1,
            "state_dict": net.state_dict(),
            "loss_dict": loss_dict,
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "loss_optimizer": loss_optimizer.state_dict() if loss_optimizer is not None else None,
            "lr_scheduler": lr_scheduler.state_dict(),
            "depth_array": depth_array,
            "train_norm_stats": train_norm_stats,   ##TODO change numpy scallar to torch scallar as weights_only=True does not trust numpy scalar
            "test_norm_stats": test_norm_stats,   ##TODO change numpy scallar to torch scallar as weights_only=True does not trust numpy scalar
            "model": "MLIC",
            "criterion_state": criterion_state(criterion),
            "best": {"loss": best_loss, "ecs": best_ecs, "f1": best_f1,
                     "bpp_loss": best_bpp_loss, "rmse": best_rmse},
        }

        if args.save:
             save_checkpoint(
                state=state,
                dir_path=dir_path,
                filename="last_checkpoint.pth.tar"
            )           

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
    model, criterion, train_dataloader, optimizer, aux_optimizer, loss_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, gradient_accumulation_steps=1, verbose=False,
    grad_logger=None,
    grad_projection=False,
):
    model.train()
    device = next(model.parameters()).device
    depth_array = train_dataloader.dataset.input.attrs.get("depth", None)
    season_idx_list = train_dataloader.dataset.input.attrs.get("season_idx", None)
    sst_full = train_dataloader.dataset.input.attrs.get("sst", None)
    # Epoch means of every term, on the tensors the optimiser actually sees. The
    # validation numbers are computed on m/s and cannot stand in for these.
    term_sums, n_batches = {}, 0
    proj_stats = []
    params = [p for p in model.parameters() if p.requires_grad]

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
        for k, v in out_criterion.items():
            if torch.is_tensor(v) and v.numel() == 1:
                term_sums[k] = term_sums.get(k, 0.0) + v.detach().item()
        n_batches += 1
        if grad_logger is not None and grad_logger.wants(epoch, i):
            grad_logger.log(epoch, current_step, criterion, out_criterion)
        
        # Scale loss by accumulation steps
        loss = out_criterion["loss"] / gradient_accumulation_steps
        anchor, aux = split_loss(criterion, out_criterion) if grad_projection else (None, None)
        if aux is not None:
            # grad(value terms + rate) + grad(structure terms), the latter with its
            # component along the former removed when they conflict.
            proj_stats.append(projected_backward(
                params, anchor, aux, scale=1.0 / gradient_accumulation_steps))
        else:
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

    if n_batches:
        means = {k: v / n_batches for k, v in term_sums.items()}
        for k, v in means.items():
            tb_logger.add_scalar(f"[Train]: {k}", v, epoch + 1)
        logger_train.info(f"Train epoch {epoch}: " + " | ".join(
            f"{k}: {v:.6g}" for k, v in means.items()))
    if proj_stats:
        n = len(proj_stats)
        frac = sum(s["projected"] for s in proj_stats) / n
        cos = sum(s["cos"] for s in proj_stats) / n
        kept = sum(s["aux_norm_after"] / s["aux_norm"] for s in proj_stats
                   if s["aux_norm"] > 0) / n
        tb_logger.add_scalar("[Train]: proj_conflict_frac", frac, epoch + 1)
        tb_logger.add_scalar("[Train]: proj_cos_aux_anchor", cos, epoch + 1)
        tb_logger.add_scalar("[Train]: proj_aux_norm_kept", kept, epoch + 1)
        logger_train.info(f"Projection epoch {epoch}: conflicting steps {frac:.2f} | "
                          f"mean cos(aux, anchor) {cos:+.3f} | aux norm kept {kept:.3f}")

    return current_step

if __name__ == '__main__':  

    # Launch knobs, overridable from the environment so a sweep needs no file edits:
    #   SEED=7 N=32 M=48 SLICES=3 python train.py
    # Defaults = chapter-2 vanilla frame (todo Task 6): recon only, smoother off,
    # batch 4, lr 1e-5 (smoother_report.md §11-12). EXP defaults to H/off/s{SLICES}.
    env = os.environ.get
    SLICES = int(env("SLICES", "6"))
    sys.argv = [
        "train.py",
        #"--metrics", "mse",
        "--exp", env("EXP", f"back_to_the_future/H/off/64_96_3"),
        "--gpu_id", env("GPU", "0"),
        "--epochs", env("EPOCHS", "2000"),
        "--lambda", "1.0",
        "-lr", env("LR", "1e-5"),
        "--num-workers", "10",
        "--clip_max_norm", "1.0",
        "--seed", env("SEED", "42"), # 42, 7, 666
        "--batch-size", env("BATCH", "4"),
        "--test-batch-size", "4",
        "--patch-size", "196", "256",
        "--gradient_accumulation_steps", "1",
        #"--checkpoint", "/Odyssey/private/o23gauvr/code/MLIC/experiments/back_to_the_future/PF2/20260923_194034/checkpoints/best_checkpoint_rmse.pth.tar",
        #"--resume",
        "--save",
        #"--verbose", 
        # Gradient remedies for the structure terms (report 14.7). Keep them AFTER
        # the flags above: the launch block reads sys.argv by position.
        #"--grad_projection",
        #"--grad_weighting", "--grad_weight_ratio", "0.2", "--grad_weight_every", "100",
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
                    "weighted_recon": 0.0, 
                    "deriv": 0.0,
                    "weighted_deriv": 0.0,  # weighted derivative loss (emphasis on first depth indices)
                    "curvature_recon": 0.0,
                    "soft_peak": 0.0,  # soft peak localization
                    "lsd": 0.0,  # log spectral distance
                    "wasserstein_peak": 0.0,  # Wasserstein peak alignment
                    # "max_pos": 0.0,
                    # "max_value": 0.0,
                    # "extrema_pos": 0.0,
                    # "extrema_value": 0.0},  
                    "soft_max_pos": 0.0,
                    "matched_extrema_pos": 0.0,   # degenerate (report 14.4b); kept to reproduce E1
                    "local_extrema_pos": 0.0,     # E1b. Keep ONE entry: in a dict literal the last duplicate wins silently
                    "prominence_recall": 0.0,
        },
        "structure_params": {
                             # matched_extrema_pos only (E1 as run); inert otherwise
                             "search_m": 10.0,
                             # local_extrema_pos: capture window in true metres per
                             # level. Wider than the tolerance to improve: F1 goes
                             # 0.26 -> ~0.6 between +-10 and +-40 m, so 40 m reaches
                             # most of the misses.
                             "capture_m": 40.0,
                             # softmax temperature, relative to each extremum's
                             # prominence (a level half a prominence down keeps e^-2)
                             "local_beta": 4.0,
                             # scale only; cancelled by the factor normalisation
                             "pos_scale_m": 50.0,
                             # reference-extremum gate, shared by local_extrema_pos
                             # and prominence_recall. NB: converted to levels via the
                             # median dz, so "100 m" is +-23 m at the surface and
                             # +-120 m at 1000 m (report 14.4b). Kept as in E1/E2.
                             "prominence_scales_m": [10.0, 25.0, 50.0, 100.0],
                             "min_prominence_frac": 0.05,
                             },
        "extrema_method": "minmax",
        #"dlw_method": "uncertainty", #"uncertainty", "dwa", "gradnorm" #"ruw"
        # cr_treshold is a FLOOR on the compression ratio, imposed as
        # ReLU(bits_per_profile - reference/cr_treshold). Since 22 Sep the reference
        # is the NATIVE field (train.py passes native_depth_levels), so this targets
        # `cr_native` -- the column test_metrics.py reports -- and means the same bit
        # budget per water column whatever depth grid the model runs on. Before that
        # it referenced the model's own grid, so a 64-level model got 0.2048
        # bits/profile against a 157-level model's 0.5024 for the same number here.
        "cr_treshold": 10000.0,
        # Bits per level in that reference. None = the training tensor's own dtype,
        # which is what every run so far used (float32 -> 32). test_metrics.py builds
        # cr_native against a float32 reference, so on a float64 run set this to 32
        # or the loss will target twice the budget the metric reports.
        "rate_reference_bits_per_level": None,
        "recon_treshold": None,   #In RMSE will be convert to MSE 
        "lmbda": float(sys.argv[8]),
        "use_smoothl1": False,
        #"auto_normalize": False,
        "use_factor_weights": True,
        "factor_warmup_epochs": 150,  # Only used when method="factor"
    }


#        "lambda_deriv": 0.1,
#        "lambda_extrema": 0.5

   
    cfg = model_config()
    cfg["N"] = 64 #int(os.environ.get("N", "64"))  # 32 / 64 / 96 (192, 640 seen before)
    cfg["M"] = 96 #int(os.environ.get("M", "96"))  # 48 / 96 / 160
    cfg["slice_num"] = 3 #SLICES  # 3 / 6 / 10
    assert cfg["M"] % (16 * cfg["slice_num"]) == 0, "M must be a multiple of 16 x slice_num (global inter-context heads)"
    cfg["context_window"] = 5
    cfg['act'] = torch.nn.GELU
    cfg["enable_channel_context"] = True
    cfg["enable_local_context"] = True
    cfg["enable_global_inter_context"] = True
    cfg["enable_global_intra_context"] = True

    cfg["add_seasons"] = {"use":False, "mode":"embed"}
    cfg["add_sst"] = False

    # Smoother off for chapter 2 (vanilla codec, Checkpoint B 7 Oct); it returns with
    # the gradient multi-task losses of chapter 3. SMOOTHER=1 reproduces PF1 / PF2.
    cfg['output_low_band_filter_use'] = os.environ.get("SMOOTHER", "0") == "1"
    cfg['output_low_band_filter_mode'] = "learnable_gauss"  # "iir", "learnable_fir" , #learnable_dogs , learnable_gauss #zero_phase_lfilt

    rgb = {"use":False, "method":"depth_layers"} #PCA
    chn = "3" if rgb["use"] else "157"  # Number of channels
    dtype_str = "float32"
    # if rgb["use"] and rgb["method"] == "CAE":
    #     cae_ckpt_path = "/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/CAE/CAE/channels_[5000, 3000, 1000, 3]/upsample_mode_trilinear/linear_layer_False/cr_100000/1_conv_per_layer/padding_cubic/interp_size_5/final_upsample_upsample/act_fn_LeakyRelu/use_final_act_fn_True/lr_0.001/normalization_mean_std_along_depth/manage_nan_supress_with_max_depth/n_profiles_None/2025-03-04_06-01/checkpoints/val_loss=0.01-epoch=970.ckpt"
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     rgb_model = load_model(cae_ckpt_path, device=device)
    #     rgb_model.eval()
    #     rgb["model"] = rgb_model



    load_datamodule = False
    save_dm = False
    norm_method = "mean_std_along_depth" #"min_max_along_depth" #"min_max" #"mean_std_along_depth" #mean_std"
    dm_type = "enatl_natl" #"enatl_natl" #"good_split"
    data_name = "enatl_natl" #enatl_natl", natl, natl_sst
    data_name_dir = data_name if dm_type=="good_split" else ""
   


    dm_path = f"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/{data_name}_dm_157_192_256_norm_per_split_alternate_days_7_60_10.pkl"
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
                test_norm="on_test",  # "on_test" or "on_train"
                manage_nan="supress_with_max_depth",
                reshape=["factor_64"],
                days_split={"method":  "alternate_days", "value": (7,60), "n_gap": 10},
                filtering=False,
                depth_grid="native", #"native" uniform #equidistributed
                n_depth_levels=None, #157 #96 #64
                #uniform_z=False,
                shuffle=True,
                rgb=rgb,
                dtype_str=dtype_str,
                normalize_per_split=True
            )
            #days_split={"method":"ratio", "value": (0.03,0.97), "n_gap": 10}
            #days_split={"method": "alternate_days", "value": (7,60) , "n_gap": 10}, 




        else:
            datamodule = AEDatamodule(
                data_name=data_name,
                dl_kw={"batch_size": int(sys.argv[18]), "num_workers": int(sys.argv[12])},
                norm_stats={"method": norm_method},
                manage_nan="supress_with_max_depth",
                reshape={"factor_64": True, "spatial_crop":0},
                normalize_per_split=True,
                rgb=rgb,
                dtype_str=dtype_str,
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

    if loss_params['recon_treshold'] is not None:
        tresh = loss_params['recon_treshold']
        train_norm = datamodule.train_ds.input.attrs.get("norm_stats", None)
        train_norm_method = train_norm["method"]
        
        if train_norm_method == "min_max":
            x_min = train_norm["params"]["x_min"]
            x_max = train_norm["params"]["x_max"]  
            tresh = tresh / (x_max - x_min)
        elif train_norm_method == "mean_std" or train_norm_method == "mean_std_along_depth":
            mean = train_norm['params']["mean"]
            std = train_norm['params']["std"]
            tresh = tresh / std


        loss_params['recon_treshold'] = tresh



    cfg["in_channels"] = datamodule.test_shape[1] # e.g., 3 for RGB, 157 for your current data

    if cfg["add_seasons"]["use"]==True and cfg["add_seasons"]["mode"]=="embed":
        cfg["in_channels"] += 8  # Add one channel for season embedding
    elif cfg["add_seasons"]["use"]==True and cfg["add_seasons"]["mode"]=="one_hot":
        cfg["in_channels"] += 4  # Add four channels for one-hot season encoding

    if cfg["add_sst"]:
        cfg["in_channels"] += 1  # Add one channel for SST

    # Save experiment configuration to log file
    # Claim the run directory atomically. Two jobs started in the same second used
    # to share one directory: the second overwrote the first's config logs, then
    # crashed on makedirs(checkpoints) (E1b seed 666, job 55350, 24 Sep). mkdir is
    # atomic, so on a collision step the timestamp forward a second and retry.
    run_parent = f"/Odyssey/private/o23gauvr/code/MLIC/experiments/{sys.argv[2]}/{loss_params['method']}_loss_{cfg['N']}_{cfg['M']}_{float(sys.argv[8])}_CR_{loss_params['cr_treshold']}_{dm_type}_{data_name_dir}_{norm_method}"
    os.makedirs(run_parent, exist_ok=True)
    start = datetime.now()
    for offset in range(3600):
        timestamp = (start + timedelta(seconds=offset)).strftime("%Y%m%d_%H%M%S")
        checkpoint_dir_name = f"{run_parent}/{timestamp}"
        try:
            os.makedirs(checkpoint_dir_name, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"could not claim a free run directory under {run_parent}")
    
    config_log_path = os.path.join(checkpoint_dir_name, "experiment_config.log")
    with open(config_log_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"EXPERIMENT LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment name: {sys.argv[2]}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Datamodule type: {dm_type}\n")
        f.write(f"Data name: {data_name}\n")
        f.write(f"Load datamodule: {load_datamodule}\n")
        f.write(f"Datamodule path: {dm_path}\n")
        f.write(f"Norm method: {norm_method}\n")
        f.write("DATAMODULE PARAMETERS:\n")
        f.write("-" * 23 + "\n")
        for key, value in datamodule.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        
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
        _sig_idx = resolve_significant_depth_idx(datamodule.depth_array)
        f.write(f"significant_depth_m: {SIGNIFICANT_DEPTH_M}\n")
        f.write(f"significant_depth_idx: {_sig_idx} (z = {datamodule.depth_array[_sig_idx]:.1f} m)\n")
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

        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"GPU count: {torch.cuda.device_count()}\n")

    print(f"Experiment configuration saved to: {config_log_path}")

    main(datamodule,cfg,loss_params,checkpoint_dir_name)
