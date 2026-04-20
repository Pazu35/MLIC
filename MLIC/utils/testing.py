import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MLIC.MLIC.utils.metrics import compute_psnr, get_f1_score
from MLIC.MLIC.utils.utils import *


def test_one_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger, validation=True):
    model.eval()
    device = next(model.parameters()).device

    stage="Validation" if validation else "Test"

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    deriv_loss = AverageMeter()
    f1_score = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    rmse_loss = AverageMeter()
    ecs_loss = AverageMeter()

    rgb_psnr = AverageMeter()
    rgb_rmse_loss = AverageMeter() 

    criterion_losses = {}

    
    rgb_model = test_dataloader.dataset.input.attrs.get("rgb_model", None)
    depth_array = test_dataloader.dataset.input.z.values.astype(test_dataloader.dataset.input.dtype)
    
    norm_stats = test_dataloader.dataset.input.attrs["norm_stats"]
    season_idx_list = test_dataloader.dataset.input.attrs.get("season_idx", None)
    sst_full = test_dataloader.dataset.input.attrs.get("sst", None)

    

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            season_idx = season_idx_list[i * d.size(0):(i + 1) * d.size(0)] if season_idx_list is not None else None
            sst = sst_full[i * d.size(0):(i + 1) * d.size(0), :] if sst_full is not None else None

            out_net = model(d, season_idx=season_idx, sst=sst)

            if norm_stats["method"] == "min_max":
                d_min = norm_stats["params"]["x_min"]
                d_max = norm_stats["params"]["x_max"]
                out_net['x_hat'] = out_net['x_hat'] * (d_max - d_min) + d_min
                d = d * (d_max - d_min) + d_min

            elif norm_stats["method"] == "mean_std":
                mean = norm_stats["params"]["mean"]
                std = norm_stats["params"]["std"]
                out_net['x_hat'] = (out_net['x_hat'] * std) + mean
                d = (d * std) + mean
            elif norm_stats["method"] == "mean_std_along_depth":
                mean = norm_stats["params"]["mean_along_depth"]
                std = norm_stats["params"]["std_along_depth"]
                mean = torch.from_numpy(mean).to(device=d.device, dtype=d.dtype)
                std = torch.from_numpy(std).to(device=d.device, dtype=d.dtype)
                out_net['x_hat'] = out_net['x_hat'] * std  + mean
                d = d * std + mean 

            if rgb_model is not None:
                out_net['x_hat'] = rgb_model.model_AE.decoder(out_net['x_hat'].unsqueeze(-1)).squeeze(-1)
                rgb_d = rgb_model.model_AE.decoder(d.unsqueeze(-1)).squeeze(-1)
                d = torch.from_numpy(test_dataloader.dataset.input.attrs['original_data'].isel(time=slice(i * d.size(0), (i + 1) * d.size(0))).transpose("time", "z", "lat", "lon").values).to(device=d.device, dtype=d.dtype)   ##shuffle==False
                
                if rgb_model.norm_stats['method'] == "mean_std_along_depth":
                    mean,std = rgb_model.norm_stats['params'].values()
                    mean = torch.from_numpy(mean).to(d.device)
                    std = torch.from_numpy(std).to(d.device)
                    rgb_d = (rgb_d*std) + mean
                    out_net['x_hat'] = (out_net['x_hat']*std) + mean

                    
                elif rgb_model.norm_stats['method'] == "mean_std":
                    mean,std = rgb_model.norm_stats['params'].values()
                    rgb_d = rgb_d*std + mean
                    out_net['x_hat'] = (out_net['x_hat']*std) + mean
                
                elif rgb_model.norm_stats['method'] == "min_max":
                    x_min,x_max = rgb_model.norm_stats['params']['x_min'],rgb_model.norm_stats['params']['x_max'] #rgb_model.norm_stats['params'].values()
                    rgb_d = rgb_d*(x_max - x_min) + x_min
                    out_net['x_hat'] = (out_net['x_hat']*(x_max-x_min)) + x_min


                psnr_rgb = compute_psnr(out_net['x_hat'], rgb_d)
                rgb_psnr.update(psnr_rgb)
                rmse_rgb = torch.sqrt(F.mse_loss(out_net['x_hat'], rgb_d))
                rgb_rmse_loss.update(rmse_rgb)

                tb_logger.add_scalar('{}'.format(f'[{stage}]: rgb_psnr'), rgb_psnr.avg, epoch + 1)
                tb_logger.add_scalar('{}'.format(f'[{stage}]: rgb_rmse_loss'), rgb_rmse_loss.avg, epoch + 1)




            # out_net['x_hat'] = out_net['x_hat'][:,-len(depth_array):,:]
            # d = d[:,-len(depth_array):,:]
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            for v in out_criterion.values():
                if torch.is_tensor(v) and v is not None:
                    loss.update(v)

            # rec = torch2img(out_net['x_hat'])
            # img = torch2img(d)
            # rec = out_net['x_hat'].clamp_(0, 1).squeeze().detach().cpu().numpy()
            # img = d.clamp_(0, 1).squeeze().detach().cpu().numpy()
            dp = out_net['x_hat'][:, 1:] - out_net['x_hat'][:, :-1]
            dt = d[:, 1:] - d[:, :-1]
            deriv_loss.update(F.mse_loss(dp, dt))
            psnr.update(compute_psnr(out_net['x_hat'], d))
            f1_score.update(get_f1_score(d.detach().cpu().numpy(), out_net['x_hat'].detach().cpu().numpy()))
            bpp_loss.update(out_criterion["bpp_loss"].item())

            rmse_loss.update(torch.sqrt(F.mse_loss(out_net['x_hat'], d)))

            ecs = np.abs(depth_array[torch.argmax(out_net['x_hat'],dim=1).detach().cpu()] - depth_array[torch.argmax(d,dim=1).detach().cpu()])
            ecs_loss.update(ecs.mean())

            for key, value in out_criterion.items():
                if value is not None and torch.is_tensor(value):
                    if key not in criterion_losses:
                        criterion_losses[key] = AverageMeter()
                    criterion_losses[key].update(value.item())


            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            # img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format(f'[{stage}]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(f'[{stage}]: f1_score'), f1_score.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(f'[{stage}]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(f'[{stage}]: deriv_loss'), deriv_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(f'[{stage}]: aux_loss'), aux_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(f'[{stage}]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(f'[{stage}]: rmse_loss'), rmse_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format(f'[{stage}]: ecs_loss'), ecs_loss.avg, epoch + 1)

    for key, meter in criterion_losses.items():
        tb_logger.add_scalar(f'[{stage}]: {key}', meter.avg, epoch + 1)


    if "loss_weights" in dir(criterion):
        if vars(criterion).get('use_factor_weights', False):
            w = criterion.active_weights
        else:
            w = criterion.loss_weights
        for k, v in w.items():
            tb_logger.add_scalar('{}'.format(f'[{stage}]: {k}_loss weight'), v, epoch + 1)
    


    loss_dict = {"loss": loss.avg, "bpp_loss": bpp_loss.avg, "deriv_loss": deriv_loss.avg, "f1_score": f1_score.avg, "psnr": psnr.avg, "rmse_loss": rmse_loss.avg, "ecs_loss": ecs_loss.avg, "rgb_psnr": rgb_psnr.avg, "rgb_rmse_loss": rgb_rmse_loss.avg}


    # for key, value in out_criterion.items():
    #     tb_logger.add_scalar('{}'.format(ff'[{stage}]: {key}'), value.avg, epoch + 1)



    criterion_log = " | ".join(
    f"{k}: {v.item():.4f}" if torch.is_tensor(v) else f"{k}: {v}"
    for k, v in out_criterion.items()
    )

    logger_val.info(
        f"{stage} epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"Deriv loss: {deriv_loss.avg:.6f} | "
        f"F1 Score: {f1_score.avg:.6f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"RMSE: {rmse_loss.avg:.6f} | "
        f"ECS: {ecs_loss.avg:.6f} | "
        f"{criterion_log} "
    )

    # if out_criterion["mse_loss"] is not None:
    #     logger_val.info(
    #         f"Test epoch {epoch}: Average losses: "
    #         f"Loss: {loss.avg:.4f} | "
    #         f"MSE loss: {mse_loss.avg:.6f} | "
    #         f"Bpp loss: {bpp_loss.avg:.4f} | "
    #         f"Aux loss: {aux_loss.avg:.2f} | "
    #         f"PSNR: {psnr.avg:.6f} | "
    #         f"MS-SSIM: {ms_ssim.avg:.6f}"
    #     )
    #     tb_logger.add_scalar('{}'.format(f'[{stage}]: mse_loss'), mse_loss.avg, epoch + 1)
    # if out_criterion["ms_ssim_loss"] is not None:
    #     logger_val.info(
    #         f"Test epoch {epoch}: Average losses: "
    #         f"Loss: {loss.avg:.4f} | "
    #         f"MS-SSIM loss: {ms_ssim_loss.avg:.6f} | "
    #         f"Bpp loss: {bpp_loss.avg:.4f} | "
    #         f"Aux loss: {aux_loss.avg:.2f} | "
    #         f"PSNR: {psnr.avg:.6f} | "
    #         f"MS-SSIM: {ms_ssim.avg:.6f}"
    #     )
    #     tb_logger.add_scalar('{}'.format(f'[{stage}]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)

    return loss_dict

def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp, out["cost_time"]


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    return x_hat, cost_time



# def test_model(test_dataloader, net, logger_test, save_dir, epoch):
#     net.eval()
#     device = next(net.parameters()).device

#     avg_psnr = AverageMeter()
#     avg_ms_ssim = AverageMeter()
#     avg_bpp = AverageMeter()
#     avg_enc_time = AverageMeter()
#     avg_dec_time = AverageMeter()

#     with torch.no_grad():
#         for i, img in enumerate(test_dataloader):
#             img = img.to(device)
#             B, C, H, W = img.shape
#             pad_h = 0
#             pad_w = 0
#             if H % 64 != 0:
#                 pad_h = 64 * (H // 64 + 1) - H
#             if W % 64 != 0:
#                 pad_w = 64 * (W // 64 + 1) - W
#             img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
#             # warmup GPU
#             if i == 0:
#                 bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
#             # avoid resolution leakage
#             net.update_resolutions(16, 16)
#             bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
#             # avoid resolution leakage
#             net.update_resolutions(16, 16)
#             x_hat, dec_time = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i))
#             rec = torch2img(x_hat)
#             img = torch2img(img)
#             img.save(os.path.join(save_dir, '%03d_gt.png' % i))
#             rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
#             p, m = compute_metrics(rec, img)
#             avg_psnr.update(p)
#             avg_ms_ssim.update(m)
#             avg_bpp.update(bpp)
#             avg_enc_time.update(enc_time)
#             avg_dec_time.update(dec_time)
#             logger_test.info(
#                 f"Image[{i}] | "
#                 f"Bpp loss: {bpp:.2f} | "
#                 f"PSNR: {p:.4f} | "
#                 f"MS-SSIM: {m:.4f} | "
#                 f"Encoding Latency: {enc_time:.4f} | "
#                 f"Decoding Latency: {dec_time:.4f}"
#             )
#     logger_test.info(
#         f"Epoch:[{epoch}] | "
#         f"Avg Bpp: {avg_bpp.avg:.4f} | "
#         f"Avg PSNR: {avg_psnr.avg:.4f} | "
#         f"Avg MS-SSIM: {avg_ms_ssim.avg:.4f} | "
#         f"Avg Encoding Latency:: {avg_enc_time.avg:.4f} | "
#         f"Avg decoding Latency:: {avg_dec_time.avg:.4f}"
#     )
