import torch
import torch.nn as nn
import torch.nn.functional as F


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, loss_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, gradient_accumulation_steps=1
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        # Only zero gradients at the beginning of accumulation
        if i % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            loss_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        
        # Scale loss by accumulation steps
        loss = out_criterion["loss"] / gradient_accumulation_steps
        loss.backward()
        
        # Only step optimizer after accumulating gradients
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            # Handle auxiliary loss
            aux_loss = model.aux_loss() / gradient_accumulation_steps
            aux_loss.backward()
            aux_optimizer.step()
            loss_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            for key, value in out_criterion.items():
                tb_logger.add_scalar('{}'.format(f'[train]: {key}'), value.item(), current_step)

            # Log original (unscaled) loss values for TensorBoard
            # tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
            # tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: aux_loss'), model.aux_loss().item(), current_step)
            

        if i % 100 == 0:
            criterion_log = " | ".join(
                f"{k}: {v.item():.4f}" if torch.is_tensor(v) else f"{k}: {v}"
                for k, v in out_criterion.items()
            )

            logger_train.info(
                f"Train epoch {epoch}: "
                f"[{i*len(d)}/{len(train_dataloader.dataset)} "
                f"({100. * i / len(train_dataloader):.0f}%)]\t"
                f"{criterion_log} | Aux loss: {model.aux_loss().item():.4f}"
            )

    return current_step


def warmup_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, loss_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, lr_scheduler
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        loss_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        if epoch < 1:
            lr_scheduler.step()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        loss_optimizer.step()

        current_step += 1
        if current_step % 100 == 0:
            for key, value in out_criterion.items():
                tb_logger.add_scalar('{}'.format(f'[train]: {key}'), value.item(), current_step)

            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)


        if i % 100 == 0:
            criterion_log = " | ".join(
                f"{k}: {v.item():.4f}" if torch.is_tensor(v) else f"{k}: {v}"
                for k, v in out_criterion.items()
            )

            logger_train.info(
                f"Train epoch {epoch}: "
                f"[{i*len(d)}/{len(train_dataloader.dataset)} "
                f"({100. * i / len(train_dataloader):.0f}%)]\t"
                f"{criterion_log} | Aux loss: {model.aux_loss().item():.4f}"
            )


    return current_step
