import torch
import torch.nn as nn
import torch.optim as optim


def configure_optimizers(net, criterion, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = [
        p for n, p in net.named_parameters() if not n.endswith(".quantiles")
    ]
    aux_parameters = [
        p for n, p in net.named_parameters() if n.endswith(".quantiles")
    ]

    #loss_parameters = [criterion.log_vars]
    #loss_parameters = [

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = set(parameters) & set(aux_parameters)
    union_params = set(parameters) | set(aux_parameters)

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (p for p in parameters if p.requires_grad),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (p for p in aux_parameters if p.requires_grad),
        lr=args.aux_learning_rate,
    )

    if len (list(criterion.parameters())) > 0:
        print("Configuring loss optimizer")
        loss_optimizer = optim.Adam(
            (p for p in criterion.parameters() if p.requires_grad),
            lr=args.loss_learning_rate,
        )
    else:
        loss_optimizer = None

    return optimizer, aux_optimizer, loss_optimizer
