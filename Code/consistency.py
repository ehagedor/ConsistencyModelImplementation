import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from numpy.random import Generator
from model import EDMPrecond
from model import DhariwalUNet
import pickle
import copy
from tqdm import tqdm
import logging
import diffusion as D
import torch.nn as nn
from torch.optim import RAdam
from ema import create_ema_and_scales_fn
from parameters import create_argparser, model_and_diffusion_defaults, model_defaults
import numpy as np


diffusion_steps = 1000
epsilon = 0.002
ro = 7
steps = 18
net1 = EDMPrecond(32,3)


def update_ema(ema_rate, ema_params, model_params):
        for rate, params in zip(ema_rate, ema_params):
            update_ema(params, model_params, rate=rate)

def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def distillation(filepath):
    with open(filepath, "wb") as f:
        net = pickle.load(f)['ema'].to(device='GPU')
    return net



def make_models(img_res, c_in, model_kwargs):
    model = EDMPrecond(img_resolution=img_res, img_channels=c_in)#, model_kwargs=model_kwargs)
    #teacher_kwargs = copy.deepcopy(img_res, c_in)#, model_kwargs=model_kwargs)
    train_model = EDMPrecond(img_resolution=img_res, img_channels=c_in)#, model_kwargs=model_kwargs)
    teacher_model = EDMPrecond(img_resolution=img_res, img_channels=c_in)
    teacher_diffusion = None
    return train_model, teacher_model, teacher_diffusion, model
    
    

def train():
    args = create_argparser().parse_args()
    torch.cuda.empty_cache()
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset= torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=True, transform=transform)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=8)
    
    train_model, teacher_model, teacher_diffusion, model = make_models(model_defaults()["img_resolution"], model_defaults()["in_channels"], model_defaults())
    with open(args.teacher_model_dir, "rb") as f:
        teacher_model = pickle.load(f)['ema'].to(device='cuda')
    teacher_model.eval()
    train_model.requires_grad_(False)
    train_model.to(device='cuda')
    train_model.requires_grad_(False)
    train_model.train()
    target_model_param_groups_and_shapes = get_param_groups_and_shapes(train_model.named_parameters())
            
    target_model_master_params = make_master_params(target_model_param_groups_and_shapes)
    model.to(device='cuda')
    model.train()
    ema_rate = (
            float(x) for x in args.ema_rate.split(",")
        )
    ema_params = [
                copy.deepcopy(list(model.parameters()))
                for _ in range(len(args.ema_rate))
            ]
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    opt = RAdam(
                    list(model.parameters()),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
    
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}:")
        print(epoch % args.save_interval)
        if epoch % args.save_interval == 0:
            save(train_model, opt, ema_rate, ema_params, epoch, args) 
        pbar = tqdm(dataloader)
        #print(epoch % args.save_interval)
        
        for i, data in enumerate(pbar):
            images, labels = data
            
            images = images.to('cuda')
            
            #ema fn here
            ema, num_scales = ema_scale_fn(i)
            #loss
            losses = D.consistency_loss(
                model=model, x_start=images, num_scales=num_scales, model_kwargs=None, 
                target_model=train_model, teacher_model=teacher_model, teacher_diffusion=teacher_diffusion, noise=None, loss_norm=args.loss_norm)
            loss = losses["loss"].mean()
            loss.backward()
            opt.step()
            update_ema(ema_rate=ema_rate, ema_params=ema_params, master_params=list(model.parameters()))
            
            
            
            
            
            update_target_ema(ema_scale_fn=ema_scale_fn, 
                              target_model_master_params=target_model_master_params, 
                              master_params=list(model.parameters()),
                              target_model_param_groups_and_shapes=target_model_param_groups_and_shapes, 
                              step=i*(epoch+1),
                              target_model_master_params1=list(train_model.parameters())
                              )
          
            
def save(model, opt, ema_rate, ema_params, epoch, args):
    import blobfile as bf

    step = epoch

    def save_checkpoint(rate, params):
        state_dict = master_params_to_state_dict(params, model)
        print(f"saving model {rate}...")
        if not rate:
            filename = f"model{step:06d}.pt"
        else:
            filename = f"ema_{rate}_{step:06d}.pt"
        with bf.BlobFile(bf.join(args.save_dir, filename), "wb") as f:
            torch.save(state_dict, f)

    for rate, params in zip(ema_rate, ema_params):
            save_checkpoint(rate, params)

    print("saving optimizer state...")
    with bf.BlobFile(
        bf.join(args.save_dir, f"opt{step:06d}.pt"),
        "wb",
    ) as f:
        torch.save(opt.state_dict(), f)

    if model:
        print("saving target model state")
        filename = f"target_model{step:06d}.pt"
        with bf.BlobFile(bf.join(args.save_dir, filename), "wb") as f:
            torch.save(model.state_dict(), f)

def master_params_to_state_dict(master_params, model):
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = master_params[i]
    return state_dict
   
def update_target_ema(ema_scale_fn, target_model_master_params, master_params, target_model_param_groups_and_shapes, step, target_model_master_params1):
        target_ema, scales = ema_scale_fn(step)
        with torch.no_grad():
            update_ema1(
                target_params=target_model_master_params1,
                source_params=master_params,
                rate=target_ema,
            )
            master_params_to_model_params(
                target_model_param_groups_and_shapes,
                target_model_master_params[0],
            ) 
 
def update_ema(ema_rate, ema_params, master_params):
    for rate, params in zip(ema_rate, ema_params):
        update_ema1(params, master_params, rate=rate) 
 
def update_ema1(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    i=0
    for targ, src in zip(target_params, source_params):
        
        
        
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)   
    

def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]

def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    normal_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        
        master_param.requires_grad = True
        for (_, param) in param_group:
            norm_param = nn.Parameter(param).view(shape)
            #norm_param.requires_grad = True
            normal_params.append(norm_param)
        master_params.append(master_param)
    return [master_params, normal_params]

def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)

def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])

def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    return torch._C._nn.flatten_dense_tensors(tensors)

if __name__ == "__main__":
    train()

