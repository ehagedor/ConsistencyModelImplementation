import numpy as np
import torch
import torch.nn.functional as F
from piq import LPIPS


sigma_max = 80.0
sigma_min = 0.002
rho = 7.0
distillation = True
sigma_data: float = 0.5




def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_scalings_for_boundary_condition( sigma):
        c_skip = sigma_data**2 / (
            (sigma - sigma_min) ** 2 + sigma_data**2
        )
        c_out = (
            (sigma - sigma_min)
            * sigma_data
            / (sigma**2 + sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in


def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
  
def consistency_loss(model, x_start, num_scales, model_kwargs, target_model, teacher_model, teacher_diffusion, noise, loss_norm):
    
    print('a')
    def denoise(model, x_t, sigmas, **model_kwargs):
        import torch.distributed as dist

        #if not distillation:
        #    c_skip, c_out, c_in = [
        #        append_dims(x, x_t.ndim) for x in get_scalings(sigmas)
        #    ]
        #else:
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim)
            for x in get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        
        x_hat = c_in * x_t
        #print(x_hat.shape)
        #print(sigmas.shape)
        #print(rescaled_t.shape)
        model_output = model(x=x_hat, sigma=rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised
    
    
    if model_kwargs is None:
        model_kwargs = {}
    if noise is None:
        noise = torch.rand_like(x_start)
    if loss_norm == "lpips":
            lpips_loss = LPIPS(replace_pooling=True, reduction="none")
    dims = x_start.ndim
    
    def denoise_fn(x, t):
            return denoise(model, x, t, **model_kwargs)[1]

    if target_model:

        @torch.no_grad()
        def target_denoise_fn(x, t):
            
            return denoise(target_model, x, t, **model_kwargs)[1]

    else:
        raise NotImplementedError("Must have a target model")
    
    if teacher_model:

            @torch.no_grad()
            def teacher_denoise_fn(x, t):
                return denoise(teacher_model, x, t, **model_kwargs)[1]
            
    def euler_solver(samples, t, next_t, x0):
        x = samples
        denoiser = teacher_denoise_fn(x, t)
        d = (x - denoiser) / append_dims(t, dims)
        samples = x + d * append_dims(next_t - t, dims)
        
        return samples
    
    indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )
    
    t = sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        )
    t = t**rho
    
    t2 = sigma_max ** (1 / rho) + (indices + 1) / (num_scales - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        )
    t2 = t2**rho
    
    x_t = x_start + noise * append_dims(t, dims)
    
    dropout_state = torch.get_rng_state()
    distiller = denoise_fn(x_t, t)
    
    x_t2 = euler_solver(x_t, t, t2, x_start).detach()
    
    torch.set_rng_state(dropout_state)
    distiller_target = target_denoise_fn(x_t2, t2)
    distiller_target = distiller_target.detach()
    
    
    if loss_norm == "l1":
        diffs = torch.abs(distiller - distiller_target)
        loss = mean_flat(diffs)
    elif loss_norm == "l2":
        diffs = (distiller - distiller_target) ** 2
        loss = mean_flat(diffs)
    elif loss_norm == "l2-32":
        distiller = F.interpolate(distiller, size=32, mode="bilinear")
        distiller_target = F.interpolate(
            distiller_target,
            size=32,
            mode="bilinear",
        )
        diffs = (distiller - distiller_target) ** 2
        loss = mean_flat(diffs) 
    elif loss_norm == "lpips":
        if x_start.shape[-1] < 256:
            distiller = F.interpolate(distiller, size=224, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target, size=224, mode="bilinear"
            )

        loss = (
            lpips_loss(
                (distiller + 1) / 2.0,
                (distiller_target + 1) / 2.0,
            )
            
        )
        terms = {}
        terms["loss"] = loss

        return terms
    
    
        
        
        
        
        
        
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
