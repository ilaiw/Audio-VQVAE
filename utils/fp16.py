# Utils for fp16 training.
import math
import torch
from torch.optim import Optimizer


def adam_step(p: torch.Tensor, out_p: torch.Tensor, exp_avg: torch.Tensor,
              exp_avg_sq: torch.Tensor, grad: torch.Tensor, lr: float,
              beta1: float, beta2: float, eps: float, scale: float, step: int,
              eps_mode: int, bias_correction: int, weight_decay: float):
    assert bias_correction == 1
    assert eps_mode == 1

    grad = grad.float()
    grad.div_(scale)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = exp_avg_sq.sqrt().add_(eps)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

    p.add_(exp_avg/denom + weight_decay*p.float(), alpha=-step_size)


# Automatic loss scaling
class LossScalar(object):
    def __init__(self,
                 loss_scale,
                 init_scale=2. ** 16,
                 scale_factor=2. ** (1. / 1000),
                 scale_window=1):
        if loss_scale is None:
            # Use dynamic loss scaling
            self.dynamic = True
            self.loss_scale = init_scale
        else:
            self.dynamic = False
            self.loss_scale = loss_scale
        self.max_loss_scale = 2.**24
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.unskipped = 0
        self.overflow = False

    def get_scale(self):
        return self.loss_scale

    def update_scale(self, overflow):
        if overflow and self.dynamic:
            self.loss_scale /= 2.
            self.unskipped = 0
        else:
            self.unskipped += 1

        if self.unskipped == self.scale_window and self.dynamic:
            self.loss_scale = min(self.max_loss_scale, self.loss_scale * self.scale_factor)
            self.unskipped = 0


def check_overflow(val):
    return (val == float('inf')) or (val == -float('inf')) or (val != val)


def grad_norm(params, scale):
    params = list(params)
    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += p.grad.norm(p=2, dtype=torch.float32)**2
    grad_norm = float(grad_norm**0.5)
    return grad_norm / scale


def clipped_grad_scale(grad_norm, max_grad_norm, scale):
    clip = grad_norm / max_grad_norm
    if clip > 1:
        scale = clip * scale
    return scale


class FusedAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    def step(self, closure=None, scale=1.0):
        """Performs a single optimization step. Scales gradients down by scale
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = torch.tensor([], dtype=torch.float)
                adam_step(
                    p.data,
                    out_p,
                    exp_avg,
                    exp_avg_sq,
                    grad,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    scale,
                    state["step"],
                    self.eps_mode,
                    bias_correction,
                    group["weight_decay"],
                )

        return loss
