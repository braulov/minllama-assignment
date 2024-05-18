from typing import Callable, Iterable, Tuple
import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                state["step"] += 1
                # Access hyperparameters from the `group` dictionary
                alpha, w, eps = group["lr"], group["weight_decay"], group["eps"]
                m, v, step = state["m"], state["v"], state["step"]
                # Update first and second moments of the gradients
                grad = grad + w * p.data
                new_grad = grad
                betas = group["betas"]
                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                bias_correction1 = 1 - (betas[0] ** step)
                bias_correction2 = 1 - (betas[1] ** step)
                m = betas[0] * m + (1 - betas[0]) * new_grad
                v = betas[1] * v + (1 - betas[1]) * torch.square(new_grad)
                state["m"], state["v"] = m, v
                # Update parameters
                x = m * alpha * math.sqrt(bias_correction2) / bias_correction1
                y = torch.sqrt(v) + eps * math.sqrt(bias_correction2)
                c = torch.div(m * alpha * math.sqrt(bias_correction2) / bias_correction1, torch.sqrt(v) + eps * math.sqrt(bias_correction2))
                p.data -= torch.div(m * alpha * math.sqrt(bias_correction2) / bias_correction1, torch.sqrt(v) + eps * math.sqrt(bias_correction2))

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data += p.data * w * alpha
        return loss
