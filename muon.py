"""Muon optimizer — Newton-Schulz orthogonalization of gradients.

Based on DeepSeek V4's adoption of the Muon optimizer (from Keller Jordan's work),
which applies Newton-Schulz iterations to orthogonalize the momentum buffer before
taking a step. This normalizes all singular values toward 1, producing faster
convergence and better training stability.

Only applies to 2D+ weight matrices. 1D params (biases, norms) and embeddings use AdamW.
"""

import torch
from torch.optim import Optimizer


@torch.no_grad()
def _newton_schulz_iter(G: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Approximately orthogonalize G via Newton-Schulz iterations.

    Uses the quintic polynomial variant with pre-tuned coefficients for
    rapid convergence in ~5 iterations.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    rows, cols = G.shape
    if rows > cols:
        G = G.T
        transposed = True
    else:
        transposed = False

    # Normalize to unit spectral norm for convergence
    G = G / (G.norm() + 1e-7)
    for _ in range(n_iters):
        A = G @ G.T
        B = b * A + c * A @ A
        G = a * G + B @ G

    if transposed:
        G = G.T
    return G


class Muon(Optimizer):
    """Muon optimizer: orthogonalized momentum updates for weight matrices.

    Key details (matching reference implementation):
    1. Momentum is accumulated first, then Newton-Schulz orthogonalizes the buffer
    2. Shape-aware LR scaling: update scaled by sqrt(max(rows, cols) / min(rows, cols))
    3. Nesterov-style momentum for the orthogonalized step
    4. Weight decay is decoupled (applied before step)

    For non-2D parameters (biases, norms, embeddings): uses standard AdamW.

    Args:
        muon_params: iterable of 2D+ weight parameters for Muon
        adamw_params: iterable of parameters for AdamW fallback
        lr: learning rate for muon params (default: 0.02)
        momentum: momentum coefficient (default: 0.95)
        adamw_lr: learning rate for adamw params (default: lr * 0.1)
        adamw_betas: betas for adamw params (default: (0.9, 0.99))
        weight_decay: weight decay coefficient (default: 0.0)
        ns_iters: number of Newton-Schulz iterations (default: 5)
        nesterov: use Nesterov momentum (default: True)
    """
    def __init__(self, muon_params, adamw_params=None, lr=0.02, momentum=0.95,
                 adamw_lr=None, adamw_betas=(0.9, 0.99), weight_decay=0.0,
                 ns_iters=5, nesterov=True):
        if adamw_lr is None:
            adamw_lr = lr * 0.1

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_iters=ns_iters, nesterov=nesterov)

        muon_params = list(muon_params)
        param_groups = [
            {'params': muon_params, 'lr': lr, 'momentum': momentum,
             'weight_decay': weight_decay, 'ns_iters': ns_iters,
             'nesterov': nesterov, 'is_muon': True},
        ]

        if adamw_params is not None:
            adamw_params = list(adamw_params)
            param_groups.append(
                {'params': adamw_params, 'lr': adamw_lr, 'betas': adamw_betas,
                 'weight_decay': weight_decay, 'is_muon': False}
            )

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get('is_muon', True):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        lr = group['lr']
        momentum = group['momentum']
        wd = group['weight_decay']
        ns_iters = group['ns_iters']
        nesterov = group.get('nesterov', True)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(p)

            buf = state['momentum_buffer']

            # Weight decay (decoupled, applied before step)
            if wd != 0:
                p.mul_(1 - lr * wd)

            # Update momentum buffer: buf = momentum * buf + grad
            buf.mul_(momentum).add_(grad)

            # Nesterov lookahead: use grad + momentum * buf as the update input
            if nesterov:
                update = grad.add(buf, alpha=momentum)
            else:
                update = buf.clone()

            # Orthogonalize the update via Newton-Schulz (in fp32 for stability)
            if update.ndim >= 2:
                original_shape = update.shape
                u2d = update.reshape(update.shape[0], -1).float()
                u2d = _newton_schulz_iter(u2d, ns_iters)
                update = u2d.to(dtype=p.dtype).reshape(original_shape)

                # Shape-aware LR scaling
                rows, cols = u2d.shape
                scale = (max(rows, cols) / min(rows, cols)) ** 0.5
            else:
                scale = 1.0

            # Step
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group):
        lr = group['lr']
        beta1, beta2 = group.get('betas', (0.9, 0.99))
        wd = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # Weight decay
            if wd != 0:
                p.mul_(1 - lr * wd)

            # Adam update
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            step = state['step']
            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step

            denom = (exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(1e-8)
            step_size = lr / bc1

            p.addcdiv_(exp_avg, denom, value=-step_size)
