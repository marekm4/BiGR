import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
import pdb
from torch import nn
import math

def binary_cross_entropy(pred, y): 
    return -(pred.log()*y + (1-y)*(1-pred).log())

class BinaryDiffusion(nn.Module):
    def __init__(self, denoise_fn, p_flip, n_repeat=1, aux=0.0, focal=0.0, alpha=-1, n_sample_steps=256):
        super().__init__()

        self.num_timesteps = n_sample_steps
        self._denoise_fn = denoise_fn
        self.loss_final = 'mean'
        self.use_softmax = False
        self.n_repeat = n_repeat # sample t n times
        
        self.scheduler = noise_scheduler(self.num_timesteps, beta_type='linear')
        self.p_flip = p_flip
        self.focal = focal
        self.aux = aux
        self.alpha = alpha
        
    def sample_time(self, b, device):
        t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
        return t

    def q_sample(self, x_0, t):
        x_t = self.scheduler(x_0, t) # t >= 1 <=T#
        return x_t

    def _train_loss(self, x_0, cond=None):
        # different timesteps for different patches (tokens)
        x_0 = x_0 * 1.0
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t = self.sample_time(b, device)

        # make x noisy and denoise
        x_t = self.q_sample(x_0, t)

        x_t_in = torch.bernoulli(x_t)*1.0
        
        x_0_hat_logits = self._denoise_fn(x_t_in, time_steps=t-1, c=cond)
        # if label is not None:
        #     if self.guidance and np.random.random() < 0.1:
        #         label = None
        #     x_0_hat_logits = self._denoise_fn(x_t_in, label=label, time_steps=t-1) 
        # else:
        #     x_0_hat_logits = self._denoise_fn(x_t_in, time_steps=t-1)
        
        if self.p_flip:
            if self.focal >= 0:
                x_0_ = torch.logical_xor(x_0, x_t_in)*1.0
                kl_loss = focal_loss(x_0_hat_logits, x_0_, alpha=self.alpha, gamma=self.focal)
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
            else:
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        else:
            if self.focal >= 0:
                kl_loss = focal_loss(x_0_hat_logits, x_0, alpha=0, gamma=self.focal)
            else:
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        if torch.isinf(kl_loss).max():
            pdb.set_trace()

        if self.loss_final == 'weighted':
            weight = (1 - ((t-1) / self.num_timesteps)).view(-1, 1)
        elif self.loss_final == 'mean':
            weight = 1.0
        else:
            raise NotImplementedError
        
        loss = (weight * kl_loss).mean()
        loss_all = weight * kl_loss
        
        kl_loss_all = kl_loss
        kl_loss = kl_loss.mean()
        
        with torch.no_grad():
            if self.use_softmax:
                acc = (((x_0_hat_logits[..., 1] > x_0_hat_logits[..., 0]) * 1.0 == x_0.view(-1)) * 1.0).sum() / float(x_0.numel())
            else:
                acc = (((x_0_hat_logits > 0.0) * 1.0 == x_0) * 1.0).sum() / float(x_0.numel())

        if self.aux > 0:
            # ftr = (((t-1)==0)*1.0).view(-1, 1, 1)
            ftr = (((t-1)==0)*1.0).view(-1, 1)

            x_0_l = torch.sigmoid(x_0_hat_logits)
            x_0_logits = torch.cat([x_0_l.unsqueeze(-1), (1-x_0_l).unsqueeze(-1)], dim=-1)

            x_t_logits = torch.cat([x_t_in.unsqueeze(-1), (1-x_t_in).unsqueeze(-1)], dim=-1)

            p_EV_qxtmin_x0 = self.scheduler(x_0_logits, t-1)
            
            q_one_step = self.scheduler.one_step(x_t_logits, t)
            unnormed_probs = p_EV_qxtmin_x0 * q_one_step
            unnormed_probs = unnormed_probs / (unnormed_probs.sum(-1, keepdims=True)+1e-6)
            unnormed_probs = unnormed_probs[...,0]
            
            x_tm1_logits = unnormed_probs * (1-ftr) + x_0_l * ftr
            x_0_gt = torch.cat([x_0.unsqueeze(-1), (1-x_0).unsqueeze(-1)], dim=-1)
            p_EV_qxtmin_x0_gt = self.scheduler(x_0_gt, t-1)
            unnormed_gt = p_EV_qxtmin_x0_gt * q_one_step
            unnormed_gt = unnormed_gt / (unnormed_gt.sum(-1, keepdims=True)+1e-6)
            unnormed_gt = unnormed_gt[...,0]

            x_tm1_gt = unnormed_gt

            if torch.isinf(x_tm1_logits).max() or torch.isnan(x_tm1_logits).max():
                pdb.set_trace()
            aux_loss = binary_cross_entropy(x_tm1_logits.clamp(min=1e-6, max=(1.0-1e-6)), x_tm1_gt.clamp(min=0.0, max=1.0))
            
            loss_all = self.aux * (weight * aux_loss) + loss_all
            
            aux_loss = (weight * aux_loss).mean()
            loss = self.aux * aux_loss + loss
            
        stats = {'loss': loss, 'bce_loss': kl_loss, 'loss_all': loss_all, 'bce_loss_all': kl_loss_all, 'acc': acc}

        if self.aux > 0:
            stats['aux loss'] = aux_loss
        return stats
    
    
    def sample(self, temp=1.0, sample_steps=None, return_all=False, cond=None, mask=None, cfg_scale=None, full=False, return_logits=False):
        device = cond.device
        
        cond_z, uncond_z = torch.split(cond, len(cond) // 2, dim=0)
        b = cond_z.shape[0]
        
        x_t = torch.bernoulli(0.5 * torch.ones((b, self._denoise_fn.in_dim), device=device))

        if mask is not None:
            m = mask['mask'].unsqueeze(0)
            latent = mask['latent'].unsqueeze(0)
            x_t = latent * m + x_t * (1-m)
        sampling_steps = np.array(range(1, self.num_timesteps+1))

        if sample_steps != self.num_timesteps:
            idx = np.linspace(0.0, 1.0, sample_steps)
            idx = np.array(idx * (self.num_timesteps-1), int)
            sampling_steps = sampling_steps[idx]

        if return_all:
            x_all = [x_t]
        
        sampling_steps = sampling_steps[::-1]

        for i, t in enumerate(sampling_steps):
            ####### CFG schedule #######
            real_cfg = cfg_scale
            ############################
            
            t = torch.full((b,), t, device=device, dtype=torch.long)
            
            x_0_logits = self._denoise_fn(x_t, time_steps=t-1, c=cond_z)
            x_0_logits = x_0_logits / temp
            
            x_0_logits_uncond = self._denoise_fn(x_t, time_steps=t-1, c=uncond_z)
            x_0_logits_uncond = x_0_logits_uncond / temp
            
            # x_0_logits = (1 + cfg_scale) * x_0_logits - cfg_scale * x_0_logits_uncond
            x_0_logits = x_0_logits_uncond + real_cfg * (x_0_logits - x_0_logits_uncond)
            
            x_0_logits = torch.sigmoid(x_0_logits)

            if self.p_flip:
                x_0_logits =  x_t * (1 - x_0_logits) + (1 - x_t) * x_0_logits
            
            if not t[0].item() == 1:
                t_p = torch.full((b,), sampling_steps[i+1], device=device, dtype=torch.long)
                
                x_0_logits = torch.cat([x_0_logits.unsqueeze(-1), (1-x_0_logits).unsqueeze(-1)], dim=-1)
                x_t_logits = torch.cat([x_t.unsqueeze(-1), (1-x_t).unsqueeze(-1)], dim=-1)

                p_EV_qxtmin_x0 = self.scheduler(x_0_logits, t_p)
                q_one_step = x_t_logits

                for mns in range(sampling_steps[i] - sampling_steps[i+1]):
                    q_one_step = self.scheduler.one_step(q_one_step, t - mns)
                
                unnormed_probs = p_EV_qxtmin_x0 * q_one_step
                unnormed_probs = unnormed_probs / unnormed_probs.sum(-1, keepdims=True)
                unnormed_probs = unnormed_probs[...,0]
                
                x_tm1_logits = unnormed_probs
                x_tm1_p = torch.bernoulli(x_tm1_logits)
            
            else:
                x_0_logits = x_0_logits
                x_tm1_p = (x_0_logits > 0.5) * 1.0

            x_t = x_tm1_p

            if mask is not None:
                m = mask['mask'].unsqueeze(0)
                latent = mask['latent'].unsqueeze(0)
                x_t = latent * m + x_t * (1-m)


            if return_all:
                x_all.append(x_t)
        
        if return_all:
            return torch.cat(x_all, 0), None
        else:
            if not return_logits:
                return x_t
            else:
                return x_0_logits
            
    def forward(self, x, cond=None):
        return self._train_loss(x, cond)


class noise_scheduler(nn.Module):
    def __init__(self, steps=40, beta_type='linear'):
        super().__init__()

        if beta_type == 'linear':

            beta = 1 - 1 / (steps - np.arange(1, steps+1) + 1) 

            k_final = [1.0]
            b_final = [0.0]

            for i in range(steps):
                k_final.append(k_final[-1]*beta[i])
                b_final.append(beta[i] * b_final[-1] + 0.5 * (1-beta[i]))

            k_final = k_final[1:]
            b_final = b_final[1:]


        elif beta_type == 'cos':

            k_final = np.linspace(0.0, 1.0, steps+1)

            k_final = k_final * np.pi
            k_final = 0.5 + 0.5 * np.cos(k_final)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]
        
        elif beta_type == 'sigmoid':
            
            def sigmoid(x):
                z = 1/(1 + np.exp(-x))
                return z

            def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=0.0):
                # A gamma function based on sigmoid function.
                v_start = sigmoid(start / tau)
                v_end = sigmoid(end / tau)
                output = sigmoid((t * (end - start) + start) / tau)
                output = (v_end - output) / (v_end - v_start)
                return np.clip(output, clip_min, 1.)
            
            k_final = np.linspace(0.0, 1.0, steps+1)
            k_final = sigmoid_schedule(k_final, 0, 3, 0.8)
            b_final = (1 - k_final) * 0.5

            beta = []
            for i in range(steps):
                b = k_final[i+1] / k_final[i]
                beta.append(b)
            beta = np.array(beta)

            k_final = k_final[1:]
            b_final = b_final[1:]


        else:
            raise NotImplementedError
        
        k_final = np.hstack([1, k_final])
        b_final = np.hstack([0, b_final])
        beta = np.hstack([1, beta])
        self.register_buffer('k_final', torch.Tensor(k_final))
        self.register_buffer('b_final', torch.Tensor(b_final))
        self.register_buffer('beta', torch.Tensor(beta))  
        self.register_buffer('cumbeta', torch.cumprod(self.beta, 0))  
        # pdb.set_trace()

        # print(f'Noise scheduler with {beta_type}:')

        # print(f'Diffusion 1.0 -> 0.5:')
        data = (1.0 * self.k_final + self.b_final).data.numpy()
        # print(' '.join([f'{d:0.4f}' for d in data]))

        # print(f'Diffusion 0.0 -> 0.5:')
        data = (0.0 * self.k_final + self.b_final).data.numpy()
        # print(' '.join([f'{d:0.4f}' for d in data]))

        # print(f'Beta:')
        # print(' '.join([f'{d:0.4f}' for d in self.beta.data.numpy()]))

    
    def one_step(self, x, t):
        dim = x.ndim - 1
        k = self.beta[t].view(-1, *([1]*dim))
        x = x * k + 0.5 * (1-k)
        return x

    def forward(self, x, t):
        dim = x.ndim - 1
        k = self.k_final[t].view(-1, *([1]*dim))
        b = self.b_final[t].view(-1, *([1]*dim))
        out = k * x + b
        return out
    

def focal_loss(inputs, targets, alpha=-1, gamma=1):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    p_t = (1 - p_t)
    p_t = p_t.clamp(min=1e-6, max=(1-1e-6)) # numerical safety
    loss = ce_loss * (p_t ** gamma)
    if alpha == -1:
        neg_weight = targets.sum(dim=-1)
        neg_weight = neg_weight / targets[0].numel()
        neg_weight = neg_weight.view(-1, 1)
        alpha_t = (1 - neg_weight) * targets + neg_weight * (1 - targets) + (1./targets[0].numel())
        # alpha_t = (1 - neg_weight) * targets + neg_weight * (1 - targets)
        loss = alpha_t * loss
    elif alpha > 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss

