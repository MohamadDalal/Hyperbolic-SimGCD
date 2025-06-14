import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import util.lorentz as L
import util.poincare as P
import wandb
import os
import math
from scipy.special import beta

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        x_norm = torch.norm(x, dim=1)
        return x_proj, logits, [(x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min())]

# Poincare linear layer from original HNN paper. Only computes matrix without bias
class PoincareLinearOriginal(nn.Module):
    def __init__(self, in_dim, out_dim, out_split=1, bias=False, gain=1.):
        super(PoincareLinearOriginal, self).__init__()
        gain = 1. ###
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_split = out_split
        weight = torch.empty(in_dim, out_dim).normal_( 
            mean=0, std=(2 * self.in_dim * self.out_dim / out_split) ** -0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        #self.bias = nn.Parameter(torch.empty(out_dim), requires_grad=bias)
        #self.reset_parameters()
        self.beta_ni = beta(self.out_dim / out_split / 2, 1 / 2)
        self.beta_n = beta(self.out_dim / 2, 1 / 2)
    
    def reset_parameters(self):
    #    nn.init.zeros_(self.bias)
        pass
    
    def forward(self, x, c):
        x_norm = x.norm(dim=-1, keepdim=True)
        Wx = torch.matmul(x, self.weight_v)
        Wx_norm = Wx.norm(dim=-1, keepdim=True)
        x = (1/c)*P.tanh((Wx_norm / x_norm) * P.artanh(c**0.5 * x_norm))*(Wx / Wx_norm)
        if self.out_split > 1:
            size = x.size()
            x = P.logmap0(x).contiguous().view(*size[:-1], self.out_split, size[-1] // self.out_split)
            x = P.expmap0(x * self.beta_ni / self.beta_n)
        return x


# TODO: Investigate that it computes math properly. Taking the direction of the input into account
class PoincareLinear(nn.Module):
    def __init__(self, in_dim, out_dim, out_split=1, bias=True, gain=1.):
        super(PoincareLinear, self).__init__()
        gain = 1. ###
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_split = out_split
        weight = torch.empty(in_dim, out_dim).normal_( 
            mean=0, std=(2 * self.in_dim * self.out_dim / out_split) ** -0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(out_dim), requires_grad=bias)
        self.reset_parameters()
        self.beta_ni = beta(self.out_dim / out_split / 2, 1 / 2)
        self.beta_n = beta(self.out_dim / 2, 1 / 2)
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x, c):
        x = poincare_linear(
            x, 
            self.weight_g, 
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            c,
            # out_split=self.out_split)
            out_split=1)
        if self.out_split > 1:
            size = x.size()
            x = P.logmap0(x).contiguous().view(*size[:-1], self.out_split, size[-1] // self.out_split)
            x = P.expmap0(x * self.beta_ni / self.beta_n)
        return x

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, out_split={}, bias={}'.format(
            self.in_dim, self.out_dim, self.out_split, self.bias.requires_grad
        )

@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return 2 * z_norm / rc * P.arsinh(
        (2. * torch.matmul(rcx, z_unit) * drcr.cosh() - (1. + cx2) * drcr.sinh()) 
        / torch.clamp_min(1. - cx2, 1e-15))

def poincare_linear(x, weight_g, weight_v, bias, c, out_split : int = 1):
    #print("\n\n\n\n\n\n\n\n\n\n")
    #print(x.shape)
    #print(x.max(dim=-1)[0])
    #print(x.min(dim=-1)[0])
    #print(x.mean(dim=-1))
    #print(x.std(dim=-1))
    #print(x.pow(2).sum(dim=-1, keepdim=True).shape)
    #print(x.pow(2).sum(dim=-1, keepdim=True).max())
    #print(x.pow(2).sum(dim=-1, keepdim=True).min())
    #print(x.pow(2).sum(dim=-1, keepdim=True).mean())
    #print(x.pow(2).sum(dim=-1, keepdim=True).std())
    rc = c.sqrt()
    x = unidirectional_poincare_mlr(x, weight_g, weight_v, bias, c)
    #print("\n\n\n\n\n\n\n\n\n\n")
    #print(x.shape)
    #print(x.max(dim=-1)[0])
    #print(x.min(dim=-1)[0])
    #print(x.mean(dim=-1))
    #print(x.std(dim=-1))
    #print(x.pow(2).sum(dim=-1, keepdim=True).shape)
    #print(x.pow(2).sum(dim=-1, keepdim=True).max())
    #print(x.pow(2).sum(dim=-1, keepdim=True).min())
    #print(x.pow(2).sum(dim=-1, keepdim=True).mean())
    #print(x.pow(2).sum(dim=-1, keepdim=True).std())
    x = (rc * x).sinh() / rc
    if out_split > 1:
        size = x.size()
        x = x.view(*size[:-1], out_split, size[-1] // out_split)

    return P._project(x / (1 + (1 + c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt()), -c, dim=-1)
    #return x / (1 + (1 + c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt())

# I couldn't get this to work, and it is useless since I need to calculate x_time alone because of f(v) and I detach x_time from the result at the end anyways.
# class LorentzLinear(nn.Module):
#     """
#     Hyperbolic linear layer.
#     """

#     def __init__(self, in_features, out_features, dropout, use_bias = False):
#         super(LorentzLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.use_bias = use_bias
#         self.bias = nn.Parameter(torch.Tensor(out_features))
#         self.weight = nn.Parameter(torch.Tensor(out_features+1, in_features+1))
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
#         torch.nn.init.constant_(self.bias, 0)

#     def forward(self, x, curv):
#         drop_weight = F.dropout(self.weight, self.dropout, training=self.training).transpose(-1, -2)
#         print(drop_weight.shape)
#         v = drop_weight.narrow(-1, 0, 1)
#         x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
#         x_full = torch.cat([x_time, x], dim=-1)
#         print(x_full.shape)
#         print(torch.norm(x_full @ drop_weight, dim=-1, keepdim=False)**2  - (1/curv))
#         print(torch.sum(x_full@v, dim=-1, keepdim=False))
#         fv = torch.sqrt(torch.norm(x_full @ drop_weight, dim=-1, keepdim=True)**2 - (1/curv))/torch.sum(x_full@v, dim=-1, keepdim=True)
#         print(v.repeat(1,x_full.shape[0]).transpose(-1, -2).shape)
#         print(v.repeat(1,x_full.shape[0]).transpose(-1, -2))
#         fv = fv*v.repeat(1, x_full.shape[0]).transpose(-1, -2)
#         print(fv)
#         print(drop_weight.narrow(-1, 1, drop_weight.shape[-1] - 1).shape)
#         #M = torch.cat([fv, drop_weight.narrow(-1, 1, drop_weight.shape[-1] - 1)], dim=-1)
#         #mv = self.manifold.mobius_matvec(drop_weight, x, curv)
#         y_space = x_full @ drop_weight.narrow(-1, 1, drop_weight.shape[-1] - 1)
#         y_time = fv @ x_full.transpose(-1, -2)
#         print(y_space.shape)
#         print(y_time.shape)
#         #res = x_full @ M.transpose(-1, -2)
#         res = res.narrow(-1, 1, x_full.shape[-1] - 1)
#         # This just moves from Lorentz to the manifold (computes x_time) so we don't need it
#         #res = self.manifold.proj(res.narrow(-1, 1, x_full.shape[-1] - 1), curv)
#         if self.use_bias:
#             #bias = self.manifold.proj_tan0(self.bias.view(1, -1), curv)
#             #hyp_bias = self.manifold.expmap0(bias, curv)
#             #hyp_bias = self.manifold.proj(hyp_bias, curv)
#             #res = self.manifold.mobius_add(res, hyp_bias, c=curv)
#             #res = self.manifold.proj(res, curv)
#             #raise NotImplementedError("Bias not implemented for LorentzLinear")
#             res = res + self.bias
#             #res = self.manifold.proj(res, curv)
#         return res

#     def extra_repr(self):
#         return 'in_features={}, out_features={}'.format(
#             self.in_features, self.out_features
#         )

# Same as above, but does not compute x_time
class LorentzLinearSimple(nn.Module):
    """
    Hyperbolic linear layer. Source: https://arxiv.org/abs/2105.14686
    """

    def __init__(self, in_features, out_features, dropout, use_bias = False):
        super(LorentzLinearSimple, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = self.weight_v = nn.Parameter(torch.Tensor(out_features, in_features+1))
        self.weight_g = nn.Parameter(self.weight.norm(dim=0))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, x, curv):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        #v = drop_weight.narrow(-1, 0, 1)
        x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
        x_full = torch.cat([x_time, x], dim=-1)
        #fv = torch.sqrt(torch.norm(x_full @ drop_weight.transpose(-1, -2), dim=-1, keepdim=True) - (1/curv))/torch.sum(x_full*v)
        #M = torch.cat([fv, drop_weight.narrow(-1, 1, drop_weight.shape[-1] - 1)], dim=-1)
        #mv = self.manifold.mobius_matvec(drop_weight, x, curv)
        res = x_full @ drop_weight.transpose(-1, -2)
        # This just moves from Lorentz to the manifold (computes x_time) so we don't need it
        #res = self.manifold.proj(res.narrow(-1, 1, x_full.shape[-1] - 1), curv)
        if self.use_bias:
            #bias = self.manifold.proj_tan0(self.bias.view(1, -1), curv)
            #hyp_bias = self.manifold.expmap0(bias, curv)
            #hyp_bias = self.manifold.proj(hyp_bias, curv)
            #res = self.manifold.mobius_add(res, hyp_bias, c=curv)
            #res = self.manifold.proj(res, curv)
            #raise NotImplementedError("Bias not implemented for LorentzLinear")
            res = res + self.bias
            #res = self.manifold.proj(res, curv)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class Hyperbolic_DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256,
                 curv_init: float = 1.0, alpha_init: float = 1.0, learn_curv: bool = True, learn_alpha: bool = True,
                 poincare: bool = False, euclidean_clip_value = None, original_poincare_layer: bool = False):
        super().__init__()
        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        # Curvature is learned in log space
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        #self.proj_alpha = nn.Parameter(torch.tensor(out_dim**-0.5).log())
        #self.proj_alpha = nn.Parameter(torch.tensor(1.7035**-1).log(), requires_grad=learn_alpha)
        #self.proj_alpha = nn.Parameter(torch.tensor(1).log(), requires_grad=learn_alpha)
        self.proj_alpha = nn.Parameter(torch.tensor(alpha_init).log(), requires_grad=learn_alpha)
        self.poincare = poincare
        self.original_poincare_layer = original_poincare_layer
        self.euclidean_clip_value = euclidean_clip_value
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # DONE: Replace last layer with a hyperbolic classifier
        #self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

        # Poincare Linear is already normalized
        #self.last_layer = nn.utils.weight_norm(PoincareLinear(bottleneck_dim, out_dim, out_split=1, bias=False))
        if self.poincare:
            if self.original_poincare_layer:
                self.last_layer = PoincareLinearOriginal(bottleneck_dim, out_dim, out_split=1, bias=False)
            else:
                self.last_layer = PoincareLinear(bottleneck_dim, out_dim, out_split=1, bias=False)
        else:
            self.last_layer = LorentzLinearSimple(bottleneck_dim, out_dim, dropout=0.0, use_bias=False)
        # Weights are initialized to a gaussian distribution in Poincare Linear, so only fill with 1 if using norm_last_layer
        if norm_last_layer:
            self.last_layer.weight_g.data.fill_(1)
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Pass features through head and last projection layers
    # Then use an exponential map to lift features to hyperbolic space
    def forward(self, x):
        x = self.mlp(x)
        #x = nn.functional.normalize(x, dim=-1, p=2)
        #x = self.last_layer(x)
        log_stats = []

        # Clamp scaling factor such that it does not up-scale the feature norms.
        # Extra: Clamp scale factor such that it does not down-scale the features too much, as that can lead to training to stop.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.proj_alpha.data = torch.clamp(self.proj_alpha.data, max=0.0, min=math.log(1e-1))
        # Clamp curvatue in case it becomes too high or too low
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        x_norm = torch.norm(x, dim=1)
        log_stats.append((x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min()))
        if self.euclidean_clip_value is not None:
            # This is done to prevent vanishing gradients according to Guo et al. "Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers"
            x = torch.where(x.norm(dim=-1, keepdim=True) < self.euclidean_clip_value, x, self.euclidean_clip_value*F.normalize(x, dim=-1))
            x_norm = torch.norm(x, dim=1)
            clipped_norm1 = (x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min())
            #for i in (x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min()):
            #    log_stats[0].append(i)
        x = x * self.proj_alpha.exp()
        x_norm = torch.norm(x, dim=1)
        clipped_norm2 = (x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min())
        with torch.autocast("cuda", dtype=torch.float32):
            x_lorentz = L.exp_map0(x, self.curv.exp())
        x_norm = torch.norm(x_lorentz, dim=1)
        log_stats.append((x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min()))
        # DONE: Clamp the norm of the Poincare embeddings to be in a certain range (This is what the project method does)
        # DONE: Check if the rest of the Poincare modules expect negative curvature (They do)
        if self.poincare:
            with torch.autocast("cuda", dtype=torch.float32):
                x_poincare = P.expmap0(x, -self.curv.exp(), project=False)
                #x_norm = torch.norm(x_poincare, dim=1)
                #print(x_norm.max())
                x_poincare = P.project(x_poincare, -self.curv.exp(), eps=3e-3)
                #x_norm = torch.norm(x_poincare, dim=1)
                #print(x_norm.max())
                #exit()
            x_norm = torch.norm(x_poincare, dim=1)
            log_stats.append((x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min()))
            logits = self.last_layer(x_poincare, self.curv.exp())
        else:
            logits = self.last_layer(x_lorentz, self.curv.exp())
        x_norm = torch.norm(logits, dim=1)
        log_stats.append((x_norm.mean(), x_norm.std(), x_norm.max(), x_norm.min()))
        if self.euclidean_clip_value is not None:
            log_stats.append(clipped_norm1)
            log_stats.append(clipped_norm2)
        return x_poincare if self.poincare else x_lorentz, logits, log_stats
    
    def train_curvature(self, train = True):
        """
        Set the curvature parameter to be trainable or not.
        """
        self.curv.requires_grad = train

    def get_curvature(self):
        """
        Returns the curvature parameter.
        """
        return self.curv.exp()
    
    def get_proj_alpha(self):
        """
        Returns the projection weight parameter.
        """
        return self.proj_alpha.exp()


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

# TODO: Consider using a learnable temperature like in CLIP and MERU
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, hyperbolic=False, poincare=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hyperbolic = hyperbolic
        self.poincare = poincare

    def forward(self, features, labels=None, mask=None, curv=1.0, use_angles=False, DEBUG_DIR=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            curv: curvature to use when computing hyperbolic distance
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        if use_angles:
            if self.hyperbolic:
                # Result of this: Highest distance will be lowest value. Lowest distance will be 0
                if self.poincare:
                    anchor_dot_contrast = torch.div(
                    torch.matmul(F.normalize(anchor_feature, dim=-1),
                                 F.normalize(contrast_feature, dim=-1).T),
                    self.temperature)
                    # for numerical stability, as soft max is translation invariant
                    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                    logits = anchor_dot_contrast - logits_max.detach()
                else:
                    # Unlike dot product. We want to minimize this, not maximize it
                    M = - L.pairwise_oxy_angle(anchor_feature, contrast_feature, curv=curv, eps=1e-6) / self.temperature
                    # for numerical stability, as soft max is translation invariant
                    logits_max, _ = torch.max(M[~torch.eye(*M.shape,dtype = torch.bool)].view(M.shape[0], M.shape[1]-1), dim=1, keepdim=True)
                    logits = M - logits_max.detach()
            else:
                # Result of this: Lowest similarity will be lowest value. Highest similarity will be 0
                anchor_dot_contrast = torch.div(
                    torch.matmul(anchor_feature, contrast_feature.T),
                    self.temperature)

                # for numerical stability, as soft max is translation invariant
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
        else:
            # compute logits. DONE: Make sure that the direction of the values is correct after stabilizing
            if self.hyperbolic:
                # Result of this: Highest distance will be lowest value. Lowest distance will be 0
                if self.poincare:
                    minus_distance = - P.pairwise_dist(anchor_feature, contrast_feature, curv=-curv, eps=1e-6) / self.temperature
                else:
                    minus_distance = - L.pairwise_dist(anchor_feature, contrast_feature, curv=curv, eps=1e-6) / self.temperature
                M = minus_distance

                # for numerical stability, as soft max is translation invariant
                logits_max, _ = torch.max(M[~torch.eye(*M.shape,dtype = torch.bool)].view(M.shape[0], M.shape[1]-1), dim=1, keepdim=True)
                logits = minus_distance - logits_max.detach()

            else:
                # compute logits
                # Result of this: Lowest similarity will be lowest value. Highest similarity will be 0
                anchor_dot_contrast = torch.div(
                    torch.matmul(anchor_feature, contrast_feature.T),
                    self.temperature)

                # for numerical stability
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits*logits_mask) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        labels = ("logits", "exp_logits", "log_prob", "mean_log_prob_pos")
        for n, i in enumerate((logits, exp_logits, log_prob, mean_log_prob_pos)):
            if True in i.isnan() and DEBUG_DIR is not None:
                print(f"{labels[n]} are NaN")
                torch.set_printoptions(profile="full")
                #print(i)
                torch.set_printoptions(profile="default")
                print(i.mean())
                print(i.std())
                print(curv)
                for m, j in enumerate((logits, exp_logits, log_prob, mean_log_prob_pos)):
                    torch.save(j, os.path.join(DEBUG_DIR, f"{labels[m]}_debug.pt"))
                    wandb.save(os.path.join(DEBUG_DIR, f"{labels[m]}_debug.pt"))
                torch.save(features, os.path.join(DEBUG_DIR, f"features_debug.pt"))
                wandb.save(os.path.join(DEBUG_DIR, f"features_debug.pt"))
                #wandb.log({labels[n]: i})
                raise ValueError(f'{labels[n]} have NaN')

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, ((logits.mean(), logits.std(), logits.max(), logits.min()),
                      (torch.exp(logits).mean(), torch.exp(logits).std(), torch.exp(logits).max(), torch.exp(logits).min()),
                      (exp_logits.mean(), exp_logits.std(), exp_logits.max(), exp_logits.min()),
                      (log_prob.mean(), log_prob.std(), log_prob.max(), log_prob.min()),
                      (mean_log_prob_pos.mean(), mean_log_prob_pos.std(), mean_log_prob_pos.max(), mean_log_prob_pos.min()))



def info_nce_logits(features, args, curv=1.0, temperature=1.0, device='cuda', use_angles=False, DEBUG_DIR=None):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    if args.hyperbolic:
        if args.poincare:
            if use_angles:
                features = F.normalize(features, dim=1)
                similarity_matrix = torch.matmul(features, features.T)
            else:
                similarity_matrix = - P.pairwise_dist(features, features, curv=-curv, eps=1e-6)
        else:
            if use_angles:
                similarity_matrix = - L.pairwise_oxy_angle(features, features, curv=curv, eps=1e-6)
            else:
                similarity_matrix = - L.pairwise_dist(features, features, curv=curv, eps=1e-6)
        if True in similarity_matrix.isnan() and DEBUG_DIR is not None:
            print("Hyperbolic distance is NaN")
            torch.set_printoptions(profile="full")
            #print(similarity_matrix)
            torch.set_printoptions(profile="default")
            print(similarity_matrix.mean())
            print(similarity_matrix.std())
            print(curv)
            torch.save(features, os.path.join(DEBUG_DIR, f"features_debug.pt"))
            wandb.save(os.path.join(DEBUG_DIR, f"features_debug.pt"))
            torch.save(similarity_matrix, os.path.join(DEBUG_DIR, f"sim_mat_debug.pt"))
            wandb.save(os.path.join(DEBUG_DIR, f"sim_mat_debug.pt"))
            return None, None
            #raise ValueError('Hyperbolic distance has NaN')
    else:
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2) # Is this falsly hard-coded?

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
