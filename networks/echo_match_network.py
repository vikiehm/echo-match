import torch
import torch.nn as nn
from networks.diffusion_network import DiffusionNet
from utils.fmap_util import get_mask
from utils.registry import NETWORK_REGISTRY

class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # compute linear operator matrix representation C1 and C2
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat

        D = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        C_i = []
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C

class Similarity(nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super(Similarity, self).__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard

    def forward(self, log_alpha):
        log_alpha = log_alpha / self.tau
        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))

        if self.hard:
            # Straight through.
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = alpha_hard - alpha.detach() + alpha
        else:
            ret = alpha
        return ret

@NETWORK_REGISTRY.register()
class Echo_Match_Net(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, cfg, input_type='xyz', augmentation={'train': {}, 'test': {}}):
        super().__init__()
        self.cfg = cfg

        # feature extractor
        self.feature_extractor = DiffusionNet(
            in_channels=cfg["feature_extractor"]["in_channels"],
            out_channels=cfg["feature_extractor"]["out_channels"],
            hidden_channels=cfg["feature_extractor"]["out_channels"],
            n_block=4,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
            input_type=input_type,
            augmentation=augmentation
        )

        # learnable tau
        learned_tau = nn.Parameter(torch.Tensor([0.10]))
        self.permutation_network = Similarity(tau=learned_tau)

        # another diffusionnet for regress overlap
        self.overlap_diffusion_net = DiffusionNet(
            in_channels=cfg["overlap"]["neighbor_size"],
            out_channels=1,
            hidden_channels=cfg["overlap"]["hidden_channels"],
            n_block=cfg["overlap"]["blocks"],
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
            last_activation=nn.Sigmoid(),
        )

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"], resolvant_gamma=cfg["fmap"]["resolvant_gamma"])