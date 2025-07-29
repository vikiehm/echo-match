import torch
import torch.nn.functional as F
from .partial_base_model import PartialBaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap, get_gt_fmap_with_regularization

@MODEL_REGISTRY.register()
class Echo_Match_Model(PartialBaseModel):
    def __init__(self, opt):
        super(Echo_Match_Model, self).__init__(opt)
        self.prune_strategy = opt.get('prune_strategy', 'f2_only')
        self.lambda_ = opt.get('lambda', 0)
        print(f"prune_strategy: {self.prune_strategy}")

    def feed_data(self, data):
        # get data pair
        data = to_device(data, self.device)
        data_x, data_y = data['first'], data['second']

        # get spectral operators
        evecs_x, evecs_y = data_x['evecs'][0], data_y['evecs'][0]
        evals_x, evals_y = data_x['evals'][0], data_y['evals'][0]
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'][0], data_y['evecs_trans'][0]

        # extract features
        feat_x = self.networks['echo_match_net'].feature_extractor(data=data_x)
        feat_y = self.networks['echo_match_net'].feature_extractor(data=data_y)

        feat_x_normed = F.normalize(feat_x, dim=-1, p=2)
        feat_y_normed = F.normalize(feat_y, dim=-1, p=2)
        # compute similarity and permutation matrices
        similarity = torch.bmm(feat_x_normed, feat_y_normed.transpose(1, 2))
        Pxy = self.networks['echo_match_net'].permutation_network(similarity).squeeze(0)
        Pyx = self.networks['echo_match_net'].permutation_network(similarity.transpose(1, 2)).squeeze(0)
        clean_verts_x = data_x["verts"][0]
        clean_verts_y = data_y["verts"][0]

        # get or compute distance matrices
        if "dist" in data_x and "dist" in data_y:
            dists_x = data_x["dist"][0]
            dists_y = data_y["dist"][0]
        else:
            dists_x = torch.cdist(clean_verts_x, clean_verts_x)
            dists_y = torch.cdist(clean_verts_y, clean_verts_y)

        # get nearest neighbor indices
        k_x = min(self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"], dists_x.shape[1])
        k_y = min(self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"], dists_y.shape[1])
        _, idx_x = torch.topk(dists_x, k_x, largest=False, sorted=True)
        _, idx_y = torch.topk(dists_y, k_y, largest=False, sorted=True)

        # compute overlap scores
        matrix_x = Pxy @ Pyx
        matrix_y = Pyx @ Pxy

        score_x = matrix_x.gather(1, idx_x)
        score_y = matrix_y.gather(1, idx_y)

        score_x = score_x / score_x.max()
        score_y = score_y / score_y.max()

        # Pad scores with zeros if k is smaller than overlap_in
        if k_x < self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"]:
            pad_size = self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"] - k_x
            score_x = torch.cat([score_x, torch.zeros(score_x.shape[0], pad_size, device=score_x.device)], dim=1)
            
        if k_y < self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"]:
            pad_size = self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"] - k_y
            score_y = torch.cat([score_y, torch.zeros(score_y.shape[0], pad_size, device=score_y.device)], dim=1)

        # predict overlap scores using diffusion net
        overlap_score12 = self.networks['echo_match_net'].overlap_diffusion_net(x_in=score_x.unsqueeze(0), data=data_x).flatten()
        overlap_score21 = self.networks['echo_match_net'].overlap_diffusion_net(x_in=score_y.unsqueeze(0), data=data_y).flatten()

        gt_partiality_mask12 = data['first'].get('partiality_mask', torch.ones((evecs_x.shape[0])).long().to(self.device)).squeeze()
        gt_partiality_mask21 = data['second'].get('partiality_mask', torch.ones((evecs_y.shape[0])).long().to(self.device)).squeeze()

        if self.prune_strategy == "f2_only":
            if self.networks['echo_match_net'].training:
                use_feat_x = feat_x
                use_feat_y = feat_y * gt_partiality_mask21.unsqueeze(0).unsqueeze(-1)
            else:
                use_feat_x = feat_x
                use_feat_y = feat_y
        elif self.prune_strategy == "both":
            if self.networks['echo_match_net'].training:
                use_feat_x = feat_x * gt_partiality_mask12.unsqueeze(0).unsqueeze(-1)
                use_feat_y = feat_y * gt_partiality_mask21.unsqueeze(0).unsqueeze(-1)
            else:
                use_feat_x = feat_x
                use_feat_y = feat_y
        elif self.prune_strategy == "none":
            use_feat_x = feat_x
            use_feat_y = feat_y
        else:
            raise ValueError(f"Unknown prune strategy: {self.prune_strategy}")

        # predict fmap
        Cxy = self.networks['echo_match_net'].fmreg_net(use_feat_x, use_feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # gt partiality mask
        gt_partiality_mask12 = data['first'].get('partiality_mask', torch.ones((evecs_x.shape[0])).long().to(self.device)).squeeze()
        gt_partiality_mask21 = data['second'].get('partiality_mask', torch.ones((evecs_y.shape[0])).long().to(self.device)).squeeze()

        # gt correspondence
        corr_x = data['first']['corr'][0]
        corr_y = data['second']['corr'][0]

        # gt functional map
        if self.lambda_ == 0:
            P = torch.zeros((evecs_y.shape[0], evecs_x.shape[0])).to(self.device) # [Ny, Nx]
            P[corr_y, corr_x] = 1
            C_gt = evecs_trans_y @ P @ evecs_x
            C_gt = C_gt.unsqueeze(0)
        else:
            C_gt = get_gt_fmap_with_regularization(corr_x, corr_y, evecs_x, evecs_y, evals_x, evals_y, lambda_=self.lambda_, resolvant_gamma=0.5, device=self.device)

        # loss
        fmap_loss, acc_loss, nce_loss = self.losses["echo_match_loss"](C_gt, Cxy, corr_x, corr_y, feat_x, feat_y,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

        self.loss_metrics = {'fmap_loss': fmap_loss, 'acc_loss': acc_loss, 'nce_loss': nce_loss}

    def validate_single(self, data, timer):
        # start record
        timer.start()
        
        # get data pair
        data = to_device(data, self.device)
        data_x, data_y = data['first'], data['second']

        # get spectral operators
        evecs_x, evecs_y = data_x['evecs'][0], data_y['evecs'][0]
        evals_x, evals_y = data_x['evals'][0], data_y['evals'][0]
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'][0], data_y['evecs_trans'][0]

        # extract features
        feat_x = self.networks['echo_match_net'].feature_extractor(data=data_x)
        feat_y = self.networks['echo_match_net'].feature_extractor(data=data_y)

        feat_x_normed = F.normalize(feat_x, dim=-1, p=2)
        feat_y_normed = F.normalize(feat_y, dim=-1, p=2)
        # compute similarity and permutation matrices
        similarity = torch.bmm(feat_x_normed, feat_y_normed.transpose(1, 2))
        Pxy = self.networks['echo_match_net'].permutation_network(similarity).squeeze(0)
        Pyx = self.networks['echo_match_net'].permutation_network(similarity.transpose(1, 2)).squeeze(0)
        clean_verts_x = data_x["verts"][0]
        clean_verts_y = data_y["verts"][0]

        # get or compute distance matrices
        if "dist" in data_x and "dist" in data_y:
            dists_x = data_x["dist"][0]
            dists_y = data_y["dist"][0]
        else:
            dists_x = torch.cdist(clean_verts_x, clean_verts_x)
            dists_y = torch.cdist(clean_verts_y, clean_verts_y)

        # get nearest neighbor indices
        k_x = min(self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"], dists_x.shape[1])
        k_y = min(self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"], dists_y.shape[1])
        _, idx_x = torch.topk(dists_x, k_x, largest=False, sorted=True)
        _, idx_y = torch.topk(dists_y, k_y, largest=False, sorted=True)

        # compute overlap scores
        matrix_x = Pxy @ Pyx
        matrix_y = Pyx @ Pxy

        score_x = matrix_x.gather(1, idx_x)
        score_y = matrix_y.gather(1, idx_y)

        score_x = score_x / score_x.max()
        score_y = score_y / score_y.max()

        # Pad scores with zeros if k is smaller than overlap_in
        if k_x < self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"]:
            pad_size = self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"] - k_x
            score_x = torch.cat([score_x, torch.zeros(score_x.shape[0], pad_size, device=score_x.device)], dim=1)
            
        if k_y < self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"]:
            pad_size = self.networks['echo_match_net'].cfg["overlap"]["neighbor_size"] - k_y
            score_y = torch.cat([score_y, torch.zeros(score_y.shape[0], pad_size, device=score_y.device)], dim=1)

        # predict overlap scores using diffusion net
        overlap_score12 = self.networks['echo_match_net'].overlap_diffusion_net(x_in=score_x.unsqueeze(0), data=data_x).flatten()
        overlap_score21 = self.networks['echo_match_net'].overlap_diffusion_net(x_in=score_y.unsqueeze(0), data=data_y).flatten()

        use_feat_x, use_feat_y = (feat_x, feat_y) 

        # predict fmap
        Cxy = self.networks['echo_match_net'].fmreg_net(use_feat_x, use_feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cxy = Cxy.squeeze()

        # direct point-to-point map
        p2p_point = nn_query(feat_x_normed, feat_y_normed)

        # finish record
        timer.record()

        return {
            'p2p': p2p_point,
            'overlap_score12': overlap_score12,
            'overlap_score21': overlap_score21,
        }
