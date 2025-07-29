from .base_model import BaseModel
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from utils import get_root_logger
from utils.logger import AvgTimer
from utils.tensor_util import to_numpy
import pickle
from metrics.overlap_metric import calculate_overlap_iou, plot_pck_p2p, calculate_overlap_auc, write_geo_error_to_file


class PartialBaseModel(BaseModel):
    def __init__(self, opt):
        super(PartialBaseModel, self).__init__(opt)
        
    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            tb_logger (tensorboard logger): Tensorboard logger.
            update (bool): update best metric and best model. Default True
        """
        self.eval()

        metrics_result = {}
        variant_metrics_result = {}

        timer = AvgTimer()
        pbar = tqdm(dataloader)
        for index, data in enumerate(pbar):
            # prediction results
            results = self.validate_single(data, timer)
            p2p = to_numpy(results['p2p'])
            overlap_score12 = to_numpy(results['overlap_score12'])
            overlap_score21 = to_numpy(results['overlap_score21'])

            # ground truths
            data_x, data_y = data['first'], data['second']
            if 'dist' in data_x:
                dist_x = to_numpy(data_x['dist'])
            else:
                dist_x = to_numpy(torch.cdist(data_x['verts'], data_x['verts']))
            corr_x, corr_y = to_numpy(data_x['corr']), to_numpy(data_y['corr'])
            name_x, name_y = data_x['name'][0], data_y['name'][0]
            gt_overlap12, gt_overlap21 = to_numpy(data_x['partiality_mask']), to_numpy(data_y['partiality_mask'])

            # compute metrics
            geo_err = self.metrics['geo_error'](dist_x, corr_x, corr_y, p2p, return_mean=False)
            filtered_geo_err = calculate_overlap_auc(geo_err, overlap_score21, gt_overlap21, corr_y)
            iou1, iou2 = calculate_overlap_iou(overlap_score12, gt_overlap12), calculate_overlap_iou(overlap_score21, gt_overlap21)

            # save metrics and results
            metrics_result.setdefault('p2p', []).append(p2p)
            metrics_result.setdefault('overlap_score12', []).append(overlap_score12)
            metrics_result.setdefault('overlap_score21', []).append(overlap_score21)
            metrics_result.setdefault('geo_errors', []).append(geo_err)
            metrics_result.setdefault('filtered_geo_errors', []).append(filtered_geo_err)
            metrics_result.setdefault('iou1', []).append(iou1)
            metrics_result.setdefault('iou2', []).append(iou2)

            # display so far on screen
            geo_err_so_far = np.concatenate(metrics_result['geo_errors']).mean()
            miou1_so_far = np.mean(metrics_result['iou1'])
            miou2_so_far = np.mean(metrics_result['iou2'])
            logger = get_root_logger()
            logger.info(f"avg err:{geo_err_so_far:.4f} | err:{geo_err.mean():.4f} | iou1:{iou1:.4f} | iou2:{iou2:.4f} | miou1:{miou1_so_far:.4f} | miou2:{miou2_so_far:.4f} | {name_x} {name_y}")
            # plot pck per pair
            if 'plot_pck_per_pair' in self.metrics:
                pck_fig =self.metrics['plot_pck_per_pair']([geo_err], [f"Err: {geo_err.mean():.5f}"], threshold=0.20, steps=40)
                plt.savefig(os.path.join(self.opt['path']['results_root'],'pcks',  f'{name_x}-{name_y}.jpg'))
            # p2p variants
            if 'p2p_variants' in results.keys():
                for variant_name, variant_p2p in results['p2p_variants'].items():
                    variant_p2p = to_numpy(variant_p2p)
                    variant_geo_err = self.metrics['geo_error'](dist_x, corr_x, corr_y, variant_p2p, return_mean=False)
                    variant_metrics_result.setdefault(variant_name, []).append(variant_geo_err)
        # entire dataset metrics
        all_geo_errors = np.concatenate(metrics_result['geo_errors'])
        avg_geo_error = all_geo_errors.mean()
        miou1 = np.mean(metrics_result['iou1'])
        miou2 = np.mean(metrics_result['iou2'])
        for variant_name, variant_p2p in variant_metrics_result.items():
            variant_all_geo_errors = np.concatenate(variant_p2p)
            variant_avg_geo_error = variant_all_geo_errors.mean()
            variant_metrics_result[variant_name] = variant_avg_geo_error
        auc, pck_fig, pcks = self.metrics['plot_pck'](all_geo_errors, threshold=self.opt['val'].get('auc', 0.25))
        auc_all, pck_fig_all, pcks_all = plot_pck_p2p(np.concatenate(metrics_result['filtered_geo_errors']), threshold=1, steps=100)
        if self.opt['path'].get('results_root', None) is not None:
            write_geo_error_to_file(np.concatenate(metrics_result['filtered_geo_errors']), os.path.join(self.opt['path']['results_root'], 'tikz_plot.txt'))
        iou_fig = self.metrics['plot_iou_curve'](metrics_result['iou2'])

        # logging
        if tb_logger is not None: # train, tensorboard logging
            step = self.curr_iter // self.opt['val']['val_freq']
            tb_logger.add_figure('pck', pck_fig, global_step=step)
            tb_logger.add_figure('pck_all', pck_fig_all, global_step=step)
            tb_logger.add_figure('iou_curve', iou_fig, global_step=step)
            tb_logger.add_scalar('val auc', auc, global_step=step)
            tb_logger.add_scalar('val auc_all', auc_all, global_step=step)
            tb_logger.add_scalar('val avg error', avg_geo_error, global_step=step)
            tb_logger.add_scalar('val miou1', miou1, global_step=step)
            tb_logger.add_scalar('val miou2', miou2, global_step=step)
            for variant_name, variant_avg_geo_error in variant_metrics_result.items():
                tb_logger.add_scalar(f'val {variant_name} avg error', variant_avg_geo_error, global_step=step)
        else: # test, save to disk
            pck_fig.savefig(os.path.join(self.opt['path']['results_root'], 'pck.png'))
            pck_fig_all.savefig(os.path.join(self.opt['path']['results_root'], 'pck_all.png'))
            iou_fig.savefig(os.path.join(self.opt['path']['results_root'], 'iou_curve.png'))
            np.save(os.path.join(self.opt['path']['results_root'], 'pck.npy'), pcks)
            np.save(os.path.join(self.opt['path']['results_root'], 'pck_all.npy'), pcks_all)
        # display results
        logger = get_root_logger()
        logger.info(f'Avg time: {timer.get_avg_time():.4f}')
        logger.info(f'miou1: {miou1:.4f} | miou2: {miou2:.4f}')
        logger.info(f'Val auc: {auc:.4f}')
        logger.info(f'Val auc_all: {auc_all:.4f}')
        logger.info(f'Val avg error: {avg_geo_error:.4f}')
        for variant_name, variant_avg_geo_error in variant_metrics_result.items():
            logger.info(f'Val {variant_name} avg error: {variant_avg_geo_error:.4f}')

        # save metrics and results to disk
        if self.opt['val'].get('save_geo_errors', False):
            results_root = self.opt['path']['results_root']
            for metric_name, data in metrics_result.items():
                with open(os.path.join(results_root, f'{metric_name}.pkl'), 'wb') as f:
                    pickle.dump(data, f)

        # update best model state dict
        if update and (self.best_metric is None or (avg_geo_error < self.best_metric)):
            self.best_metric = avg_geo_error
            self.best_networks_state_dict = self._get_networks_state_dict()
            logger.info(f'Best model is updated, average geodesic error: {self.best_metric:.4f}')
            self.save_model(net_only=False, best=True, save_filename="best.pth")

        # train mode
        self.train()