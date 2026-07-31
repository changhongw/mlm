import librosa
import logging
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
from typing import Any, Dict, Tuple

from pytorch_lightning.core.module import LightningModule
import torch
from torch import nn
import torch.nn.functional as F

from src.utils.preprocess_utils import sylphone_feature_idx, save_notes
from src.utils.utils_mlm import (withinsample_negatives, pad_sequence, pytorch_cos_sim,
                                 define_metrics, compute_stress_metrics, compute_rhyme_metrics, output_alignment)

log = logging.getLogger(__name__)

class MLM_CAL(LightningModule):

    def __init__(self,
                 encoder: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 contrastive_loss_fn: nn.Module | None = None,
                 base_lr: float = 1e-4,
                 ):
        super(MLM_CAL, self).__init__()
        """
        Initialize the MLM_CAL model.

        Args:
            encoder: Transformer encoder.
            optimizer: optimizer to use for training.
            scheduler: learning rate scheduler.
            contrastive_loss_fn: contrastive loss function.
            base_lr: base learning rate for the optimizer.
        """

        self.save_hyperparameters(ignore=["encoder", "contrastive_loss_fn"])

        self.encoder = encoder
        self.optimizer_cls = optimizer
        self.scheduler_cls = scheduler
        self.base_lr = base_lr

        # loss & training strategy
        self.contrastive_loss = contrastive_loss_fn

        # get specific indices for sylphone features
        self.vowel_idx, self.stress_idx, self.conso_end_idx, self.longvowel_idx = sylphone_feature_idx()


    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                x_mask: torch.Tensor,
                y_mask: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder.

        Args:
            x: Input tensor for the first modality (e.g., melody); shape: [batch_size, seq_len, feature_dim].
            y: Input tensor for the second modality (e.g., lyrics); shape: [batch_size, seq_len, feature_dim].
            x_mask: Mask for the first modality; shape: [batch_size, seq_len].
            y_mask: Mask for the second modality; shape: [batch_size, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded representations for the first and second modalities.
        """
        
        xe, ye = self.encoder(x, y, x_mask, y_mask)  

        return xe, ye

        
    def training_step(self, batch, batch_idx):

        # load batch data
        x, y, x_mask, y_mask = batch
        # get batch size
        b = len(batch[0])

        # number of total negatives, expcet one positive pair
        neg_num = b - 1  

        if self.contrastive_loss.contrastive_setting == "collapse_sequence":

            x = x.float()
            y = y.float()
            xe, ye = self(x[:,:,:-1], y[:,:,:-1], x_mask, y_mask) 

            _, n, d = xe.shape
            m = ye.size(1)

            xe = torch.mean(xe, 1)
            ye = torch.mean(ye, 1)
            cosine_sim = pytorch_cos_sim(xe, ye)
            total_loss = - torch.mean(torch.diag(cosine_sim) - torch.logsumexp(cosine_sim, -1))
            
            self.log_dict({f"loss/train/total": total_loss}, sync_dist=True, batch_size=b)

        elif self.contrastive_loss.contrastive_setting == "withinbatch_negative":

            x = x.float()
            y = y.float()
            xe, ye = self(x[:,:,:-1], y[:,:,:-1], x_mask, y_mask) 

            _, n, d = xe.shape
            m = ye.size(1)

            xp = torch.repeat_interleave(xe, b , dim=0) # [0, 0, ..., 0, 1, 1, ..., 1, ..., b-1, b-1, ..., b-1]
            yp = torch.repeat_interleave(ye, b , dim=0)

            xp_mask = torch.repeat_interleave(x_mask, b , dim=0)
            yp_mask = torch.repeat_interleave(y_mask, b, dim=0)

            # get the original embeddings and masks, which are the positive pairs
            xn = torch.flatten(xe.expand(b, b, n, d), 0, 1)  # [0,1,...,b-1, 0,1,...,b-1, ..., 0,1,...,b-1]
            yn = torch.flatten(ye.expand(b, b, m, d), 0, 1)  

            xn_mask = torch.flatten(x_mask.expand(b, b, n), 0, 1)  # [b*b, n]
            yn_mask = torch.flatten(y_mask.expand(b, b, m), 0, 1)  # [b*b, m]

            contrast_m, D_posm, D_negm = self.contrastive_loss(xp, yn, xp_mask, yn_mask, b)
            contrast_l, D_posl, D_negl = self.contrastive_loss(yp, xn, yp_mask, xn_mask, b)                                                               

            contrast = contrast_m + contrast_l
            total_loss = contrast
                                                                            
            # add elems to dict
            loss_dict = dict(
                            contrast_m=contrast_m,
                            contrast_l=contrast_l,
                            D_pos=(torch.mean(D_posm) + torch.mean(D_posl))/2,
                            D_neg=(torch.mean(D_negm) + torch.mean(D_negl))/2,
                            total_loss=total_loss
                            )

            self.log_dict({f"loss/train/{k}": v for k, v in loss_dict.items()}, sync_dist=True, batch_size=b)
        
        else: # "shuffle_negative" negatives, which is the default setting

            # create within-sample negatives by shuffling within the sequence
            x, y, x_mask, y_mask = withinsample_negatives(x, y, x_mask, y_mask, self.contrastive_loss.cross_ratio,
                                                                                self.contrastive_loss.shuffle_ratio,
                                                                                neg_num
                                                                                    )

            x = x.float()
            y = y.float()
            xe, ye = self(x[:,:,:-1], y[:,:,:-1], x_mask, y_mask) 

            _, n, d = xe.shape
            m = ye.size(1)

            # get the original embeddings and masks, which are the positive pairs
            xe_ori = xe.reshape(b, -1, n, d)[:, 0, :, :]
            ye_ori = ye.reshape(b, -1, m, d)[:, 0, :, :]

            x_ori = x.reshape(b, -1, n, x.size(-1))[:, 0, :, :]
            y_ori = y.reshape(b, -1, m, y.size(-1))[:, 0, :, :]

            xmask_ori = x_mask.reshape(b, -1, n)[:, 0, :]
            ymask_ori = y_mask.reshape(b, -1, m)[:, 0, :]

            # create corss-sample negatives: for a given melody, lyrics of other songs are all negatives
            offdiag_mask = ~torch.eye(b, dtype=torch.bool)

            # get number of cross-sample negatives
            cross_num = int(neg_num * self.contrastive_loss.cross_ratio)
            
            if cross_num > 0:
                xe_n = torch.flatten(xe_ori.expand(b, b, n, d)[offdiag_mask, :, :].reshape(b, -1, n, d)[:, :cross_num, :, :], 0, 1)
                ye_n = torch.flatten(ye_ori.expand(b, b, m, d)[offdiag_mask, :, :].reshape(b, -1, m, d)[:, :cross_num, :, :], 0, 1)
                x_n = torch.flatten(x_ori.expand(b, b, n, x_ori.size(-1))[offdiag_mask, :, :].reshape(b, -1, n, x_ori.size(-1))[:, :cross_num, :, :], 0, 1)
                y_n = torch.flatten(y_ori.expand(b, b, m, y_ori.size(-1))[offdiag_mask, :, :].reshape(b, -1, m, y_ori.size(-1))[:, :cross_num, :, :], 0, 1)

                xn_mask = torch.flatten(xmask_ori.expand(b, b, n)[offdiag_mask, :].reshape(b, -1, n)[:, :cross_num, :], 0, 1)
                yn_mask = torch.flatten(ymask_ori.expand(b, b, m)[offdiag_mask, :].reshape(b, -1, m)[:, :cross_num, :], 0, 1)

                # pad or truncate cross-sample negatives to the same length of the positive melody
                xe_temp, x_temp = [], []
                for k in range(len(xe_n)):
                    target_len = sum(~xmask_ori[k//cross_num]).item()
                    max_len = max(torch.sum(~xn_mask[k//cross_num * cross_num : (k//cross_num+1) * cross_num], -1)).item()
                    max_len = max(max_len, target_len)
                    xe_temp.append(F.interpolate(xe_n[k][:sum(~xn_mask[k])].unsqueeze(0).unsqueeze(0), size=(max_len, d), 
                                                mode='nearest').squeeze(0).squeeze(0)[:target_len, :])
                    x_temp.append(F.interpolate(x_n[k][:sum(~xn_mask[k])].unsqueeze(0).unsqueeze(0), size=(max_len, x_n.size(-1)), 
                                                mode='nearest').squeeze(0).squeeze(0)[:target_len, :])
                    
                # pad or truncate cross-sample negatives to the same length of the positive lyrics
                ye_temp, y_temp = [], []
                for k in range(len(ye_n)):
                    target_len = sum(~ymask_ori[k//cross_num]).item()
                    max_len = max(torch.sum(~yn_mask[k//cross_num * cross_num: (k//cross_num+1) * cross_num], -1)).item()
                    max_len = max(max_len, target_len)
                    ye_temp.append(F.interpolate(ye_n[k][:sum(~yn_mask[k])].unsqueeze(0).unsqueeze(0), size=(max_len, d), 
                                                mode='nearest').squeeze(0).squeeze(0)[:target_len, :])
                    y_temp.append(F.interpolate(y_n[k][:sum(~yn_mask[k])].unsqueeze(0).unsqueeze(0), size=(max_len, y_n.size(-1)), 
                                                mode='nearest').squeeze(0).squeeze(0)[:target_len, :])
                    
                xe_cross = [xe_temp[i:i + cross_num] for i in range(0, len(xe_temp), cross_num)]
                xe_cross = [item for sublist in xe_cross for item in sublist]
                ye_cross = [ye_temp[i:i + cross_num] for i in range(0, len(ye_temp), cross_num)]
                ye_cross = [item for sublist in ye_cross for item in sublist]
                xe_n, ye_n, xn_mask, yn_mask = pad_sequence(xe_cross, ye_cross) 

                x_cross = [x_temp[i:i + cross_num] for i in range(0, len(x_temp), cross_num)]
                x_cross = [item for sublist in x_cross for item in sublist]
                y_cross = [y_temp[i:i + cross_num] for i in range(0, len(y_temp), cross_num)]
                y_cross = [item for sublist in y_cross for item in sublist]
                x_n, y_n, xn_mask, yn_mask = pad_sequence(x_cross, y_cross) 

                xe_n = xe_n.reshape(b, -1, xe_n.size(-2), xe_n.size(-1))
                xe = xe.reshape(b, -1, xe.size(-2), xe.size(-1))
                xe_n = torch.flatten(torch.cat((xe, xe_n), dim=1), start_dim=0, end_dim=1)

                ye_n = ye_n.reshape(b, -1, ye_n.size(-2), ye_n.size(-1))
                ye = ye.reshape(b, -1, ye.size(-2), ye.size(-1))
                ye_n = torch.flatten(torch.cat((ye, ye_n), dim=1), start_dim=0, end_dim=1)

                xn_mask = xn_mask.reshape(b, -1, xn_mask.size(-1))
                x_mask = x_mask.reshape(b, -1, x_mask.size(-1))
                xn_mask = torch.flatten(torch.cat((x_mask, xn_mask), dim=1), start_dim=0, end_dim=1)

                yn_mask = yn_mask.reshape(b, -1, yn_mask.size(-1))
                y_mask = y_mask.reshape(b, -1, y_mask.size(-1))
                yn_mask = torch.flatten(torch.cat((y_mask, yn_mask), dim=1), start_dim=0, end_dim=1)

                x_n = x_n.reshape(b, -1, x_n.size(-2), x_n.size(-1))
                x = x.reshape(b, -1, x.size(-2), x.size(-1))
                x_n = torch.flatten(torch.cat((x, x_n), dim=1), start_dim=0, end_dim=1)

                y_n = y_n.reshape(b, -1, y_n.size(-2), y_n.size(-1))
                y = y.reshape(b, -1, y.size(-2), y.size(-1))
                y_n = torch.flatten(torch.cat((y, y_n), dim=1), start_dim=0, end_dim=1)
            
            else:
                xe_n = xe.clone()
                ye_n = ye.clone()

                xn_mask = x_mask.clone()
                yn_mask = y_mask.clone()

            xp = torch.repeat_interleave(xe_ori, b , dim=0)
            yp = torch.repeat_interleave(ye_ori, b , dim=0)

            xp_mask = torch.repeat_interleave(xmask_ori, b , dim=0)
            yp_mask = torch.repeat_interleave(ymask_ori, b, dim=0)

            contrast_m, D_posm, D_negm = self.contrastive_loss(xp, ye_n, xp_mask, yn_mask, b)
            contrast_l, D_posl, D_negl = self.contrastive_loss(yp, xe_n, yp_mask, xn_mask, b)

            contrast = contrast_m + contrast_l
            total_loss = contrast
                                                                            
            # add elems to dict
            loss_dict = dict(
                            contrast_m=contrast_m,
                            contrast_l=contrast_l,
                            D_pos=(torch.mean(D_posm) + torch.mean(D_posl))/2,
                            D_neg=(torch.mean(D_negm) + torch.mean(D_negl))/2,
                            total_loss=total_loss
                            )

            self.log_dict({f"loss/train/{k}": v for k, v in loss_dict.items()}, sync_dist=True, batch_size=b)

        return total_loss


    def on_train_epoch_end(self) -> None:
        r"""Log embedding at the end of each epoch."""

        self.log("hparams/gamma", self.contrastive_loss.sdtw.gamma, sync_dist=True)
        self.log("hparams/temperature", self.contrastive_loss.temperature, sync_dist=True)
        self.log("hparams/regularization_weight", self.contrastive_loss.regularization_weight, sync_dist=True)

       
    def validation_step(self, batch, batch_idx):

        x, y, x_mask, y_mask, id_pairs, truth_info = batch 

        lyrics_ids = [item[1] for item in id_pairs[:-1]]  # last one is to keep truth info
        truth_rank_ori, truth_id, candi_num_ori = truth_info
        
        x = x.float()
        y = y.float()
        xe, ye = self(x[:,:,:-1], y[:,:,:-1], x_mask, y_mask) 

        ######### alignment cost by SDTW #######
        b, n, d = xe.shape
        m = ye.size(1)

        if self.contrastive_loss.contrastive_setting == "collapse_sequence":
            xe = torch.mean(xe, 1)
            ye = torch.mean(ye, 1)
            x_norm = F.normalize(xe, p=2, dim=-1)
            y_norm = F.normalize(ye, p=2, dim=-1)
            cosine_sim = F.cosine_similarity(x_norm, y_norm, dim=-1)
            val_loss = - torch.mean(cosine_sim[-1] - torch.logsumexp(cosine_sim[:-1], -1))
            self.log("loss/val", val_loss, sync_dist=True, batch_size=1)

        else: # for "withinbatch_negative" and "shuffle_negative" settings, which both use the same DTW-based evaluation
            D_mask = torch.zeros(b, n, m, device=xe.device).type(torch.bool)

            for k in range(b):
                D_mask[k, x_mask[k], :] = True
                D_mask[k, :, y_mask[k]] = True

            align_cost = self.contrastive_loss.sdtw(xe, ye, ~D_mask).squeeze()

            new_val, new_rank = torch.sort(align_cost[:-1], stable=True)  # sort the negative pairs, keep the last one (positive pair) unchanged
            newsort_ids = [lyrics_ids[k] for k in new_rank]
            if truth_id in newsort_ids:
                truth_rank_re = newsort_ids.index(truth_id)
                neg_idx = torch.arange(new_val.size(0), device=align_cost.device)[torch.arange(new_val.size(0), 
                                        device=align_cost.device) != torch.tensor(truth_rank_re, device=align_cost.device)]
                val_sdtw_neg = new_val[neg_idx]
                val_ids_neg = newsort_ids.copy()
                val_ids_neg.remove(truth_id)

                rank_val = new_val[truth_rank_re]
                truth_rank_ratio = sum(new_val <= rank_val).item() / candi_num_ori

            else:
                truth_rank_re = int(truth_rank_ori * candi_num_ori)
                truth_rank_ratio = truth_rank_ori
                val_sdtw_neg = new_val
                val_ids_neg = newsort_ids   # all except the last one, which is the truth pair

            D_loss = torch.cat([val_sdtw_neg, align_cost[-1].unsqueeze(0)])
            D_loss = (D_loss - torch.mean(D_loss))  / (torch.std(D_loss) + 1e-6)
            D_posm = D_loss[-1]

            pos_m = - D_posm / self.contrastive_loss.temperature
            all_m = - D_loss / self.contrastive_loss.temperature

            val_loss = - (pos_m -  torch.logsumexp(all_m, 0))

            self.log("loss/val", val_loss, sync_dist=True, batch_size=1)

            truth_rank_ratio = torch.tensor(truth_rank_ratio, device=self.device)

            self.log("metrics/truth_rank_ratio", truth_rank_ratio, sync_dist=True, batch_size=1)

            hit1 = (truth_rank_ratio < 0.01).float().mean() * 100
            hit3 = (truth_rank_ratio < 0.03).float().mean() * 100
            hit5 = (truth_rank_ratio < 0.05).float().mean() * 100

            self.log("metrics/val_hit1", hit1, sync_dist=True, batch_size=1)
            self.log("metrics/val_hit3", hit3, sync_dist=True, batch_size=1)
            self.log("metrics/val_hit5", hit5, sync_dist=True, batch_size=1)


    def on_predict_epoch_start(self):

        # set metrics computed on validation set
        self.propose_HitK = []
        self.lengthInform_HitK = []
        self.all_metrics = define_metrics()

    def predict_step(self, batch, batch_idx):

        x, y, x_mask, y_mask, id_pairs, truth_info = batch 

        lyrics_ids = [item[1] for item in id_pairs[:-1]]  # last one is to keep truth info
        truth_rank_ori, truth_id, candi_num_ori = truth_info
        
        x = x.float()
        y = y.float()
        xe, ye = self(x[:,:,:-1], y[:,:,:-1], x_mask, y_mask) 

        ######### alignment cost by SDTW #######
        b, n, d = xe.shape
        m = ye.size(1)

        if self.contrastive_loss.contrastive_setting == "collapse_sequence":
            xe = torch.mean(xe, 1)
            ye = torch.mean(ye, 1)
            x_norm = F.normalize(xe, p=2, dim=-1)
            y_norm = F.normalize(ye, p=2, dim=-1)
            cosine_sim = F.cosine_similarity(x_norm, y_norm, dim=-1)
            # total_loss = - torch.mean(cosine_sim[0] - torch.logsumexp(cosine_sim, -1))
            align_cost = 1 - cosine_sim

        else: # for "withinbatch_negative" and "shuffle_negative" settings, which both use the same DTW-based evaluation
            D_mask = torch.zeros(b, n, m, device=xe.device).type(torch.bool)

            for k in range(b):
                D_mask[k, x_mask[k], :] = True
                D_mask[k, :, y_mask[k]] = True

            # totally use dtw for ranking, instead of SDTW
            align_cost = self.contrastive_loss.sdtw(xe, ye, ~D_mask).squeeze()

        new_val, new_rank = torch.sort(align_cost[:-1], stable=True)  # sort the negative pairs, keep the last one (positive pair) unchanged
        newsort_ids = [lyrics_ids[k] for k in new_rank]
        if truth_id in newsort_ids:
            truth_rank_re = newsort_ids.index(truth_id)
            neg_idx = torch.arange(new_val.size(0), device=align_cost.device)[torch.arange(new_val.size(0), 
                                    device=align_cost.device) != torch.tensor(truth_rank_re, device=align_cost.device)]
            val_sdtw_neg = new_val[neg_idx]
            val_ids_neg = newsort_ids.copy()
            val_ids_neg.remove(truth_id)

            rank_val = new_val[truth_rank_re]
            truth_rank_ratio = sum(new_val <= rank_val).item() / candi_num_ori

        else:
            truth_rank_re = int(truth_rank_ori * candi_num_ori)
            truth_rank_ratio = truth_rank_ori
            val_sdtw_neg = new_val
            val_ids_neg = newsort_ids   # all except the last one, which is the truth pair

        self.propose_HitK.append(truth_rank_ratio)
        self.lengthInform_HitK.append(truth_rank_ori)
        

    def on_predict_epoch_end(self):

        ################ overall H@K metrics ################
        self.propose_HitK = torch.tensor(self.propose_HitK)
        self.lengthInform_HitK = torch.tensor(self.lengthInform_HitK)

        HitK = [{"Hit@1": round(torch.sum(self.propose_HitK < 0.01).item() / len(self.propose_HitK), 4) * 100,
                "Hit@3": round(torch.sum(self.propose_HitK < 0.03).item() / len(self.propose_HitK), 4) * 100,
                "Hit@5": round(torch.sum(self.propose_HitK < 0.05).item() / len(self.propose_HitK), 4) * 100,
                "Method": "proposed_seg"+str(self.trainer.datamodule.seg_lnum)
                },
                {"Hit@1": round(torch.sum(self.lengthInform_HitK < 0.01).item() / len(self.lengthInform_HitK), 4) * 100,
                "Hit@3": round(torch.sum(self.lengthInform_HitK < 0.03).item() / len(self.lengthInform_HitK), 4) * 100,
                "Hit@5": round(torch.sum(self.lengthInform_HitK < 0.05).item() / len(self.lengthInform_HitK), 4) * 100,
                "Method": "lengthInform_seg"+str(self.trainer.datamodule.seg_lnum)
                }]

        topHits = pd.DataFrame(HitK)

        # save H@K metrics to a csv file
        topHits.to_csv('outputs/pred_HitK_seg'+str(self.trainer.datamodule.seg_lnum)+
                       '_top'+str(self.trainer.datamodule.metric_topk)+ 
                       '_gamma'+ str(round(self.contrastive_loss.gamma.item(), 4))+'.csv', index=False)
        

    ####### test stage #######
    def on_test_epoch_start(self):

        self.propose_HitK = []
        self.lengthInform_HitK = []
        self.all_metrics = define_metrics()

    def test_step(self, batch, batch_idx):


        x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

        lyrics_ids = [item[1] for item in id_pairs[:-1]]  # last one is to keep truth info
        truth_rank_ori, truth_id, truth_path, melody_ori, candi_num_ori = truth_info

        # not including the last dimension which is line information
        xe, ye = self(x[:,:,:-1].float(), y[:,:,:-1].float(), x_mask, y_mask) 

        ######### alignment cost by SDTW #####
        b, n, d = xe.shape
        m = ye.size(1)

        D_mask = torch.zeros(b, n, m, device=xe.device).type(torch.bool)

        for k in range(b):
            D_mask[k, x_mask[k], :] = True
            D_mask[k, :, y_mask[k]] = True

        # totally use dtw for ranking, instead of SDTW
        align_cost = self.contrastive_loss.sdtw(xe, ye, ~D_mask).squeeze()
        cost_mats = self.contrastive_loss.sdtw._cosinesim_dist_func(xe, ye)

        new_val, new_rank = torch.sort(align_cost[:-1], stable=True)
        newsort_ids = [lyrics_ids[k] for k in new_rank]
        if truth_id in newsort_ids:
            truth_rank_re = newsort_ids.index(truth_id)
            neg_idx = torch.arange(new_val.size(0), device=align_cost.device)[torch.arange(new_val.size(0), 
                                    device=align_cost.device) != torch.tensor(truth_rank_re, device=align_cost.device)]
            val_sdtw_neg = new_val[neg_idx]
            val_ids_neg = newsort_ids.copy()
            val_ids_neg.remove(truth_id)

            rank_val = new_val[truth_rank_re]
            truth_rank_ratio = sum(new_val <= rank_val).item() / candi_num_ori

        else:
            truth_rank_re = int(truth_rank_ori * candi_num_ori)
            truth_rank_ratio = truth_rank_ori
            val_sdtw_neg = new_val
            val_ids_neg = newsort_ids   # all except the last one, which is the truth pair

        self.propose_HitK.append(truth_rank_ratio)
        self.lengthInform_HitK.append(truth_rank_ori)

        ##################### output alignment ########################
        topk_rank = new_rank[:self.trainer.datamodule.metric_topk]
        
        for k in range(self.trainer.datamodule.metric_topk): # compute metrics for each topk
            # get path provided by DTW
            rank = topk_rank[k]
            music = x[rank].cpu()
            lyrics = y[rank].cpu()
            seg_words = words[rank.cpu().item()]

            music_mask = x_mask[rank].cpu()
            lyrics_mask = y_mask[rank].cpu()

            len_x = sum(~music_mask)
            len_y = sum(~lyrics_mask)

            path_w = torch.tensor([1.0, 1.0, 1.0]).numpy()

            D, wp = librosa.sequence.dtw(C=cost_mats[rank].cpu()[:len_x, :len_y].cpu().numpy(), weights_mul=path_w)
            align_path = wp[::-1]

            stress_metric = compute_stress_metrics(music, music_mask, lyrics, lyrics_mask, align_path)
            for key in stress_metric.keys():
                self.all_metrics[key].append(stress_metric[key])

            lyrics_ = torch.stack([lyrics, y[-1].cpu()])
            lyrics_mask_ = torch.stack([lyrics_mask, y_mask[-1].cpu()])
            rhyme_metric = compute_rhyme_metrics(lyrics_, lyrics_mask_)
            for key in rhyme_metric.keys():
                self.all_metrics[key].append(rhyme_metric[key])

            assert align_path[-1][0].item() == len(melody_ori) - 1
            assert align_path[-1][1].item() == sum(~lyrics_mask) - 1

            output_alignment(lyrics[~lyrics_mask].detach().cpu(), 
                             align_path, melody_ori.detach().cpu(), 
                             id_pairs[rank], self.path_save, seg_words, rank=k)
        
        truth_path = truth_path - truth_path[0]

        assert x[-1][~x_mask[-1]].size(0) == melody_ori.size(0)
        assert truth_path[-1][0].item() == len(melody_ori) - 1

        output_alignment(y[-1][~y_mask[-1]].detach().cpu(), 
                         truth_path.cpu().numpy(), melody_ori.detach().cpu(), 
                         [truth_id, truth_id], self.path_save, words[-1], rank=None)
        
        ## ssave melody information
        note_file = os.path.join(self.path_save, truth_id+'_notes.txt')
        save_notes(melody_ori, note_file)


    def on_test_epoch_end(self):

        ################ overall H@K ################
        self.propose_HitK = torch.tensor(self.propose_HitK)
        self.lengthInform_HitK = torch.tensor(self.lengthInform_HitK)

        HitK = [{"Hit@1": round(torch.sum(self.propose_HitK < 0.01).item() / len(self.propose_HitK), 4) * 100,
                 "Hit@3": round(torch.sum(self.propose_HitK < 0.03).item() / len(self.propose_HitK), 4) * 100,
                 "Hit@5": round(torch.sum(self.propose_HitK < 0.05).item() / len(self.propose_HitK), 4) * 100,
                 "Method": "proposed_seg"+str(self.trainer.datamodule.seg_lnum)
                },
                {"Hit@1": round(torch.sum(self.lengthInform_HitK < 0.01).item() / len(self.lengthInform_HitK), 4) * 100,
                 "Hit@3": round(torch.sum(self.lengthInform_HitK < 0.03).item() / len(self.lengthInform_HitK), 4) * 100,
                 "Hit@5": round(torch.sum(self.lengthInform_HitK < 0.05).item() / len(self.lengthInform_HitK), 4) * 100,
                 "Method": "lengthInform_seg"+str(self.trainer.datamodule.seg_lnum)
                }]

        ################ save H@K metrics in a csv file ################
        topHits = pd.DataFrame(HitK)
 
        topHits.to_csv('outputs/eval_HitK_mlmcal_noDataFilter_seg'+str(self.trainer.datamodule.seg_lnum)+
                        '_gamma'+ str(round(self.contrastive_loss.gamma.item(), 4))+
                        '_a' + str(round(self.contrastive_loss.regularization_weight.item(), 4))+
                        '.csv', index=False)


        ################ save other metrics in a csv file ################
        df = pd.DataFrame(self.all_metrics)

        df.to_csv('outputs/eval_metrics_mlmcal_seg'+str(self.trainer.datamodule.seg_lnum)+
                    '_top'+str(self.trainer.datamodule.metric_topk)+ 
                    '_gamma'+str(round(self.contrastive_loss.gamma.item(), 4))+
                    '_a' + str(round(self.contrastive_loss.regularization_weight.item(), 4))+
                    '.csv', index=False)
    


    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
            Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns: 
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        optimizer = self.optimizer_cls(self.parameters())

        if self.scheduler_cls is None:
            return {"optimizer": optimizer}

        else:
            scheduler = self.scheduler_cls(optimizer=optimizer, trainer=self.trainer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }