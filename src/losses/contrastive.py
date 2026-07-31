import logging

import  torch
from torch import nn

from src.models import soft_dtw_cuda

log = logging.getLogger(__name__)


class Contrastive_loss(nn.Module):

    def __init__(self, 
                gamma: float=5.0,
                temperature: float=1.0,
                contrastive_setting: str="shuffle_negative",
                regularization_weight: float=0.5,
                cross_ratio: float=0.0, 
                shuffle_ratio: float=1.0, 
                sdtw_step_weight: str="equal", 
                 ):
        """
        Contrastive loss with Soft-DTW distance.
        
        Args:
            gamma: gamma value for Soft-DTW.
            temperature: temperature for contrastive loss.
            contrastive_setting (str): "collapse_sequence", "withinbatch_negative", or "shuffle_negative"
            regularization_weight: 
            cross_ratio: ratio of cross-modal pairs in the batch
            shuffle_ratio: ratio of lyrics to be shuffled in a segment
            sdtw_step_weight: "equal" or "lenInform"
        """

        super(Contrastive_loss, self).__init__()

        self.contrastive_setting = contrastive_setting
        self.cross_ratio = cross_ratio
        self.shuffle_ratio = shuffle_ratio
        self.sdtw = soft_dtw_cuda.SoftDTW(use_cuda=True, gamma=gamma,
                            dist_func='cosine_sim', 
                            sdtw_step_weight=sdtw_step_weight,
                            regularization_weight=regularization_weight)

        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("regularization_weight", torch.tensor(regularization_weight))

    def forward(self, x, y, x_mask, y_mask, b):

        D_m = self.sdtw_compute(x, y, x_mask, y_mask)
        D_m = D_m.reshape(b, -1)
        D_m = (D_m - D_m.mean(-1, keepdim=True)) / (D_m.std(-1, keepdim=True) + 1e-6)
        D_posm = D_m[:, 0]

        pos_m = - D_posm / self.temperature
        all_m = - D_m / self.temperature

        contrast_m = - torch.mean(pos_m -  torch.logsumexp(all_m, 1)) 

        D_negm = torch.flatten(D_m[:, 1:])

        return  contrast_m, D_posm.detach(), D_negm.detach()
    
    def sdtw_compute(self, x, y, x_mask, y_mask):

        b, n, d = x.shape
        m = y.size(1)

        D_mask = torch.zeros(b, n, m, device=x.device).type(torch.bool)

        for k in range(b):
            D_mask[k, x_mask[k], :] = True
            D_mask[k, :, y_mask[k]] = True
            
        D_sdtw = self.sdtw(x, y, ~D_mask)

        return D_sdtw