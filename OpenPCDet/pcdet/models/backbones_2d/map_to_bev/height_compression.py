import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        reduce_z = self.model_cfg.get('REDUCE_Z', None)
        if reduce_z is not None:
            reduce_z = str(reduce_z).lower()
            if reduce_z == 'max':
                spatial_features = spatial_features.max(dim=2).values
            elif reduce_z == 'sum':
                spatial_features = spatial_features.sum(dim=2)
            elif reduce_z == 'mean':
                spatial_features = spatial_features.mean(dim=2)
            else:
                raise ValueError(f'Unsupported HeightCompression REDUCE_Z={reduce_z}')
        else:
            spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
