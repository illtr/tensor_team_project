import torch
import torch.nn as nn
from model.backbone import resnet
from model.seg_model import SegHead
from utils.common import initialize_weights


class DatasetHead(nn.Module):
    """Per-dataset output projection head.
    
    Takes the shared MLP trunk output (2048-dim) and projects to
    dataset-specific grid dimensions for lane detection.
    """
    def __init__(self, num_grid_row, num_cls_row, num_grid_col, num_cls_col,
                 num_lanes, mlp_mid_dim=2048):
        super(DatasetHead, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lanes = num_lanes

        self.dim1 = num_grid_row * num_cls_row * num_lanes
        self.dim2 = num_grid_col * num_cls_col * num_lanes
        self.dim3 = 2 * num_cls_row * num_lanes
        self.dim4 = 2 * num_cls_col * num_lanes
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4

        self.projection = nn.Linear(mlp_mid_dim, self.total_dim)
        initialize_weights(self.projection)

    def forward(self, fea):
        out = self.projection(fea)
        pred_dict = {
            'loc_row': out[:, :self.dim1].view(
                -1, self.num_grid_row, self.num_cls_row, self.num_lanes),
            'loc_col': out[:, self.dim1:self.dim1 + self.dim2].view(
                -1, self.num_grid_col, self.num_cls_col, self.num_lanes),
            'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(
                -1, 2, self.num_cls_row, self.num_lanes),
            'exist_col': out[:, -self.dim4:].view(
                -1, 2, self.num_cls_col, self.num_lanes),
        }
        return pred_dict


class UnifiedParsingNet(nn.Module):
    """Unified lane detection model for CULane + TuSimple.
    
    Architecture:
        Input → Shared Backbone (ResNet) → Shared Pool (Conv2d→8ch)
        → AdaptiveAvgPool2d(10, 50) → Shared MLP Trunk
        → Dataset-specific Head → Output
    
    The AdaptiveAvgPool2d normalizes different input resolutions
    (CULane: 1600x320, TuSimple: 800x320) to a fixed feature size.
    """
    def __init__(self, pretrained=True, backbone='18', dataset_configs=None,
                 use_aux=False):
        super(UnifiedParsingNet, self).__init__()
        self.use_aux = use_aux
        self.dataset_configs = dataset_configs

        # ── Shared Backbone ──
        self.model = resnet(backbone, pretrained=pretrained)

        # ── Shared Pool Layer ──
        backbone_channels = 512 if backbone in ['34', '18', '34fca'] else 2048
        self.pool = nn.Conv2d(backbone_channels, 8, 1)

        # ── Adaptive Pooling (resolves input resolution differences) ──
        # CULane: fea is (10, 50) → unchanged
        # TuSimple: fea is (10, 25) → upsampled to (10, 50)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 50))

        # ── Fixed input dimension after adaptive pooling ──
        self.input_dim = 10 * 50 * 8  # = 4,000
        mlp_mid_dim = 2048

        # ── Shared MLP Trunk ──
        self.cls_trunk = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, mlp_mid_dim),
            nn.ReLU(),
        )

        # ── Per-Dataset Output Heads ──
        self.heads = nn.ModuleDict()
        for ds_name, ds_cfg in dataset_configs.items():
            self.heads[ds_name] = DatasetHead(
                num_grid_row=ds_cfg['num_cell_row'],
                num_cls_row=ds_cfg['num_row'],
                num_grid_col=ds_cfg['num_cell_col'],
                num_cls_col=ds_cfg['num_col'],
                num_lanes=ds_cfg['num_lanes'],
                mlp_mid_dim=mlp_mid_dim,
            )

        # ── Optional Auxiliary Segmentation Head ──
        if self.use_aux:
            max_lanes = max(cfg['num_lanes'] for cfg in dataset_configs.values())
            self.seg_head = SegHead(backbone, max_lanes * 2)

        initialize_weights(self.cls_trunk)

    def forward(self, x, dataset_name):
        x2, x3, fea = self.model(x)

        if self.use_aux:
            seg_out = self.seg_head(x2, x3, fea)

        fea = self.pool(fea)
        fea = self.adaptive_pool(fea)
        fea = fea.view(-1, self.input_dim)
        fea = self.cls_trunk(fea)

        pred_dict = self.heads[dataset_name](fea)

        if self.use_aux:
            pred_dict['seg_out'] = seg_out

        return pred_dict

    def forward_tta(self, x, dataset_name):
        """Test-time augmentation with spatial shifts."""
        x2, x3, fea = self.model(x)

        pooled_fea = self.pool(fea)
        n, c, h, w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:, :, :, :w - 1] = pooled_fea[:, :, :, 1:]
        left_pooled_fea[:, :, :, -1] = pooled_fea.mean(-1)

        right_pooled_fea[:, :, :, 1:] = pooled_fea[:, :, :, :w - 1]
        right_pooled_fea[:, :, :, 0] = pooled_fea.mean(-1)

        up_pooled_fea[:, :, :h - 1, :] = pooled_fea[:, :, 1:, :]
        up_pooled_fea[:, :, -1, :] = pooled_fea.mean(-2)

        down_pooled_fea[:, :, 1:, :] = pooled_fea[:, :, :h - 1, :]
        down_pooled_fea[:, :, 0, :] = pooled_fea.mean(-2)

        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea,
                         up_pooled_fea, down_pooled_fea], dim=0)
        fea = self.adaptive_pool(fea)
        fea = fea.view(-1, self.input_dim)
        fea = self.cls_trunk(fea)

        return self.heads[dataset_name](fea)


class DatasetModelWrapper(nn.Module):
    """Wrapper that fixes dataset_name for compatibility with eval code.
    
    The existing eval_wrapper.py expects model.forward(x) without dataset_name.
    This wrapper binds the dataset_name so the unified model can be used
    with the unchanged eval pipeline.
    """
    def __init__(self, unified_net, dataset_name):
        super(DatasetModelWrapper, self).__init__()
        self.unified_net = unified_net
        self.dataset_name = dataset_name

    def forward(self, x):
        return self.unified_net(x, self.dataset_name)

    def forward_tta(self, x):
        return self.unified_net.forward_tta(x, self.dataset_name)


def get_model(cfg):
    dataset_configs = {}
    for ds_name, ds_cfg in cfg.datasets.items():
        dataset_configs[ds_name] = {
            'num_cell_row': ds_cfg['num_cell_row'],
            'num_row': ds_cfg['num_row'],
            'num_cell_col': ds_cfg['num_cell_col'],
            'num_col': ds_cfg['num_col'],
            'num_lanes': ds_cfg['num_lanes'],
        }
    return UnifiedParsingNet(
        pretrained=True,
        backbone=cfg.backbone,
        dataset_configs=dataset_configs,
        use_aux=cfg.use_aux,
    ).cuda()
