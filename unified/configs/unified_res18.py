dataset = 'Unified'

# ── Training Hyperparameters ──
epoch = 10
batch_size = 16
optimizer = 'SGD'
learning_rate = 0.01
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695
backbone = '18'
use_aux = False

# ── Loss Weights ──
sim_loss_w = 0.0
shp_loss_w = 0.0
mean_loss_w = 0.05
var_loss_power = 2.0

# ── Logging ──
note = ''
log_path = ''
auto_backup = True

# ── Resume / Finetune ──
finetune = None
resume = None
test_model = ''
test_work_dir = ''

# ── Per-Dataset Configurations ──
datasets = {
    'CULane': {
        'data_root': '',           
        'list_path': 'list/train_gt.txt',
        'num_lanes': 4,
        'num_row': 72,
        'num_col': 81,
        'num_cell_row': 200,
        'num_cell_col': 100,
        'train_width': 1600,
        'train_height': 320,
        'original_image_width': 1640,
        'original_image_height': 590,
        'crop_ratio': 0.6,
        'row_anchor_start': 0.42,
        'row_anchor_end': 1.0,
        'row_anchor_from_pixel': False,
        'sampling_weight': 0,
    },
    'Tusimple': {
        'data_root': '',           
        'list_path': 'train_gt.txt',
        'num_lanes': 4,
        'num_row': 56,
        'num_col': 41,
        'num_cell_row': 100,
        'num_cell_col': 100,
        'train_width': 800,
        'train_height': 320,
        'original_image_width': 1280,
        'original_image_height': 720,
        'crop_ratio': 0.8,
        'row_anchor_start': 160,
        'row_anchor_end': 710,
        'row_anchor_from_pixel': True,  # anchor is in pixel coords / 720
        'sampling_weight': 10,           # 5x oversampling to balance with CULane
    },
}
