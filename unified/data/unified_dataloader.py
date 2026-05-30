import os
import numpy as np
import random
from data.dali_data import TrainCollect
from utils.dist_utils import get_rank, get_world_size, dist_print


class UnifiedTrainCollect:
    """Multi-dataset data loader with weighted alternating batches.
    
    Streams data directly from DALI (no pre-fetching).
    When a smaller dataset is exhausted mid-epoch, its DALI iterator
    is reset inline (fast, since the pipeline is already built).
    
    Args:
        batch_size: Batch size for all datasets.
        num_threads: Number of DALI worker threads.
        dataset_configs: Dict of dataset configs from unified config.
    """
    def __init__(self, batch_size, num_threads, dataset_configs):
        self.loaders = {}
        self.weights = {}
        shard_id = get_rank()
        num_shards = get_world_size()

        # Create loaders one at a time (avoid simultaneous DALI pipeline init)
        for ds_name, ds_cfg in dataset_configs.items():
            row_anchor = self._compute_row_anchor(ds_name, ds_cfg)
            col_anchor = np.linspace(0, 1, ds_cfg['num_col'])
            list_path = os.path.join(ds_cfg['data_root'], ds_cfg['list_path'])

            dist_print(f'  [{ds_name}] Creating DALI pipeline...')

            self.loaders[ds_name] = TrainCollect(
                batch_size, num_threads,
                ds_cfg['data_root'], list_path,
                shard_id, num_shards,
                row_anchor, col_anchor,
                ds_cfg['train_width'], ds_cfg['train_height'],
                ds_cfg['num_cell_row'], ds_cfg['num_cell_col'],
                ds_name,
                ds_cfg['crop_ratio'],
            )
            self.weights[ds_name] = int(ds_cfg.get('sampling_weight', 1))
            dist_print(f'  [{ds_name}] ready: {len(self.loaders[ds_name])} batches, '
                       f'weight={self.weights[ds_name]}')

        # Build round-robin pattern from weights
        # e.g. CULane=1, Tusimple=5 → ['CULane','Tusimple','Tusimple','Tusimple','Tusimple','Tusimple']
        self.pattern = []
        for ds_name, w in self.weights.items():
            self.pattern.extend([ds_name] * w)

        # Total iters per epoch = largest_dataset_batches * pattern_length
        max_batches = max(len(loader) for loader in self.loaders.values())
        self.epoch_iters = max_batches * len(self.pattern) // max(self.weights.values(), default=1)
        # Simpler: just sum up actual contributions
        self.epoch_iters = sum(
            len(self.loaders[ds]) * self.weights[ds]
            for ds in self.loaders
        )

        self.iter_count = 0
        self.pattern_idx = 0

        dist_print(f'  Pattern: {self.pattern}')
        dist_print(f'  Epoch iterations: {self.epoch_iters}')

    def _compute_row_anchor(self, ds_name, ds_cfg):
        if ds_cfg.get('row_anchor_from_pixel', False):
            return np.linspace(
                ds_cfg['row_anchor_start'],
                ds_cfg['row_anchor_end'],
                ds_cfg['num_row']
            ) / ds_cfg['original_image_height']
        else:
            return np.linspace(
                ds_cfg['row_anchor_start'],
                ds_cfg['row_anchor_end'],
                ds_cfg['num_row']
            )

    def _next_from(self, ds_name):
        """Get next batch from a dataset, resetting if exhausted."""
        try:
            return next(self.loaders[ds_name])
        except StopIteration:
            # DALI reset is fast — pipeline is already built,
            # it just resets the data cursor and refills prefetch buffers
            self.loaders[ds_name].reset()
            return next(self.loaders[ds_name])

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count >= self.epoch_iters:
            raise StopIteration

        # Round-robin through the pattern
        ds_name = self.pattern[self.pattern_idx % len(self.pattern)]
        self.pattern_idx += 1
        self.iter_count += 1

        batch = self._next_from(ds_name)
        batch['dataset_name'] = ds_name
        return batch

    def __len__(self):
        return self.epoch_iters

    def reset(self):
        for loader in self.loaders.values():
            loader.reset()
        self.iter_count = 0
        self.pattern_idx = 0

    next = __next__
