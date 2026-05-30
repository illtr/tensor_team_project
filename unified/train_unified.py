# 통합 학습용 코드 일단 Tusimple 형삭 데이터셋만 평가하도록 변경됨

import os
import torch, datetime

from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
import numpy as np


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    for b_idx, data_label in enumerate(progress_bar):
        if b_idx == 0:
            ds = data_label.get('dataset_name', '?')
            dist_print(f'[DEBUG] First batch received: dataset={ds}, '
                       f'images={data_label["images"].shape}')
        global_step = epoch * len(data_loader) + b_idx

        results = inference(net, data_label, 'Unified')

        loss = calc_loss(loss_dict, results, logger, global_step, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if hasattr(progress_bar, 'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k, v in kwargs.items():
                    if 'lane' in k:
                        continue
                    new_kwargs[k] = v
                # Show which dataset this batch came from
                ds_name = data_label.get('dataset_name', 'unknown')
                progress_bar.set_postfix(loss='%.3f' % float(loss),
                                         ds=ds_name,
                                         **new_kwargs)


def create_dataset_eval_cfg(cfg, dataset_name):
    ds_cfg = cfg.datasets[dataset_name]

    class EvalCfg:
        pass

    eval_cfg = EvalCfg()

    # Copy global settings
    eval_cfg.dataset = dataset_name
    eval_cfg.backbone = cfg.backbone
    eval_cfg.distributed = cfg.distributed
    eval_cfg.test_work_dir = cfg.test_work_dir
    eval_cfg.use_aux = cfg.use_aux

    # Copy dataset-specific settings
    eval_cfg.data_root = ds_cfg['data_root']
    eval_cfg.num_lanes = ds_cfg['num_lanes']
    eval_cfg.num_row = ds_cfg['num_row']
    eval_cfg.num_col = ds_cfg['num_col']
    eval_cfg.num_cell_row = ds_cfg['num_cell_row']
    eval_cfg.num_cell_col = ds_cfg['num_cell_col']
    eval_cfg.train_width = ds_cfg['train_width']
    eval_cfg.train_height = ds_cfg['train_height']
    eval_cfg.crop_ratio = ds_cfg['crop_ratio']

    # Compute row/col anchors
    if ds_cfg.get('row_anchor_from_pixel', False):
        eval_cfg.row_anchor = np.linspace(
            ds_cfg['row_anchor_start'], ds_cfg['row_anchor_end'],
            ds_cfg['num_row']) / ds_cfg['original_image_height']
    else:
        eval_cfg.row_anchor = np.linspace(
            ds_cfg['row_anchor_start'], ds_cfg['row_anchor_end'],
            ds_cfg['num_row'])
    eval_cfg.col_anchor = np.linspace(0, 1, ds_cfg['num_col'])

    # TuSimple-specific
    if dataset_name == 'Tusimple':
        eval_cfg.eval_mode = 'normal'

    # CULane-specific
    if dataset_name == 'CULane':
        eval_cfg.tta = getattr(cfg, 'tta', False)

    return eval_cfg


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)
        else:
            while not os.path.exists('.work_dir_tmp_file.txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()

    synchronize()
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed

    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')

    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start unified training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide', '34fca']

    # ── Build unified data loader ──
    dist_print('[DEBUG] Creating unified data loader...')
    train_loader = get_train_loader(cfg)
    dist_print(f'[DEBUG] Unified loader created. Total iterations per epoch: {len(train_loader)}')

    # ── Build unified model ──
    dist_print('[DEBUG] Creating unified model...')
    net = get_model(cfg)
    dist_print('[DEBUG] Model created.')

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    optimizer = get_optimizer(net, cfg)

    # ── Resume / Finetune ──
    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = 0
        #resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1 모델 저장 방식 문제
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(f'[DEBUG] Scheduler created. iters_per_epoch={len(train_loader)}')
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)

    max_res = 0
    res = None

    for epoch in range(resume_epoch, cfg.epoch):
        dist_print(f'\n=== Epoch {epoch}/{cfg.epoch} ===')

        train(net, train_loader, loss_dict, optimizer, scheduler,
              logger, epoch, metric_dict)
        dist_print(f'[DEBUG] Epoch {epoch} training done. Resetting loader...')
        train_loader.reset()
        dist_print(f'[DEBUG] Loader reset done.')

        # ── Evaluation: TuSimple only ──
        try:
            from model.model_unified import DatasetModelWrapper
            from evaluation.eval_wrapper import eval_lane

            eval_datasets = getattr(cfg, 'eval_datasets', ['Tusimple'])
            for ds_name in eval_datasets:
                if ds_name not in cfg.datasets:
                    continue
                dist_print(f'Evaluating on {ds_name}...')
                eval_cfg = create_dataset_eval_cfg(cfg, ds_name)

                if distributed:
                    wrapper = DatasetModelWrapper(net.module, ds_name)
                else:
                    wrapper = DatasetModelWrapper(net, ds_name)

                ds_res = eval_lane(wrapper, eval_cfg, ep=epoch, logger=logger)
                if ds_res is not None:
                    logger.add_scalar(f'Eval/{ds_name}', ds_res, global_step=epoch)
                    dist_print(f'  {ds_name} result: {ds_res:.4f}')

                    if ds_res > max_res:
                        max_res = ds_res
        except Exception as e:
            dist_print(f'Evaluation error (continuing training): {e}')

        # Save model every epoch
        save_model(net, optimizer, epoch, work_dir, distributed)

    logger.close()
