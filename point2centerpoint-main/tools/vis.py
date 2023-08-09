import argparse

import numpy as np
import torch
import random

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils.show_result import show_det_result_meshlab

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/point2centerpoint.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str,
                        default='../output/kitti_models/point2centerpoint/default/V2/ckpt/checkpoint_epoch_67.pth',
                        help='specify the pretrained model')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--out_dir', type=str, default='../output/visualize')
    parser.add_argument('--thr', type=float, default=0.3)
    parser.add_argument('--index', type=str, default=None)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    np.random.seed(666)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Visualize-------------------------')

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            if args.index is not None and args.index != data_dict['frame_id'][0]:
                continue
            logger.info(f'Visualized sample index: \t{idx}')
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            show_det_result_meshlab(data_dict, pred_dicts, args.out_dir + '/' + str(data_dict['frame_id'][0]), args.thr)

    logger.info('Vis done.')


if __name__ == '__main__':
    main()
