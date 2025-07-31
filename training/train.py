import argparse
import torch
import os

import sys

sys.path.append(os.path.dirname(os.getcwd()))

from training.trainer import do_train
from misc.utils import MinkLocParams
from datasets.dataset_utils import make_dataloaders

if __name__ == '__main__':
    # 直接在代码中设置参数，不需要命令行输入
    class Args:
        def __init__(self):
            # 根据你的需求修改这些路径
            self.config = '../config/config_chilean.txt'  # 配置文件路径
            self.model_config = '../config/model_config_chilean.txt'  # 模型配置文件路径
            self.ckpt = None  # 检查点文件路径，如果不需要从检查点恢复训练则设为None
            self.debug = False  # 是否开启调试模式
            self.visualize = False  # 是否开启可视化


    args = Args()

    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))
    print('Visualize: {}'.format(args.visualize))

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Available config files in config directory:")
        config_dir = os.path.dirname(args.config)
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.endswith('.txt'):
                    print(f"  - {f}")
        sys.exit(1)

    if not os.path.exists(args.model_config):
        print(f"Error: Model config file not found: {args.model_config}")
        print("Available model config files in config directory:")
        config_dir = os.path.dirname(args.model_config)
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.startswith('model_config') and f.endswith('.txt'):
                    print(f"  - {f}")
        sys.exit(1)

    params = MinkLocParams(args.config, args.model_config)

    # 手动设置数据集文件夹路径，指向生成pickle文件的目录
    params.dataset_folder = '/home/wzj/pan1/CASSPR/generating_queries'

    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    dataloaders = make_dataloaders(params, debug=args.debug)
    do_train(dataloaders, params, ckpt=args.ckpt, debug=args.debug, visualize=args.visualize)