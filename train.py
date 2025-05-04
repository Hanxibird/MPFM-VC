# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('./')
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import torch
#import deepspeed  # 因为不使用deepspeed相关功能，这里可以注释掉
import pdb
from hyperpyyaml import load_hyperpyyaml

# 原本用于分布式训练错误记录相关的装饰器，在单机单卡环境可以注释掉
# from torch.distributed.elastic.multiprocessing.errors import record

from MPFM.utils.executor_single import Executor
from MPFM.utils.train_utils_single import (
    # init_distributed,  # 注释掉分布式环境初始化函数调用
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)
import os

# 以下设置分布式进程相关环境变量的代码在单机单卡非分布式环境不需要，注释掉
# os.environ['RANK'] = str(dist.get_rank())
# os.environ['LOCAL_RANK'] = str(dist.get_rank())
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,6'


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    # 以下是和deepspeed相关的保存状态参数，在单机单卡非分布式且不用deepspeed时可以保留也可以删除，这里保留它
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of MPFM_join.')
    # 因为不使用deepspeed，这里添加配置参数的操作也可以注释掉，不过保留也不会影响单机单卡运行，这里注释掉
    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def load_model(model, saved_state_dict):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        try:
            if state_dict[k].shape == saved_state_dict[k].shape:
                # print('yes ', k)
                new_state_dict[k] = saved_state_dict[k]
            else:
                print('no ', k)
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    return model


# 原本的主函数使用了分布式相关的装饰器，在单机单卡非分布式环境可以去掉装饰器
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    override_dict = {k: None for k in ['flow'] if k!= args.model}
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    # 去除分布式环境初始化相关调用
    # init_distributed(args)

    # 获取数据集和数据加载器，这部分基本不用变动，正常执行获取数据相关逻辑
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs)
    # 进行一些配置检查并保存配置，这部分逻辑正常执行
    configs = check_modify_and_save_config(args, configs)
    # 初始化TensorBoard的记录器，正常执行用于记录训练相关信息
    writer = init_summarywriter(args)
    # 加载模型，按原逻辑进行
    model = configs[args.model]
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        load_model(model, checkpoint)

    # 将模型转移到GPU（如果有可用GPU的话），正常执行设备转移逻辑
    model = model.cuda()

    # 获取优化器和学习率调度器，按原逻辑执行相关初始化操作
    model, optimizer, scheduler = init_optimizer_and_scheduler(args, configs, model)

    # 保存初始检查点相关逻辑，这里可以根据实际需求决定是否保留保存初始检查点的代码，这里保留但不执行保存操作示例
    info_dict = deepcopy(configs['train_conf'])
    # save_model(model, 'init', info_dict)

    # 获取执行器实例，正常执行获取操作
    executor = Executor()

    # 训练循环，去除分布式相关的同步、进程组操作等代码，只保留核心的每个epoch训练逻辑
    for epoch in range(info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        # 去除分布式相关的同步屏障代码
        # dist.barrier()
        # 去除分布式进程组相关创建和销毁代码
        #group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict,0)
        # dist.destroy_process_group(group_join)


if __name__ == '__main__':
    main()