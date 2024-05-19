# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys

sys.path.append('/data4/liruocheng/code/ycy_SwinUNETR/')
import os
from functools import partial

from lmdb_data_no_monai import transforms_v2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, WarmupExponentialLR
from algorithms.MpFLLFormer.network.res_swin_unetr_skipadd_region_fusion import SwinUNETR_Four
from algorithms.MpFLLFormer.train.trainer_four_lmdb import run_training

from dataset.datasets import get_loader_lmdb

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from algorithms.SwinUNETR.network.swin_unetr_mulitphase_2fusion import SwinUNETR_Four
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from utils.utils import info_if_main, load_model, logger_info
from loguru import logger

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="/data4/MpFll_four/", type=str,
                    help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", action="store_true", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=600, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size(window size)")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--val_start", default=0, type=int, help="number of epoch to start validation")
parser.add_argument("--distributed", action="store_true", default=True, help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=6, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.7, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")

parser.add_argument('--train_dir', default='/data4/liruocheng/dataset/all_dataset_spacing1.5_1.5_2_lmdb_from_package')
parser.add_argument('--val_dir', default='/data4/liruocheng/dataset/ts2.2_spacing1.5_1.5_2_lmdb_from_package')
parser.add_argument('--checkpoint_path', default='/data4/MpFll_four/')
parser.add_argument('--json_path', default='/data4/liruocheng/dataset/tr6.json')  # not exist
parser.add_argument('--num_multiphase', type=int, default=4)  # 单期数据输入1，多期数据输入期数
parser.add_argument('--num_sample', type=int, default=2)  # 单期数据输入1，多期数据输入期数
parser.add_argument("--phase_id", default=[0, 1, 2, 3], type=int, help="validation frequency")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger.add(os.path.join(args.logdir, 'train_log.txt'))
    info_if_main('args:', args)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        info_if_main(f"Found total gpus {args.ngpus_per_node}")
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    logger.add(os.path.join(args.logdir, 'train_log.txt'))
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    train_transforms_my = transforms_v2.Compose_my([
        # transforms_v2.Normalize(),
        # transforms_v2.RandomContrast(np.random.RandomState(47)),
        transforms_v2.RandomCrop_xy(output_size=[args.roi_x, args.roi_y, args.roi_z], num_sample=args.num_sample,
                                    random_state=np.random.RandomState(47)),
        # transforms_v2.RandomCrop_xy_ratio(output_size=[args.roi_x, args.roi_y, args.roi_z], num_sample=args.num_sample, pos_ratio=2, neg_ratio=1,
        #            random_state=np.random.RandomState(47)),
        transforms_v2.RandomFlip(random_state=np.random.RandomState(48), type='3D', axis_prob=0.2),
        transforms_v2.RandomRotate90(random_state=np.random.RandomState(49), type='3D', execution_probability=0.2)
    ])
    loader = get_loader_lmdb(args, train_transforms_my=train_transforms_my)
    logger_info(f'{args.rank} gpu {args.gpu}')
    info_if_main(f"Batch size is: {args.batch_size} epochs: {args.max_epochs}")
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    pretrained_dir = args.pretrained_dir
    model = SwinUNETR_Four(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
    )

    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict, strict=False)
        info_if_main("Use pretrained weights")

    if args.use_ssl_pretrained:
        try:
            ssl_pretrained_path = "./pretrained_models/model_swinvit.pt"
            load_model(model, ssl_pretrained_path)
            # model_dict = torch.load("./pretrained_models/model_swinvit.pt")
            # state_dict = model_dict["state_dict"]
            # # fix potential differences in state dict keys from pre-training to
            # # fine-tuning
            # if "module." in list(state_dict.keys())[0]:
            #     info_if_main("Tag 'module.' found in state dict - fixing!")
            #     for key in list(state_dict.keys()):
            #         state_dict[key.replace("module.", "")] = state_dict.pop(key)
            # if "swin_vit" in list(state_dict.keys())[0]:
            #     info_if_main("Tag 'swin_vit' found in state dict - fixing!")
            #     for key in list(state_dict.keys()):
            #         state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # # We now load model weights, setting param `strict` to False, i.e.:
            # # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # # the decoder weights untouched (CNN UNet decoder).
            # model.load_state_dict(state_dict, strict=False)
            # info_if_main("Using pretrained self-supervised Swin UNETR backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
            include_background=False
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    # 用于validation，已弃用
    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True,
                          ignore_empty=False)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
        mode="gaussian"
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger_info(f"Total parameters count {pytorch_total_params}")

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        info_if_main("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    scheduler = None
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    elif args.lrschedule == "exponential":
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.993)
        scheduler = WarmupExponentialLR(optimizer, gamma=0.992, warmup_epochs=50)
    if args.checkpoint is not None:
        scheduler.step(epoch=start_epoch)

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_acc=best_acc,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()