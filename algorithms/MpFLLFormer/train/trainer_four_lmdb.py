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

import os
import shutil
import time

import numpy as np
import torch
from einops import rearrange
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather, reduce_by_weight, resample_3d, cal_dice, unSpatialPad

from monai.data import decollate_batch

from utils.utils import info_if_main, logger_info


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        images, labels = batch_data[0], batch_data[1]
        data = rearrange(images, "b n c w h d -> (b n) c w h d").to(args.rank, non_blocking=True).contiguous().float()
        target = rearrange(labels, "b n c w h d -> (b n) c w h d")[:, 2:3].to(args.rank, non_blocking=True).contiguous().long()

        data = (data - args.a_min) / (args.a_max - args.a_min)
        data = torch.clip(data, args.b_min, args.b_max)

        target[target == 1] = 0
        target[(target == 2) | (target == 3) | (target == 6) | (target == 9)] = 1
        target[target == 4] = 2
        target[(target == 5) | (target == 7)] = 3
        target[target > 7] = 1

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        learning_rate = optimizer.param_groups[0]['lr']
        info_if_main(
            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "lr:{:.6f}".format(learning_rate),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()

    return run_loss.avg


def val_epoch_v2(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    data_num, dice_sum = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            images, labels = batch_data[0], batch_data[1]
            if images.shape[1] != 4:
                print('pass')
                continue
            data = images.to(args.rank, non_blocking=True).contiguous().float()
            target = labels.to(args.rank, non_blocking=True)[:, 2:3].contiguous().long()

            data = (data - args.a_min) / (args.a_max - args.a_min)
            data = torch.clip(data, args.b_min, args.b_max)

            target[target == 1] = 0
            target[(target == 2) | (target == 3) | (target == 6) | (target == 9)] = 1
            target[target == 4] = 2
            target[(target == 5) | (target == 7)] = 3
            target[target > 7] = 1

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            val_outputs = torch.argmax(logits, axis=1)[0]  # b, c, h, w, d -> h, w, d
            val_outputs = unSpatialPad(val_outputs, data[0, 0]).detach().cpu().numpy().astype(np.uint8)
            val_labels = target.cpu().numpy()[0, 0, :, :, :]
            # val_outputs = torch.softmax(logits, 1).cpu().numpy()
            # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            # val_outputs = unSpatialPad(val_outputs, data.cpu().numpy()[0, 0])
            # val_labels = target.cpu().numpy()[0, 0, :, :, :]
            # target_shape = val_labels.shape
            # val_outputs = resample_3d(val_outputs, target_shape)

            dsc = cal_dice(val_outputs > 0, val_labels > 0)
            dice_sum += dsc
            data_num += 1
            
            if args.rank == 0:
                info_if_main(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    dsc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
        dist.barrier()
        avg_dice = reduce_by_weight(dice_sum/data_num, data_num)
    return avg_dice


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    info_if_main("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    best_acc=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        info_if_main("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = best_acc
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        info_if_main(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        # train_loss = 0
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        learning_rate = optimizer.param_groups[0]['lr']
        info_if_main(
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "lr:{:.6f}".format(learning_rate),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("lr", learning_rate, epoch)
        b_new_best = False
        if (epoch + 1 > args.val_start) and (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch_v2(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                info_if_main(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    info_if_main("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    info_if_main("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    info_if_main("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
