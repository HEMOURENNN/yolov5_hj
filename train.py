# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    """
    Trains YOLOv5 model with given hyperparameters, options, and device, managing datasets, model architecture, loss
    computation, and optimizer steps.

    `hyp` argument is path/to/hyp.yaml or hyp dictionary.
    """

    '''====================================================1.传参/基本配置================================================================='''
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    # 和日志相关的回调函数，记录当前代码执行的阶段
    callbacks.run("on_pretrain_routine_start")

    # Directories，训练完成后的模型权重保存的文件夹
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"  # 会自动保存其中最优的模型，和训练到最后的一个模型

    # Hyperparameters，超参数，从yaml文件加载超参数信息
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings，保存运行时的参数配置
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")

        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots 是否需要画图： 所有的labels信息、迭代的epochs、训练结果等
    cuda = device.type != "cpu"
    # 初始化随机数种子
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 存在子进程-分布式训练
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]  # 训练集和验证集的位路径
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes，数据集有多少种类别
    # 类别对应的名称， 如果只有一个类别并且data_dict里没有names这个key的话，我们将names设置为["item"]代表目标
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    # 当前数据集是否是coco数据集(80个类别)
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    '''====================================================2.建模================================================================='''
    # Model
    # 检查文件后缀是否是.pt
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    # 加载预训练权重 yolov5提供了5个不同的预训练权重，大家可以根据自己的模型选择预训练权重
    if pretrained:
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(LOCAL_RANK):
            # 如果本地不存在就从网站上下载
            weights = attempt_download(weights)  # download if not found locally
        # 加载模型以及参数
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        '''
            这里加载模型有两种方式，一种是通过opt.cfg 另一种是通过ckpt['model'].yaml
            区别在于是否使用resume 如果使用resume会将opt.cfg设为空，按照ckpt['model'].yaml来创建模型
            这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
            原因: 保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor会自己覆盖自己设定的anchor
            详情参考: https://github.com/ultralytics/yolov5/issues/459
            所以下面设置intersect_dicts()就是忽略exclude
        '''
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        # 筛选字典中的键值对  把exclude删除
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 载入模型权重
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze 冻结权重层
    # 这里只是给了冻结权重层的一个例子, 但是作者并不建议冻结权重层, 训练全部层参数, 可以得到更好的性能, 不过也会更慢
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size，检查图片大小是否满足32的倍数
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    '''====================================================3.优化器选择================================================================='''
    # Optimizer
    '''
        假设：
            nbs = 64
            batchsize = 16
            accumulate = 64 / 16 = 4
        那么模型梯度累计accumulate次之后就更新一次模型，相当于使用更大batch_size
    '''
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 权重衰减参数
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler
    if opt.cos_lr:
        # 使用one cycle 学习率  https://arxiv.org/pdf/1803.09820.pdf
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        # 使用线性学习率
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA, 单卡训练: 使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume, 断点续训
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 可以提高多gpu训练的准确性，但会显著降低训练速度。它仅适用于多GPU DistributedDataParallel 训练。
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader, 训练集数据加载
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    # 获取标签中最大类别值，与类别数作比较，如果大于等于类别数则表示有问题
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0 验证集数据集加载
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            # 自适应anchor / anchor可以理解为程序预测的box
            # 根据k-mean算法聚类生成新的锚框
            if not opt.noautoanchor:
                '''
                    参数dataset代表的是训练集，hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数 anchor_t:4.0
                    当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor。
                    best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸 
                '''
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP：多机多卡
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    # number of detection layers (to scale hyps)，获取了模型中最后一个检测层的数量，这个数量通常代表着模型中用于检测目标的层数。
    nl = de_parallel(model).model[-1].nl
    # 对超参数中的 "box" 部分进行调整，乘以一个比例因子 3 / nl，以根据检测层的数量来缩放 "box" 部分的值。
    hyp["box"] *= 3 / nl  # scale to layers，
    # 对超参数中的 "cls" 部分进行调整，乘以一个比例因子 nc / 80 * 3 / nl，以根据类别数和检测层的数量来缩放 "cls" 部分的值。
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    # 对超参数中的 "obj" 部分进行调整，乘以一个比例因子 (imgsz / 640) ** 2 * 3 / nl，以根据图像尺寸和检测层的数量来缩放 "obj" 部分的值。
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing  # 标签平滑
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    # 获取热身迭代的次数
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 初始化maps(每个类别的map)和results
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 早停止，不更新结束训练
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )

    '''====================================================4.开始训练================================================================='''
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()

        # Update image weights (optional, single-GPU only)
        '''
           如果设置进行图片采样策略（每个类别的权重，频率高的权重小），
           则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
           通过random.choices生成图片索引indices从而进行采样
        '''
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            # labels_to_image_weights: 这个函数是利用每张图片真实gt框的真实标签labels和开始训练前通过 labels_to_class_weights函数
            # 得到的每个类别的权重得到数据集中每张图片对应的权重。
            # https://github.com/Oneflow-Inc/oneflow-yolo-doc/blob/master/docs/source_code_interpretation/utils/general_py.md#192-labels_to_image_weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses 初始化训练时打印的平均损失信息
        if RANK != -1:
            # DDP模式打乱数据，并且ddp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)  # 进度条，方便展示信息
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

        # 梯度清零
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup，预热训练（前nw次迭代）热身训练迭代的次数iteration范围[1:nw]  选取较小的accumulate，学习率以及momentum,慢慢的训练
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    """
                         bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                         其他的参数学习率从0增加到lr*lf(epoch).
                         lf为上面设置的余弦退火的衰减函数
                         动量momentum也从0.9慢慢变到hyp['momentum'](default=0.937)
                    """
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale 默认关闭
            # Multi-scale 多尺度训练   从[imgsz*0.5, imgsz*1.5+gs]间随机选取一个尺寸(32的倍数)作为当前batch的尺寸送入模型开始训练
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward / 前向传播
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # 模型反向传播accumulate次（iterations）后再根据累计的梯度更新一次参数
            if ni - last_opt_step >= accumulate:
                # unscale gradients，将梯度按比例还原回原始值，这是由于使用了混合精度训练（mixed precision training），
                # 在计算梯度时采用了较小的精度以提高训练速度，但在更新参数时需要将梯度还原到原始精度。
                scaler.unscale_(optimizer)
                # 对梯度进行裁剪，限制梯度的范数不超过指定的阈值，这有助于防止梯度爆炸的问题。
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                # 使用混合精度训练中的 scaler 对优化器进行更新步骤，即执行一步优化器的参数更新。
                scaler.step(optimizer)  # optimizer.step
                # 更新 scaler 的状态，以便下一次迭代使用
                scaler.update()
                # 清空之前的梯度，以便进行下一次的梯度计算。
                optimizer.zero_grad()
                if ema:
                    # 如果启用了指数移动平均（EMA）技术，这行代码将用于更新模型的参数以更新移动平均参数。
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler 进行学习率衰减
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            # 将model中的属性赋值给ema
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            # 判断当前的epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # notest: 是否只测试最后一轮  True: 只测试最后一轮   False: 每轮训练完都测试mAP
            if not noval or final_epoch:  # Calculate mAP
                '''
                    测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                    results: [1] Precision 所有类别的平均precision(最大f1时)
                             [1] Recall 所有类别的平均recall
                             [1] map@0.5 所有类别的平均mAP@0.5
                             [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                             [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                    maps: [80] 所有类别的mAP@0.5:0.95
                '''
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Update best mAP
            # fi 是我们寻求最大化的值。在YOLOv5中，fitness函数实现对 [P, R, mAP@.5, mAP@.5-.95] 指标进行加权。
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model
            '''
                - 保存带checkpoint的模型用于inference或resuming training
                - 保存模型, 还保存了epoch, results, optimizer等信息
                - optimizer将不会在最后一轮完成后保存
                - model保存的是EMA的模型
            '''

            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    '''====================================================5.打印训练信息================================================================='''
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    # weights 权重的路径./weights/yolov5s.pt....
    # yolov5提供4个不同深度不同宽度的预训练权重 用户可以根据自己的需求选择下载
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    # cfg 配置文件（网络结构） anchor/backbone/numclasses/head，训练自己的数据集需要自己生成
    # 生成方式——例如我的yolov5s_mchar.yaml 根据自己的需求选择复制./models/下面.yaml文件，5个文件的区别在于模型的深度和宽度依次递增
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    # data 数据集配置文件（路径） train/val/label/， 该文件需要自己生成
    # 训练集和验证集的路径 + 类别数 + 类别名称
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    # hpy超参数设置文件（lr/sgd/mixup）./data/hyps/下面有5个超参数设置文件，每个文件的超参数初始值有细微区别，用户可以根据自己的需求选择其中一个
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    # epochs 训练轮次， 默认轮次为100次
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    # batchsize 训练批次， 默认bs=16
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    # imagesize 设置图片大小, 默认640*640
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    # rect 是否采用矩形训练，默认为False
    '''
    正方形训练：
        正方形训练可以统一所有图片的大小，从而训练方便，但是问题在填充图片的过程中，我们引入了很多冗余信息。为了处理这个问题，yolov3提出使用矩形训练。
    矩阵训练：
        - 原来图片的长边还是填充到最大长度，但是短边只填充到32的倍数。
        - 这样处理过后可以引入较少的冗余信息。加快训练速度。
        但是引入了其他问题，第一就是图片集的大小不一样，yolov3的处理是将尺寸接近的放到一起处理，这就导致不能使用dataloader中的shuffle功能。
    '''
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    # resume 是否接着上次的训练结果，继续训练
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    # nosave 不保存模型  默认False(保存)  在./runs/exp*/train/weights/保存两个模型 一个是最后一次的模型 一个是最好的模型
    # best.pt/ last.pt 不建议运行代码添加 --nosave
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    # noval 最后进行测试, 设置了之后就是训练结束都测试一下， 不设置则每轮都计算mAP, 建议不设置
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    # noautoanchor 不自动调整anchor, 默认False, 自动调整anchor
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    # 使用超参数优化算法进行自动调参，默认关闭
    # yolov5采用遗传算法对超参数进行优化，寻找一组最优的训练超参数。开启后传入参数n，训练每迭代n次进行一次超参数进化;开启后不传入参数，则默认为const=300。
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    # 配合上面自动使用，超参数优化的保存位置
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    # 断点训练+超参数优化。超参数优化会导致参数改变，如果采用断点训练的话，则需要指定参数的路径
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    # bucket谷歌优盘 / 一般用不到
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    # cache 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    '''
    image-weights 用于解决类别不平衡问题:
        1、读取训练样本中的GT，保存为一个列表；
        2、计算训练样本列表中不同类别个数，然后给每个类别按相应目标框数的倒数赋值，数目越多的种类权重越小，形成按种类的分布直方图；
        3、对于训练数据列表，训练时按照类别权重筛选出每类的图像作为训练数据。使用random.choice(population, weights=None, *, cum_weights=None, k=1)
        更改训练图像索引，可达到样本均衡的效果。
    '''
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 多尺度训练，这里默认关闭
    # 这个开启的话，对网络的精度会有提升，根据情况选择是否开启
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    '''
    single-cls 单类别训练：
        - 单类别训练指的是在训练模型时只针对单个物体类别进行训练。通常情况下，目标检测模型需要能够识别和定位多个不同类别的物体，
        但有时候我们可能只对某个特定类别感兴趣，或者数据集中只包含单个类别的物体。

        - 在单类别训练中，模型的输出将会被限制为检测单个类别的物体，这样可以简化模型的训练过程并提高模型在特定类别上的性能。
        通过单类别训练，模型可以更好地专注于学习如何检测和定位特定类别的物体，从而提高检测准确性。
    '''
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    # 优化器选择 / 提供了三种优化器
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    '''
    sync-bn 同步批量归一化
        - 在传统的批归一化(Batch Normalization，简称 BN)中，每个 GPU 会对数据的均值和方差进行单独计算
        - 在多 GPU 训练时，每个 GPU 计算的均值和方差可能会不同，导致模型训练不稳定.
        - 为了解决这个问题，SyncBN 技术将 BN 的计算放在了整个分布式训练过程中进行，确保所有 GPU 上计算的均值和方差是一致的，从而提高模型训练的稳定性和效果，
        但同时也会增加训练时间和硬件要求，因此需要根据具体的训练数据和硬件资源来决定是否使用SynCBN.
    '''
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    '''
    workers 线程数
        - DataLoader中的num workers参数，默认为8
        - Dataloader中oumworKers表示加载处理数据使用的线程数，使用多线程加载数据时，每个线程会负青加载和处理一批数据，数据加载处理完成后，会送入相应的队列中，
        最后主线程会从队列中读取数据，并送入GPU中进行模型计算numworkers为0表示不使用多线程，仅使用主线程进行数据加载和处理。
    '''
    parser.add_argument("--workers", type=int, default=0, help="max dataloader workers (per RANK in DDP mode)")
    '''
    project 保存路径
        - 和detect 里面的参数一样，即每次训练结果保存的主路径
        - 主路径:每次训练会生成一个单独的子文件夹，主路径就是存放这些单独子文件夹的地方，可以自己命名例如'runs/train'。
        比如说第一次训练保存结果的文件夹是exp1，第二次是exp2，第三次是exp3，则这些子文件夹都会放在主路径'runs/train'下面。
    '''
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    # project 参数介绍，里面的子文件夹
    parser.add_argument("--name", default="exp", help="save to project/name")
    # 是否覆盖已有文件夹。默认关闭。每次训练都会生成一个子文件夹，例如exp1，exp2，以此类推开启的话，新生成的就会直接覆盖之前的训练结果
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # quad dataloader 是一种数据加载器，它可以并行地从磁盘读取和处理多个图像，并将它们打包成四张图像，从而减少了数据读取和预处理的时间，并提高了数据加载的效率
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    # cos-lr 余弦学习率，使用cos衰减学习率
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    # 标签平滑策略，--label-smoothing 0.1, 表示在每个标签的真实概率上添加一个ε=0.1的噪声，从而使模型对标签的波动更加鲁棒;
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    # patience 参数指定为整数n时，表示模型在训练时，若连续n个epoch验证精度都没有提升，则认为训练已经过拟合，停止训练
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    # freeze 冻结网络。网络共有10层，迁移学习时，可以冻结一部分参数，只训练后面的层达到加快训练的目的指定n，冻结前n(0<n<=10)层参数
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    # save-period 固定周期保存权重。默认关闭，只保留最好和最后的网络权重。开启后，会根据设定的数值，每隔这个数值的epochs就会保留一次网络权重
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    # seed，随机种子，保证结果复现
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    '''
    local_rank 分布式训练
        python train.py --local rank 1.2
        这样，第一个进程将使用第2号 GPU，第二个进程将使用第3号GPU
        注意，如果使用了--ocal rank 参数，那么在启动训练脚本时需要使用 PyTorch 的分布式训练工具，例如 torch.distributed.launch,
    '''
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    # 在线可视化工具wandb，类似于tensorboard工具，想了解这款工具可以查看https://zhuanlan.zhihu.com/p/266337608
    parser.add_argument("--entity", default=None, help="Entity")
    # upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    # 边框图像记录间隔，训练过程中，每隔多少个epoch记录一次带有边界框的图片，类似于训练过程数据的可视化
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    '''
        artifact alias 数据集工件版本别名
            用于指定要使用的数据集工件的版本别名。
            命令行使用方法:python train.py--artifact alias latest
            在使用MLFlow等工县跟踪模型训练和数据集版本时，会给每个版本分配唯一的别名。通过指定此参数，可以使用特定版本的教据集工件。默认情况下，使用最新版本的数据集工件。
    '''
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    # 将ndjson 输出在控制台
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    # 将ndjson 记录在文件中
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    '''====================================================1.检查版本及依赖包安装================================================================='''
    """Runs training or hyperparameter evolution with specified options and optional callbacks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    '''====================================================2.重新初始化================================================================='''
    # Resume (from specified or most recent last.pt)
    # 初始化可视化工具wandb,wandb使用教程看：https://zhuanlan.zhihu.com/p/266337608
    # 断点训练使用教程可以查看：https://blog.csdn.net/CharmsLUO/article/details/123410081
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        # 相关的opt参数也要替换成last.pt中的opt参数 safe_load()yaml文件加载数据
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        # argparse.Namespace 可以理解为字典
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        # 不使用断点训练就在加载输入的参数
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        # 根据opt.project生成目录  如: runs/train/exp18
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    '''====================================================3.DDP model（分布式训练）================================================================='''
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 多卡训练GPU
    # 并且当且仅当使用cuda时并且有多块gpu时可以使用ddp模式，否则抛出报错信息。batch_size需要整除总的进程数量。
    # 另外DDP模式不支持AutoBatch功能，使用DDP模式必须手动指定batch size。
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        # 初始化多进程
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # Train
    if not opt.evolve:
        # 如果不进行超参进化 那么就直接调用train()函数，开始训练
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    # 否则使用超参进化算法(遗传算法) 求出最佳超参 再进行训练
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        # 超参进化列表 (突变规模, 最小值, 最大值)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mixup (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),
        }  # segment copy-paste (probability)

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        with open(opt.hyp, errors="ignore") as f:  # 载入初始超参
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        # evolve_yaml 超参进化后文件保存地址
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        initial_values = []

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        """
                使用遗传算法进行参数进化 默认是进化300代
                这里的进化算法原理为：根据之前训练时的hyp来确定一个base hyp再进行突变，具体是通过之前每次进化得到的results来确定之前每个hyp的权重，
            有了每个hyp和每个hyp的权重之后有两种进化方式；
                1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
                2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
                
                - evolve.txt会记录每次进化之后的results+hyp
                - 每次进化时，hyp会根据之前的results进行从大到小的排序；
                - 再根据fitness函数计算之前每次进化得到的hyp的权重
                    (其中fitness是我们寻求最大化的值。在YOLOv5中，fitness函数实现对 [P, R, mAP@.5, mAP@.5-.95] 指标进行加权。)
                - 再确定哪一种进化方式，从而进行进化。
                - 这部分代码其实不是很重要并且也比较难理解，因为正常训练也不会用到超参数进化。
        """
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


def generate_individual(input_ranges, individual_length):
    """Generates a list of random values within specified input ranges for each gene in the individual."""
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def run(**kwargs):
    """
    Executes YOLOv5 training with given options, overriding with any kwargs provided.

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
