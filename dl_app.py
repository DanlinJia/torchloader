from numpy.lib import type_check
# from tensorflow.python.keras.engine.base_layer import default
from builtins import help, print
import argparse
import os
from posixpath import join
import random
from torch.distributed.distributed_c10d import destroy_process_group
import yaml
import numpy as np
from shutil import copyfile
# from subprocess import PIPE
import time, datetime
from time import strftime

import logging
import warnings
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.multiprocessing as mp
from multiprocessing import Process, Pipe, Lock
import threading

# from models.vanilla_resnet_slow import *
# resnet34S = resnet34
# resnet18S = resnet18
# from models.vanilla_resnet import *
# from models.small_model import Small_Net, Small_Net_2
# from models.resnet20_imagenet import resnet20
# from models.vgg_grasp import vgg11, vgg13, vgg16, vgg19
from set_dataset import *
from models.vanilla_resnet_slow import resnet18S, resnet34S
from models.vanilla_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg_grasp import *
from models.small_model import *
from models.resnet20_imagenet import resnet20
from models.builder import build_model

import torch.autograd.profiler as profiler


def conn_message(mesg_type, mesg_data=None):
    return {'time': str(datetime.datetime.now()), 'type': mesg_type, 'msg': mesg_data}

def print_app_log(app, log_str):
    print("{}, worker{}, app {}: {}".format(datetime.datetime.now(), app.node_id, app.appid, log_str) )

class application():
    def __init__(self, appid, arch, depth, batch, workers, work_space, master, port, \
                arrival_time=0, cuda_device=[0, 1], start_iter=0, start_epoch=1, \
                end_iter=500, end_epoch=1, node_size=1, world_size=1, base_rank=0):
        self.appid = appid
        self.arch = arch
        self.depth = depth
        self.batch = batch
        self.workers = workers
        self.work_space = work_space
        self.master = master
        self.port = port
        self.arrival_time = arrival_time
        self.cuda_device = cuda_device
        self.device_num = world_size
        self.process = None
        self.start_iter = start_iter
        self.start_epoch = start_epoch
        self.end_iter = end_iter
        self.end_epoch = end_epoch
        self.node_size = node_size
        self.world_size = world_size
        self.base_rank = base_rank
        self.checkpoint = False
        self.finish_time = -1
        self.subprocess_conns = {}
        self.paused_counter = 0
        self.finished_counter = 0
        self.node_id = 0
        # self.logger = logging.getLogger("{}.log".format(self.appid))
        # self.parse_args()
        # self.best_acc1 = 0

    def print_info(self):
        print("appid: {}, model: {}{}, batch: {}, workers: {}, output: {}, port: {}, cuda_device: {}, world_size: {}, node_size: {}, base_rank: {}" \
            .format(self.appid, self.arch, self.depth, self.batch, self.workers, self.work_space, self.port, self.cuda_device, self.world_size, self.node_size, self.base_rank))

    def parse_args(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_names = sorted(name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))

        parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        parser.add_argument('--data', default='/work/yanzhi_group/datasets/imagenet/',
                            help='path to dataset')
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            # choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet18)')
        parser.add_argument('--depth', default=None, type=int,
                            help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
        parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 8)')
        parser.add_argument('--iteration', default=500, type=int, )
        parser.add_argument('--epochs', default=1, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                            help='initial learning rate', dest='lr')
        parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
                            help='how many every epoch before lr drop (default: 30)')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                            help='weight decay (default: 1e-4)', dest='weight_decay')
        parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                            help='print frequency (default: 10)')
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        parser.add_argument('--optimizer', type=str, default='sgd',
                            help='define optimizer')
        parser.add_argument('--lr-scheduler', type=str, default='default',
                            help='define lr scheduler')

        parser.add_argument('--trace', type=str, default='trace',
                            help='plot trace')
        parser.add_argument('--profile', action='store_true',
                            help='Use profiler')
        parser.add_argument('--profile-epochs', default=0, type=int,
                            help='Use profiler')
        parser.add_argument('--profile-name', type=str, default='test',
                            help='name of profile test')
        parser.add_argument('--record-shapes', action='store_true',
                            help='define lr scheduler')
        parser.add_argument('--profile-memory', action='store_true',
                            help='Use profiler')
        parser.add_argument('--use-cuda', action='store_true',
                            help='define lr scheduler')
        parser.add_argument('--profile-batches', default=-1, type=int,
                            help='Use profiler')
        parser.add_argument('--appid', default=1, type=int, help="Application Unique ID")
        parser.add_argument('--dataset', type=str, default='imagenet',
                            help='define dataset')
        parser.add_argument('--checkpoint', dest='checkpoint', action='store_false',
                            help="if resume frome a checkpoint")
        parser.add_argument('--start_iteration', default=0, type=int)
        parser.add_argument("--mode", "-m")

        args = parser.parse_args()
        args.appid = self.appid
        args.arch = self.arch
        args.depth = self.depth
        args.batch_size = self.batch
        args.workers = self.workers
        args.work_space = self.work_space
        args.dist_url = 'tcp://{}:{}'.format(self.master, self.port)
        args.multiprocessing_distributed = True
        args.iteration = self.end_iter
        args.epochs = self.end_epoch
        args.checkpoint = self.checkpoint
        args.start_iteration = self.start_iter
        args.cuda_device = self.cuda_device
        args.print_freq = 100
        args.world_size = self.world_size
        return args
    
    def attach_node(self, node_id):
        self.node_id = node_id         

def send_pause_to_subprocess(app):
    # the number of subprocess of app
    sub_procs = len(app.subprocess_conns)
    try:
        if sub_procs > 1:
            # pause subprocess 1 first, which needs to save the model
            model_save_conn, _ = app.subprocess_conns[1]
            
            # send Pause to all subprocesses to inform that the Pause event starts.
            for sub_idx in app.subprocess_conns:
                master_conn, _ = app.subprocess_conns[sub_idx]
                print_app_log(app, "{} send pause signal to {}th subprocess".format(master_conn, sub_idx))
                master_conn.send(conn_message("Pause"))
            
        #     # waiting for checkpoint saved
        #     if model_save_conn.recv():
        #         app.paused_counter += 1
            
        #     # enable other subprocesses to
        #     for sub_idx in app.subprocess_conns:
        #         if sub_idx > 1:
        #             master_conn, _ = app.subprocess_conns[sub_idx]
        #             master_conn.send(conn_message("Pause"))
        # sub_idx = 0
        # master_conn, _ = app.subprocess_conns[sub_idx]
        # print_app_log(app, "send pause signal to main subprocess ")
        # master_conn.send(conn_message("Pause"))
    except Exception as e:
            print("send pause to {}".format(sub_idx), e)

def send_exit_to_subprocess(app):
    try:
        for sub_idx in app.subprocess_conns:
            master_conn, _ = app.subprocess_conns[sub_idx]
            master_conn.send(conn_message("Exit"))
    except Exception as e:
            print("send exit to {}".format(sub_idx), e)

def send_exit_to_rank0(app):
    try:
        master_conn, _ = app.subprocess_conns[0]
        master_conn.send(conn_message("Exit"))
    except Exception as e:
        print("send exit to rank0", e)

def app_messager(app, app_conn):
    while 1:
        if app_conn.poll():
            ws_event = app_conn.recv()
            if ws_event['type'] == "Pause":
                print_app_log(app, "recevives a Pause signal from app_master.")
                t = threading.Thread(target=send_pause_to_subprocess, args=(app, ), name="pause subprocess", daemon=True)
                t.start()
                # send_pause_to_subprocess(app)
            else:
                print("unknow message")
        for sub_idx in app.subprocess_conns:
            master_conn, _ = app.subprocess_conns[sub_idx]
            if master_conn.poll(timeout=0.01):
                try:
                    sub_event = master_conn.recv()
                except Exception as e:
                    continue
                if sub_event['type'] == "Paused":
                    app.paused_counter += 1
                    print("worker: {} receive a pause signal, now counter is {}".format(app.appid, app.paused_counter))
                    if app.paused_counter == len(app.cuda_device):
                        app_conn.send(conn_message("Paused", {"iter": sub_event['msg']["iter"] , "epoch":sub_event['msg']["epoch"]}))
                        return
                    elif app.paused_counter == 1:
                        send_exit_to_subprocess(app)

                elif sub_event['type'] == "Finished":
                    app.finished_counter += 1
                    if app.finished_counter == len(app.cuda_device):
                        for s_idx in app.subprocess_conns:
                            master_conn, worker_conn = app.subprocess_conns[s_idx]
                            master_conn.close()
                            worker_conn.close()
                        try:
                            dist.destroy_process_group()
                        except Exception as e:
                            print("destroy_process_group failed", e)
                        app_conn.send(conn_message("Finished"))
                        return
                elif sub_event['type'] == "HeartBeat":
                    app_conn.send(sub_event)

def save_train_model(epoch, iter, model, optimizer, loss, scheduler, args, model_path=''):
    try:
        if model_path=='':
            model_path = os.path.join(args.work_space, "app-{}-{}.tar".format(
                        args.appid, "gpu{}".format(args.gpu) if args.gpu!=None else "cpu{}".format(args.gpu)))

        print(f"save model to {model_path}")
        torch.save({
                'iter': iter,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, model_path)
        print("model is saved to {}".format(model_path))
    except Exception as e:
        print("app: {}".format(args.appid), e)

def resume_train_model(model, optimizer, args, model_path=''):
    if model_path=='':
        latest_ts = 0
        for f in os.listdir(args.work_space):
            if "app-{}".format(args.appid) in f:
                if latest_ts<os.path.getctime(os.path.join(args.work_space, f)):
                    model_path = os.path.join(args.work_space, f)
        
        # model_path = os.path.join(args.work_space, "app-{}-{}.tar".format(
        #             args.appid, "gpu1"))
    # if model_path=='':
    #     model_path = os.path.join(args.work_space, "app-{}.tar".format(args.appid))
    try:
        print("{} resumes from {}".format(args.appid, model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        iter = checkpoint['iter']
        loss = checkpoint['loss']
        # scheduler = checkpoint['scheduler']
        return model, optimizer, epoch, iter, loss#, scheduler
    except Exception as e:
        print(e)
        raise

def main_worker(local_rank, ngpus_per_node, app, args):
    criterion = None
    scheduler = None
    optimizer = None
    epoch = 1

    args.gpu = local_rank
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    try:
        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                global_rank = app.base_rank + local_rank
            # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = 1
            print("create cluster on {} with {}, having {} processes, current rank is {}" \
                    .format(args.dist_backend, args.dist_url, args.world_size, global_rank))
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, timeout=datetime.timedelta(seconds=180),
                                    world_size=args.world_size, rank=global_rank)
    except Exception as e:
        print("distributed cluster: ", e)

    try:
        print("=> creating model '{}{}'".format(args.arch, args.depth))
        model = build_model(args.arch, args.depth)
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(app.batch / ngpus_per_node)
                args.workers = int(app.workers / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
                print("1")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model = torch.nn.DataParallel(model)
                model.cuda()
                print("2")
            else:
                model = torch.nn.DataParallel(model).cuda()
                print("3")

    except Exception as e:
        print("transport model to GPU", e)

    cudnn.benchmark = True

    try:
        #warmup
        optimizer_init_lr = args.lr
    
        if (args.optimizer == 'sgd'):
            optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif (args.optimizer == 'adam'):
            optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
    except Exeception as e:
        print("initialize model: ", e)


    # if this is a paused app to resmue
    try:
        if app.checkpoint:
            model, optimizer, epoch, args.start_iteration, criterion = resume_train_model(model, optimizer, args)
            criterion.cuda(args.gpu)
            print("resumes at epoch {} iter{}".format(epoch, args.start_iteration))
    except Exception as e:
        print("read checkpoint: ", e)
        
    if scheduler == None:
        try:
            #  scheduler
            if args.lr_scheduler == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-08)
            elif args.lr_scheduler == 'default':
                # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
                epoch_milestones = [60, 120]
                """Set the learning rate of each parameter group to the initial lr decayed
                    by gamma once the number of epoch reaches one of the milestones
                """
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_milestones, gamma=0.1)
            else:
                raise Exception("unknown lr scheduler")
        except Exception as e:
            print("scheduler: ", e)
        
    if criterion==None:
        # define loss function (criterion)
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        # Data loading code
        # traindir = os.path.join(args.data, 'train')
        # valdir = os.path.join(args.data, 'val')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        # train_dataset = datasets.ImageFolder(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        # if args.distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # else:
        #     train_sampler = None

        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        # val_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(valdir, transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ])),
        #     batch_size=args.batch_size, shuffle=False,
        #     num_workers=args.workers, pin_memory=True)

        ########################## Get Dataset ###############

    try:
        print("data loading")
        if "cifar10" in args.arch:
            args.dataset="cifar10"
        else:
            args.dataset="imagenet"
        train_loader, test_loader, train_sampler = get_dataset(args)
    except Exception as e:
        print("dataloading: ", e)

    # try:    
    # training
    print("before training")
    _, sub_conn = app.subprocess_conns[args.gpu]
    if sub_conn is None:
        raise Exception("Sub_conn is None")
    print("get sub_conn")
    for e in range(epoch, args.epochs + 1):
        epoch_start = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # reset start_iteration for new epoch
        if epoch!=e:
            args.start_iteration = 0
        # train for one epoch
        last_iter = train(train_loader, model, criterion, optimizer, scheduler, e, sub_conn, app, args)

        if last_iter != args.iteration:
            epoch_end = time.time()
            print_app_log(app, "subprocess {} epoch_time for {} iter {}".format(args.gpu, last_iter, epoch_end - epoch_start))
            while 1:
                if sub_conn.poll(timeout=0.01):
                    message_type = sub_conn.recv()['type']
                    if message_type == "Exit":
                        print_app_log(app, "subprocess {} exit at epoch {} iter {}".format(args.gpu, e, last_iter))
                        return
            # return
            
        # adjust learning rate
        scheduler.step()

        epoch_end = time.time()
        print_app_log(app, "subprocess {} epoch_time {}".format(args.gpu, epoch_end - epoch_start))

    print_app_log(app, "subprocess {} sends finish signal.".format(args.gpu))
    sub_conn.send(conn_message("Finished"))
    # avoid immediate exist, waiting for app_master receiving Finish signal
    # time.sleep(1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch ,sub_conn, app, args):
    try:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        forward_time = AverageMeter()
        backward_time = AverageMeter()
        update_time = AverageMeter()
        losses = AverageMeter()
        datatrans_time = AverageMeter()
        lossupdate_time = AverageMeter()
        # switch to train mode
        model.train()

        
        end = time.time()
        iter_inx =  args.start_iteration
        print(f"training starts from iteration {iter_inx}")
        for i, (input, target) in enumerate(train_loader):
            if iter_inx != -1 and iter_inx == args.iteration:
                break
            # measure data loading time
            dataloading_ts = time.time()
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # measure data transfer time
            datatransfer_ts = time.time()
            output = model(input)
            ce_loss = criterion(output, target)
            forward_ts = time.time()
            # get loss from GPU, deprecated for performance concern
            #losses.update(ce_loss.item(), input.size(0))
            loss_ts = time.time()
            # compute gradient and do SGD step
            # replace optimizer.zero_grad(), taking less memory oprations
            for param in model.parameters():
                param.grad = None
            ce_loss.backward()
            backward_ts = time.time()
            # update
            optimizer.step()
            update_ts = time.time()
            # measure elapsed time with breakdown
            start = end
            end = time.time()
            batch_time.update(end - start)
            data_time.update(dataloading_ts - start)
            datatrans_time.update(datatransfer_ts - dataloading_ts)
            forward_time.update(forward_ts - datatransfer_ts)
            lossupdate_time.update(loss_ts - forward_ts)
            backward_time.update(backward_ts - loss_ts)
            update_time.update(update_ts- backward_ts)

            if iter_inx % args.print_freq == 0:
                print("iteration {} takes {}".format(iter_inx, end-start))
                sub_conn.send(conn_message("HeartBeat", 
                            {"worker":int(args.gpu), "app_id":int(args.appid), "epoch":int(epoch), "iter": int(iter_inx),
                            "iter_time": float(np.array(batch_time.tracker[-10:len(batch_time.tracker)]).mean()),
                            "dataloading_time": float(np.array(data_time.tracker[-10:len(batch_time.tracker)]).mean())}))

            iter_inx += 1

            # check if to pause in each iteration
            if sub_conn.poll(timeout=0.001):
                message = sub_conn.recv()
                print_app_log(app, "subprocess {} receives a mesg {}".format(args.gpu, message))
                if message['type'] == "Pause":
                    if args.gpu == 1 or len(args.cuda_device)==1:
                        save_train_model(epoch, iter_inx, model, optimizer, criterion, scheduler, args)
                        print_app_log(app, "subprocess {} saves the model.".format(args.gpu))
                        print_app_log(app, "subprocess {} sends the paused signal back to app_main.".format(args.gpu))
                        sub_conn.send(conn_message("Paused", {"iter":iter_inx, "epoch": epoch}))
                    else:
                        print_app_log(app, "subprocess {} waits the main subprocess saving the model.".format(args.gpu, message))
                    while 1:
                        if sub_conn.poll(timeout=0.01):
                            message = sub_conn.recv()
                            if message['type'] == "Exit":
                                print_app_log(app, "receives exit")
                                if args.gpu == 1 or len(args.cuda_device)==1:
                                    break
                                print_app_log(app, "subprocess {} sends the paused signal back to app_main.".format(args.gpu))
                                sub_conn.send(conn_message("Paused", {"iter":iter_inx, "epoch": epoch}))
                                break
                    break


        # df = pd.DataFrame(columns=["iteration","dataloading","datatransfer", "forward","lossupdate", "backward","update","loss"])
        # df["iteration"] = np.array(batch_time.tracker)
        # df["dataloading"] = np.array(data_time.tracker)
        # df["datatransfer"] = np.array(datatrans_time.tracker)
        # df["forward"] = np.array(forward_time.tracker)
        # df["lossupdate"] = np.array(lossupdate_time.tracker)
        # df["backward"] = np.array(backward_time.tracker)
        # df["update"] = np.array(update_time.tracker)
        # #df["loss"] = np.array(losses.tracker[:len(df)])
        # trace_path = os.path.join(args.work_space, "device_{}.csv".format(args.gpu))
        # if not os.path.exists(trace_path):
        #     df.to_csv(trace_path, index=False)
        # else:
        #     previous_df = pd.read_csv(trace_path, header=0)
        #     pd.concat([df, previous_df]).to_csv(trace_path, index=False)
                
        return iter_inx

    except Exception as e:
        raise e

def validate( val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.tracker = []

    def update(self, val, n=1):
        self.val = val
        self.tracker.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def app_main(worker_conn, app: application):
    os.environ["CUDA_VISIBLE_DEVICES"] = " ".join( str(d)+',' for d in app.cuda_device)[:-1]
    args = app.parse_args()
    
    # the folder to save log and model checkpoints
    work_space = os.path.join("trace", args.work_space)
    if not os.path.exists(work_space):
        os.system("mkdir -p {}".format(work_space))
    args.work_space = work_space

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

    args.distributed = app.node_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = len(app.cuda_device)
    if args.multiprocessing_distributed:
        args.world_size = app.world_size
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        for i in range(ngpus_per_node):
            master_conn, sub_conn = Pipe()
            app.subprocess_conns[i] = (master_conn, sub_conn)

        # start a messager to communicate sub_processes
        p = Process(target=app_messager, args=(app, worker_conn, ), name="app_messeger_{}".format(app.appid), daemon=False)
        p.start()
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, app, args), join=True)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, app, args)

if __name__ == '__main__':
    app_main()
