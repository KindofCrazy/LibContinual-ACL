import os
import random
import torch
from torch import nn
from time import time
from tqdm import tqdm
from core.data import get_dataloader
from core.utils import init_seed, AverageMeter, get_instance, GradualWarmupScheduler, count_parameters
import core.model as arch
from torch.utils.data import DataLoader
import numpy as np
import sys
from core.utils import Logger, fmt_date_str
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from copy import deepcopy
from pprint import pprint

#! 删除不必要的代码
# 1. _init_optim
# 2. _init_buffer
# 3. stage2_train


class Trainer(object):
    """
    The Trainer.
    
    Build a trainer from config dict, set up optimizer, model, etc.
    """

    def __init__(self, rank, config):
        self.rank = rank
        self.config = config
        self.config['rank'] = rank
        self.distribute = self.config['n_gpu'] > 1  # 暂时不考虑分布式训练
        self.logger = self._init_logger(config)           
        self.device = self._init_device(config) 
        
        pprint(self.config)

        self.init_cls_num, self.inc_cls_num, self.task_num = self._init_data(config)
        self.model = self._init_model(config) 
        (
            self.train_loader,
            self.test_loader,
        ) = self._init_dataloader(config)
        

        self.train_meter, self.test_meter = self._init_meter()

        self.val_per_epoch = config['val_per_epoch']

        
        if self.config["classifier"]["name"] == "bic":
            self.stage2_epoch = config['stage2_epoch']


    def _init_logger(self, config, mode='train'):
        '''
        Init logger.

        Args:
            config (dict): Parsed config file.

        Returns:
            logger (Logger)
        '''

        save_path = config['save_path']
        log_path = os.path.join(save_path, "log")
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        log_prefix = config['classifier']['name'] + "-" + config['backbone']['name'] + "-" + f"epoch{config['epoch']}" #mode
        log_file = os.path.join(log_path, "{}-{}.log".format(log_prefix, fmt_date_str()))

        logger = Logger(log_file)

        # hack sys.stdout
        sys.stdout = logger

        return logger

    def _init_device(self, config):
        """"
        Init the devices from the config.
        
        Args:
            config(dict): Parsed config file.
            
        Returns:
            device: a device.
        """
        init_seed(config['seed'], config['deterministic'])
        device = torch.device("cuda:{}".format(config['device_ids']))

        return device


    def _init_files(self, config):
        pass

    def _init_writer(self, config):
        pass

    def _init_meter(self, ):
        """
        Init the AverageMeter of train/val/test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        train_meter = AverageMeter(
            "train",
            ["batch_time", "data_time", "calc_time", "loss", "acc1"],
        )

        test_meter = AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
        )

        return train_meter, test_meter


    def _init_data(self, config):
        return config['init_cls_num'], config['inc_cls_num'], config['task_num']

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the pretrained params or resume from a
        checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        # TODO 需要加载ACL模型，需要修改，舍弃原有的模型加载方式
        backbone = get_instance(arch, "backbone", config)
        dic = {"backbone": backbone, "device": self.device}

        model = get_instance(arch, "classifier", config, **dic)
        print(backbone)
        print("Trainable params in the model: {}".format(count_parameters(model)))

        model = model.to(self.device)
        return model
    
    def _init_dataloader(self, config):
        '''
        Init DataLoader

        Args:
            config (dict): Parsed config file.

        Returns:
            train_loaders (list): Each task's train dataloader.
            test_loaders (list): Each task's test dataloader.
        '''
        train_loaders = get_dataloader(config, "train")
        test_loaders = get_dataloader(config, "test", cls_map=train_loaders.cls_map)

        return train_loaders, test_loaders

    # TODO 每一个任务要训练 self.epochs 遍，每一遍都要调用 self._train() 方法
    def train_loop(self,):
        """
        The norm train loop:  before_task, train, test, after_task
        """
        experiment_begin = time()
        for task_idx in range(self.task_num):
            self.task_idx = task_idx
            print("================Task {} Start!================".format(task_idx))
            self.buffer.total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num
            if hasattr(self.model, 'before_task'):
                self.model.before_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))

            dataloader = self.train_loader.get_loader(task_idx)

            print("================Task {} Training!================".format(task_idx))
            print("The training samples number: {}".format(len(dataloader.dataset)))

            best_acc = 0.
            for epoch_idx in range(self.init_epoch if task_idx == 0 else self.inc_epoch):

                print("learning rate: {}".format(self.scheduler.get_last_lr()))
                print("================ Train on the train set ================")
                train_meter = self._train(epoch_idx, dataloader)
                print("Epoch [{}/{}] |\tLoss: {:.4f} \tAverage Acc: {:.2f} ".format(epoch_idx, self.init_epoch if task_idx == 0 else self.inc_epoch, train_meter.avg('loss'), train_meter.avg("acc1")))

                self.model.after_epoch(task_idx, epoch_idx, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))

                if (epoch_idx+1) % self.val_per_epoch == 0 or (epoch_idx+1)==self.inc_epoch:
                    print("================ Test on the test set ================")
                    test_acc = self._validate(task_idx)
                    best_acc = max(test_acc["avg_acc"], best_acc)
                    print(
                    " * Average Acc: {:.2f} Best acc {:.2f}".format(test_acc["avg_acc"], best_acc)
                    )
                    print(
                    " * Per-Task Acc:{}".format(test_acc['per_task_acc'])
                    )

            if hasattr(self.model, 'after_task'):
                self.model.after_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))

            print("================Task {} Testing!================".format(task_idx))
            test_acc = self._validate(task_idx)
            best_acc = max(test_acc["avg_acc"], best_acc)
            print(" * Average Acc: {:.2f} Best acc {:.2f}".format(test_acc["avg_acc"], best_acc))
            print(" * Per-Task Acc:{}".format(test_acc['per_task_acc']))


    def _train(self, epoch_idx, dataloader):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index

        Returns:
            dict:  {"avg_acc": float}
        """
        # TODO 每个`epoch`训练。在 `model.observe()` 内部进行参数更新，仅返回损失与准确率。
        self.model.train()
        self.discriminator.train()
        meter = deepcopy(self.train_meter)
        meter.reset()

        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                output, acc, loss = self.model.observe(batch)

                pbar.update(1)
                
                meter.update("acc1", 100 * acc)
                meter.update("loss", loss.item())

        return meter


    def _validate(self, task_idx):
        # TODO 重新加载对应 `task_id` 对应模型参数，并进行`Shared` 的替换，再用该模型对该任务进行`inference`。

        dataloaders = self.test_loader.get_loader(task_idx)

        self.model.eval()
        total_meter = deepcopy(self.test_meter)
        meter = deepcopy(self.test_meter)
        
        total_meter.reset()
        meter.reset()
        
        per_task_acc = []
        with torch.no_grad():
            for t, dataloader in enumerate(dataloaders):
                meter.reset()
                for batch_idx, batch in enumerate(dataloader):
                    output, acc = self.model.inference(batch)
                    meter.update("acc1", 100 * acc)
                    total_meter.update("acc1", 100 * acc)

                per_task_acc.append(round(meter.avg("acc1"), 2))
        
        return {"avg_acc" : round(total_meter.avg("acc1"), 2), "per_task_acc" : per_task_acc}