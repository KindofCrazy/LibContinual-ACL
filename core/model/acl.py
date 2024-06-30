import imp
import os
import random
from tabnanny import check
#from cv2 import threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from .finetune import Finetune
from copy import deepcopy

class Shared(nn.Module):
    def __init__(self, num_class, input_size=[3, 32, 32], latent_dim=512):
        super(Shared, self).__init__()
        self.num_class = num_class
        self.input_size = input_size
        self.latent_dim = latent_dim
        hiddens = [64, 128, 256, 1024, 1024, 512]

        self.conv1 = nn.Conv2d(input_size[0], hiddens[0], kernel_size=input_size[1]//8)
        s = self.compute_conv_output_size(input_size[1], input_size[1]//8)
        s = s//2
        self.conv2 = nn.Conv2d(hiddens[0], hiddens[1], kernel_size=input_size[1]//10)
        s = self.compute_conv_output_size(s, input_size[1]//10)
        s = s//2
        self.conv3 = nn.Conv2d(hiddens[1], hiddens[2], kernel_size=2)
        s = self.compute_conv_output_size(s, 2)
        s = s//2
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hiddens[2]*s*s, hiddens[3])
        self.fc2 = nn.Linear(hiddens[3], hiddens[4])
        self.fc3 = nn.Linear(hiddens[4], hiddens[5])
        self.fc4 = nn.Linear(hiddens[5], self.latent_dim)    

    def forward(self, x):
        x = x.view_as(x)
        h = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        h = self.drop2(self.relu(self.fc3(h)))
        h = self.drop2(self.relu(self.fc4(h)))
        return h
    
    def compute_conv_output_size(self, Lin,kernel_size,stride=1,padding=0,dilation=1):
        return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Private(nn.Module):
    def __init__(self, num_tasks, num_class, input_size=[3, 32, 32], latent_dim=512):
        super(Private, self).__init__()
        self.num_tasks = num_tasks
        self.num_class = num_class
        self.input_size = input_size
        self.latent_dim = latent_dim
        hiddens = [32, 32]
        flatten=1152

        self.task_out = nn.ModuleList()
        for _ in range(num_tasks):
            self.conv = nn.Sequential()
            self.conv.add_module('conv1', nn.Conv2d(input_size[0], hiddens[0], kernel_size=input_size[1]//8))
            self.conv.add_module('relu1', nn.ReLU(inplace=True))
            self.conv.add_module('drop1', nn.Dropout(0.2))
            self.conv.add_module('maxpool1', nn.MaxPool2d(2))
            self.conv.add_module('conv2', nn.Conv2d(hiddens[0], hiddens[1], kernel_size=input_size[1]//10))
            self.conv.add_module('relu2', nn.ReLU(inplace=True))
            self.conv.add_module('dropout2', nn.Dropout(0.5))
            self.conv.add_module('maxpool2', nn.MaxPool2d(2))
            self.task_out.append(self.conv)
            
            self.linear = nn.Sequential()
            self.linear.add_module('fc1', nn.Linear(flatten, self.latent_dim))
            self.linear.add_module('relu3', nn.ReLU(inplace=True))
            self.task_out.append(self.linear)

    def forward(self, x, task_id):
        x = x.view_as(x)
        out = self.task_out[2*task_id](x)
        out = out.view(out.size(0), -1)
        out = self.task_out[2*task_id+1](out)
        return out


class Model(nn.Module):
    def __init__(self, num_tasks, num_class, input_size, latent_dim, head_units):
        super(Model, self).__init__()
        self.num_tasks = num_tasks
        self.num_class = num_class
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.taskcla = [(t, num_class) for t in range(num_tasks)]
        self.hidden1 = head_units
        self.hidden2 = head_units//2


        self.shared = Shared(num_class, input_size, latent_dim)
        self.private = Private(num_tasks, num_class, input_size, latent_dim)

        self.head = nn.ModuleList()
        for i in range(self.num_tasks):
            self.head.append(
                nn.Sequential(
                    nn.Linear(2*self.latent_dim, self.hidden1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(self.hidden1, self.hidden2),
                    nn.ReLU(inplace=True),
                    # 假设每个任务中 num_class 一样
                    # 这里对应的 self.taskcla = [(t, num_class) for t in range(num_tasks)]
                    nn.Linear(self.hidden2, self.taskcla[i][1])
                )
            )

    def forward(self, x_s, x_p, tt, task_id):
        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        shared_out = self.shared(x_s)
        private_out = self.private(x_p, task_id)
        out = torch.cat([private_out, shared_out], dim=1)
        # 用不同任务的 head 对相应任务的输出进行分类
        # 这里的 tt 是该数据对应的 task_id
        return torch.stack([self.head[tt[i]].forward(out[i]) for i in range(out.size(0))])
    
    def get_encoded_ftrs(self, x_s, x_p, task_id):
        return self.shared(x_s), self.private(x_p, task_id)
    
    def get_parameters(self, config):
        return self.parameters()


# kwargs=config["classifier"]["kwargs"]
class Discriminator(torch.nn.Module):
    def __init__(self, kwargs, task_id):
        super(Discriminator, self).__init__()

        self.num_tasks=kwargs['ntasks']
        self.units=kwargs['head_units']
        self.latent_dim=kwargs['latent_dim']

        if kwargs["diff"] == "yes":
            self.dis = torch.nn.Sequential(
                GradientReversal(kwargs["lam"]),
                torch.nn.Linear(self.latent_dim, kwargs['units']),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(kwargs["units"], kwargs['units']),
                torch.nn.Linear(kwargs['units'], task_id + 2)
            )
        else:
            self.dis = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, kwargs['units']),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(kwargs['units'], kwargs['units']),
                torch.nn.Linear(kwargs['units'], task_id + 2)
            )


    def forward(self, z, labels, task_id):
        return self.dis(z)

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


    def get_size(self):
        count=sum(p.numel() for p in self.dis.parameters() if p.requires_grad)
        print('Num parameters in D       = %s ' % (self.pretty_print(count)))


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# kwargs=config["classifier"]["kwargs"]
class ACL(Finetune):
    # 舍弃 backbone 操作
    # feat_dim 舍弃算了，注意调用时此处为None
    def __init__(self, feat_dim, num_class, **kwargs):
        super(ACL, self).__init__(backbone=None, feat_dim=feat_dim, num_class=num_class, **kwargs)
        self.kwargs = kwargs
        self.nepochs = kwargs['nepochs']
        self.sbatch = kwargs['batch_size']
        self.feat_dim = feat_dim
        self.num_class = num_class
        
        self.model = Model(kwargs['ntasks'], num_class, kwargs['inputsize'], kwargs['latent_dim'], kwargs['head_units']).to(kwargs['device'])

        # optimizer & adaptive lr
        self.e_lr = kwargs["e_lr"]
        self.d_lr = kwargs["d_lr"]

        self.e_lr=[kwargs["e_lr"]] * kwargs["ntasks"]
        self.d_lr=[kwargs["e_lr"]] * kwargs["ntasks"]

        self.lr_min=kwargs["lr_min"]
        self.lr_factor=kwargs["lr_factor"]
        self.lr_patience=kwargs["lr_patience"]

        # self.samples=args.samples

        self.device=kwargs["device"]
        self.checkpoint=kwargs["checkpoint"]


        self.adv_loss_reg=kwargs["adv"]
        self.diff_loss_reg=kwargs["orth"]
        self.s_steps=kwargs["s_step"]
        self.d_steps=kwargs["d_step"]

        self.diff=kwargs["diff"]

        # self.network=network
        self.inputsize=kwargs["inputsize"]
        # self.taskcla=args.taskcla
        self.num_tasks=kwargs["ntasks"]

        # Initialize generator and discriminator
        self.model = Model(kwargs['ntasks'], num_class, kwargs['inputsize'], kwargs['latent_dim'], kwargs['head_units']).to(kwargs['device'])
        self.discriminator=self.get_discriminator(0)
        self.discriminator.get_size()

        self.latent_dim=kwargs["latent_dim"]

        self.task_loss=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_d=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_s=torch.nn.CrossEntropyLoss().to(self.device)
        self.diff_loss=DiffLoss().to(self.device)

        self.optimizer_S=self.get_S_optimizer(0)
        self.optimizer_D=self.get_D_optimizer(0)

        self.task_encoded={}

        self.mu=0.0
        self.sigma=1.0

        print()

    def get_discriminator(self, task_id):
        return Discriminator(self.kwargs, task_id).to(self.device)
    
    def get_S_optimizer(self, task_id, e_lr=None):
        if e_lr is None: e_lr=self.e_lr[task_id]
        optimizer_S=torch.optim.SGD(self.model.parameters(), momentum=self.kwargs["mom"],
                                    weight_decay=self.kwargs["e_wd"], lr=e_lr)
        return optimizer_S
    
    def get_D_optimizer(self, task_id, d_lr=None):
        if d_lr is None: d_lr=self.d_lr[task_id]
        optimizer_D=torch.optim.SGD(self.discriminator.parameters(), weight_decay=self.kwargs["d_wd"], lr=d_lr)
        return optimizer_D

    def before_task(self, task_id, buffer, train_loader, test_loaders):
        if task_id > 0:
            self.model=self.prepare_model(task_id)

        self.task_id = task_id
        self.discriminator = self.get_discriminator(task_id)

        self.best_loss=np.inf
        self.best_model=self.get_model(self.model)

        self.best_loss_d=np.inf
        self.best_model_d=self.get_model(self.discriminator)

        self.dis_lr_update=True
        self.d_lr_epoch=self.d_lr[task_id]
        self.patience_d_epoch=self.lr_patience
        self.optimizer_D=self.get_D_optimizer(task_id, self.d_lr_epoch)

        self.e_lr_epoch=self.e_lr[task_id]
        self.patience_epoch=self.lr_patience
        self.optimizer_S=self.get_S_optimizer(task_id, self.e_lr_epoch)

    def observe(self, data):
        # return 0,0,0
        # 面对一个batch的数据，需要进行对抗训练。Train Shared Module and Train Discriminator
        # `tt` 与 `td` 的意义
        self.model.train()
        self.discriminator.train()

        x, y, tt, td = data['image'], data['label'], data['tt'], data['td']
        x=x.to(self.device)
        y=y.to(self.device, dtype=torch.long)
        tt=tt.to(self.device)
        
        # Detaching samples in the batch which do not belong to the current task before feeding them to P
        t_current=self.task_id * torch.ones_like(tt)
        body_mask=torch.eq(t_current, tt).cpu().numpy()
        # x_task_module=data.to(device=self.device)
        x_task_module=x.clone()
        for index in range(x.size(0)):
            if body_mask[index] == 0:
                x_task_module[index]=x_task_module[index].detach()
        x_task_module=x_task_module.to(device=self.device)

        # Discriminator's real and fake task labels
        t_real_D=td.to(self.device)
        t_fake_D=torch.zeros_like(t_real_D).to(self.device)


        # ================================================================== #
        #                        Train Shared Module                          #
        # ================================================================== #
        # training S for s_steps
        for s_step in range(self.s_steps):
            self.optimizer_S.zero_grad()
            self.model.zero_grad()
            # print(x.get_device(), x_task_module.get_device(), tt.get_device(), y.get_device())
            
            output = self.model(x, x_task_module, tt, self.task_id)
            task_loss = self.task_loss(output, y)
            
            shared_encoded, task_encoded=self.model.get_encoded_ftrs(x, x_task_module, self.task_id)
            dis_out_gen_training=self.discriminator.forward(shared_encoded, t_real_D, self.task_id)
            adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)

            if self.diff=="yes":
                diff_loss=self.diff_loss(shared_encoded, task_encoded)
            else:
                diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                self.diff_loss_reg=0

            total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss
            total_loss.backward(retain_graph=True)

            self.optimizer_S.step()

        # ================================================================== #
        #                          Train Discriminator                       #
        # ================================================================== #
        # training discriminator for d_steps
        for d_step in range(self.d_steps):
            self.optimizer_D.zero_grad()
            self.discriminator.zero_grad()

            # training discriminator on real data
            output=self.model(x, x_task_module, tt, self.task_id)
            
            shared_encoded, task_out=self.model.get_encoded_ftrs(x, x_task_module, self.task_id)
            dis_real_out=self.discriminator.forward(shared_encoded.detach(), t_real_D, self.task_id)
            dis_real_loss=self.adversarial_loss_d(dis_real_out, t_real_D)
            dis_real_loss.backward(retain_graph=True)

            # training discriminator on fake data
            z_fake=torch.as_tensor(np.random.normal(self.mu, self.sigma, (x.size(0), self.latent_dim)), dtype=torch.float32, device=self.device)
            dis_fake_out=self.discriminator.forward(z_fake, t_real_D, self.task_id)
            dis_fake_loss=self.adversarial_loss_d(dis_fake_out, t_fake_D)
            dis_fake_loss.backward(retain_graph=True)

            self.optimizer_D.step()    

        # 以下是要返回的 output, total_loss, acc
        # test_print()
        self.model.eval()
        self.discriminator.eval()
        with torch.no_grad():
            output = self.model(x, x, tt, self.task_id)
            
            task_loss = self.task_loss(output, y)
            shared_encoded, task_encoded=self.model.get_encoded_ftrs(x, x_task_module, self.task_id)
            dis_out_gen_training=self.discriminator(shared_encoded, t_real_D, self.task_id)
            adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)
            if self.diff=="yes":
                diff_loss=self.diff_loss(shared_encoded, task_encoded)
            else:
                diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                self.diff_loss_reg=0
            total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss
            
            _, pred=output.max(1)
            acc = pred.eq(y.view_as(pred)).sum().item() / x.size(0)
            # print(acc)
            # print(pred.eq(y.view_as(pred)).size(), y.size(0))
        
        return output, acc, total_loss

    def after_task(self, task_id, buffer, train_loader, test_loaders):
        # 保存模型
        # Restore best validation model (early-stopping)
        self.model.load_state_dict(copy.deepcopy(self.best_model))
        self.discriminator.load_state_dict(copy.deepcopy(self.best_model_d))

        # 看要怎么实现 save 和 load 相关操作
        self.save_all_models(task_id)

    def after_epoch(self, task_id, epoch_id, train_loader, test_loader):
        train_res=self.eval_(train_loader, task_id)
        self.report_tr(train_res, epoch_id, self.sbatch)

        # lowering the learning rate in the beginning if it predicts random chance for the first 5 epochs
        if epoch_id == 4:
            random_chance=20.
            threshold=random_chance + 2

            if train_res['acc_t'] < threshold:
                # Restore best validation model
                self.d_lr_epoch=self.d_lr[task_id] / 10.
                self.optimizer_D=self.get_D_optimizer(task_id, self.d_lr_epoch)
                print("Performance on task {} is {} so Dis's lr is decreased to {}".format(task_id, train_res[
                    'acc_t'], self.d_lr_epoch), end=" ")

                self.e_lr_epoch=self.e_lr[task_id] / 10.
                self.optimizer_S=self.get_S_optimizer(task_id, self.e_lr_epoch)

                self.discriminator=self.get_discriminator(task_id)

                # load model
                if task_id > 0:
                    self.model=self.load_checkpoint(task_id - 1)
                else:
                    self.model=Model(self.kwargs['ntasks'], self.kwargs['num_class'], self.kwargs['inputsize'], self.kwargs['latent_dim'], self.kwargs['head_units']).to(self.kwargs['device'])


        # Valid
        valid_res=self.eval_(test_loader[-1], task_id)
        self.report_val(valid_res)

        # Adapt lr for S and D
        if valid_res['loss_tot'] < self.best_loss:
            self.best_loss=valid_res['loss_tot']
            self.best_model=self.get_model(self.model)
            self.patience_epoch=self.lr_patience
            print(' *', end='')
        else:
            self.patience_epoch-=1
            if self.patience_epoch <= 0:
                self.e_lr_epoch/=self.lr_factor
                print(' lr={:.1e}'.format(self.e_lr_epoch), end='')
                if self.e_lr_epoch < self.lr_min:
                    print()
                self.patience_epoch=self.lr_patience
                self.optimizer_S=self.get_S_optimizer(task_id, self.e_lr_epoch)

        if train_res['loss_a'] < self.best_loss_d:
            self.best_loss_d=train_res['loss_a']
            self.best_model_d=self.get_model(self.discriminator)
            self.patience_d_epoch=self.lr_patience
        else:
            self.patience_d_epoch-=1
            if self.patience_d_epoch <= 0 and self.dis_lr_update:
                self.d_lr_epoch/=self.lr_factor
                print(' Dis lr={:.1e}'.format(self.d_lr_epoch))
                if self.d_lr_epoch < self.lr_min:
                    self.dis_lr_update=False
                    print("Dis lr reached minimum value")
                    print()
                self.patience_d_epoch=self.lr_patience
                self.optimizer_D=self.get_D_optimizer(task_id, self.d_lr_epoch)
        print()
    
    def inference(self, data, task_id):
        # 考虑如何实现 inference，返回 output, acc
        test_model = self.load_model(task_id) if task_id < self.task_id else self.model
        correct_t, num=0, 0

        test_model.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            x, y, tt, td = data['image'], data['label'], data['tt'], data['td']
            x=x.to(self.device)
            y=y.to(self.device, dtype=torch.long)
            tt=tt.to(self.device)
            t_real_D=td.to(self.device)

            output=test_model(x, x, tt, task_id)
            _, pred=output.max(1)
            correct_t+=pred.eq(y.view_as(pred)).sum().item()
            num+=x.size(0)

        return output, correct_t / num

    def eval_(self, dataloader, task_id):
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num=0
        batch=0

        self.model.eval()
        self.discriminator.eval()

        res={}
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                # print(data.keys())
                x=data['image'].to(self.device)
                y=data['label'].to(self.device, dtype=torch.long)
                tt=data['tt'].to(self.device)
                t_real_D=data['td'].to(self.device)

                # Forward
                output=self.model(x, x, tt, self.task_id)
                shared_out, task_out=self.model.get_encoded_ftrs(x, x, task_id)
                _, pred=output.max(1)
                correct_t+=pred.eq(y.view_as(pred)).sum().item()

                # Discriminator's performance:
                output_d=self.discriminator.forward(shared_out, t_real_D, task_id)
                _, pred_d=output_d.max(1)
                correct_d+=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                # Loss values
                task_loss=self.task_loss(output, y)
                adv_loss=self.adversarial_loss_d(output_d, t_real_D)

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_out, task_out)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0

                total_loss = task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss

                loss_t+=task_loss
                loss_a+=adv_loss
                loss_d+=diff_loss
                loss_total+=total_loss

                num+=x.size(0)

        res['loss_t'], res['acc_t']=loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)

        return res
    
    def get_parameters(self, config):
        return self.model.get_parameters(config)

    def report_tr(self, res, e, sbatch):
        # Training performance
        print(
            '| Epoch {:3d} | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
            'Diff loss:{:.3f} |'.format(
                e, res['loss_tot'],
                res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')
        
    def report_val(self, res):
        # Validation performance
        print(' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
            res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')
        
    def get_model(self, model):
        return deepcopy(model.state_dict())
    
    def save_all_models(self, task_id):
        print("Saving all models for task {} ...".format(task_id+1))
        dis=self.get_model(self.discriminator)
        torch.save({'model_state_dict': dis,
                    }, os.path.join(self.checkpoint, 'discriminator_{}.pth.tar'.format(task_id)))

        model=self.get_model(self.model)
        torch.save({'model_state_dict': model,
                    }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))

    def load_model(self, task_id):
        # Load a previous model
        # print(self.kwargs)
        net=Model(self.kwargs['ntasks'], self.num_class, self.kwargs['inputsize'], self.kwargs['latent_dim'], self.kwargs['head_units']).to(self.kwargs['device'])
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])

        # Change the previous shared module with the current one
        current_shared_module=deepcopy(self.model.shared.state_dict())
        net.shared.load_state_dict(current_shared_module)

        return net
    
    def load_checkpoint(self, task_id):     
        print("Loading checkpoint for task {} ...".format(task_id))

        # Load a prevoius model
        net=Model(self.kwargs['ntasks'], self.num_class, self.kwargs['inputsize'], self.kwargs['latent_dim'], self.kwargs['head_units']).to(self.kwargs['device'])
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])

        return net
    
    def prepare_model(self, task_id):
        # Load a previous model and grab its shared module
        old_net = self.load_checkpoint(task_id-1)
        old_shared_module = old_net.shared.state_dict()

        # Instantiate a new model and replace its shared module
        model = Model(self.kwargs['ntasks'], self.num_class, self.kwargs['inputsize'], self.kwargs['latent_dim'], self.kwargs['head_units']).to(self.kwargs['device'])
        model.shared.load_state_dict(old_shared_module)
        model = model.to(self.device)

        return model




class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))


def test_print():
    print("Test print")
    return True