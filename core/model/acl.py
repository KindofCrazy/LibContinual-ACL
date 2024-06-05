import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .finetune import Finetune

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
    
    def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
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
        return torch.stack([self.head[tt[i]].forward(out[i]) for i in range(out.size(0))])
    
    def get_encoded_ftrs(self, x_s, x_p, task_id):
        return self.shared(x_s), self.private(x_p, task_id)

class Discriminator(torch.nn.Module):
    def __init__(self, args, task_id):
        super(Discriminator, self).__init__()

        self.num_tasks=args.ntasks
        self.units=args.units
        self.latent_dim=args.latent_dim

        if args.diff == 'yes':
            self.dis = torch.nn.Sequential(
                GradientReversal(args.lam),
                torch.nn.Linear(self.latent_dim, args.units),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.units, args.units),
                torch.nn.Linear(args.units, task_id + 2)
            )
        else:
            self.dis = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, args.units),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.units, args.units),
                torch.nn.Linear(args.units, task_id + 2)
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

class ACL(Finetune):
    # 舍弃 backbone 操作
    def __init__(self, feat_dim, num_class, **kwargs):
        super(self).__init__(feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.network = Model(kwargs['ntasks'], num_class, kwargs['inputsize'], feat_dim, kwargs['head_units'])
        self.npochs = kwargs['npochs']
        self.sbatch = kwargs['sbatch']

    def get_discriminator(self, task_id):
        return Discriminator(self.kwargs, task_id).to(self.device)
    
    def get_S_optimizer(self, task_id, e_lr=None):
        if e_lr is None:
            e_lr = self.kwargs['e_lr'][task_id]
        
    def before_task(self, task_id, buffer, train_loader, test_loaders):
        self.task_id = task_id

    def observe(self, data):
        # 面对一个batch的数据，需要进行对抗训练。Train Shared Module and Train Discriminator
        x, y = data['image'], data['label']
        x.to(self.device)
        y.to(self.device)

    def observe_epoch(self, train_loader):
        self.model.train()
        self.discriminator.train()

    def after_task(self, task_id, buffer, train_loader, test_loaders):
        # 保存模型


    def 