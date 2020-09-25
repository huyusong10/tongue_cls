# train.py - Training method for this tonuge project.
# Author: hys
# Email: 1350460443@qq.com

import os
import yaml
import datetime

import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from tensorboardX import SummaryWriter

from efficientnet_pytorch import EfficientNet
from utils import check_file, select_device, init_seeds

from runner import Runner
from tongue_data import get_loaders

# Class to wrap the original yaml instance for convinient params extraction
class Params:
    def __init__(self, file=r'./params.yml'):
        with open(file) as f:
            self.params = yaml.safe_load(f)

    def __getattr__(self, item):
        return self.params.get(item, None)

    def get_dict(self):
        return self.params

# Initialize efficientnet
def get_model(params):
    if not params.resume:
        params.weights = check_file(params.weights)
        ckpt = None
        model = EfficientNet.from_pretrained('efficientnet-b{}'.format(params.model_level), weights_path=params.weights, num_classes=params.num_classes, image_size=params.img_size)
    else:
        params.resume = check_file(params.resume)
        ckpt = torch.load(params.resume)
        model = EfficientNet.from_name('efficientnet-b{}'.format(params.model_level), num_classes=params.num_classes, image_size=params.img_size)
        model.load_state_dict(ckpt['network'], strict=False)
    return model, ckpt

# 'auto' is recommanded
# Details in https://arxiv.org/pdf/1812.01187.pdf
def get_scheduler(optim, sche_type, step_size, epochs):
    if sche_type == "exp":
        return StepLR(optim, step_size, 0.97)
    # elif sche_type == "cosine":
    #     return CosineAnnealingLR(optim, t_max)
    elif sche_type == "auto":
        return LambdaLR(optim, lambda x: (((1 + np.cos(x * np.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1)
    else:
        return None


if __name__ == "__main__":
    params_file = 'params.yml'
    params_file = check_file(params_file)
    params = Params('params.yml')  # load params

    params.save_dir = os.path.join(os.getcwd(), params.save_dir)
    os.makedirs(params.save_dir, exist_ok=True)  # build ouput directory

    device = select_device(params.device, batch_size=params.batch_size)  # build GPU env
    init_seeds(1)

    train_loader, val_loader = get_loaders(params.input_dir, params.num_classes, params.img_size, params.batch_size, params.num_workers)

    net, ckpt = get_model(params)
    net = nn.DataParallel(net).to(device)

    ''' This CrossEntropyLoss implementation comes with a softmax activation function,
        which is not suitable for this multi-label classification situation
    '''
    # loss = nn.CrossEntropyLoss()

    loss = nn.BCEWithLogitsLoss()

    ''' Adam optimizer has fastest training speed, but with Familiarity to data, SGD is recommanded
    '''
    optim = {
        "adamw" : lambda : torch.optim.AdamW(net.parameters(), lr=params.lr, betas=eval(params.beta), weight_decay=params.weight_decay),
        "rmsprop" : lambda : torch.optim.RMSprop(net.parameters(), lr=params.lr, momentum=params.momentum, eps=params.eps, weight_decay=params.weight_decay),
        "SGD": lambda : torch.optim.SGD(net.parameters(), lr=params.lr, momentum=params.momentum, nesterov=True, weight_decay=params.weight_decay)
    }[params.optim]()

    scheduler = get_scheduler(optim, params.scheduler, int(2.4 * len(train_loader)), params.epoch)

    ''' use tensorboard like 'tensorboard --logdir=. --port=8091', get your chart from browser at 'localhost:8091'
    '''
    writer = SummaryWriter(params.save_dir +f'/{datetime.datetime.now().strftime("%Y%m%d:%H%M")}')  
    params.save_dir = writer.logdir
    with open(os.path.join(params.save_dir, 'params.yml'), 'w') as f:  # save parameters
        yaml.dump(params.get_dict(), f, sort_keys=False)

    model = Runner(params, net, optim, device, loss, writer, scheduler, ckpt)  # build Runner instance

    model.train(train_loader, val_loader)  
    writer.close()
