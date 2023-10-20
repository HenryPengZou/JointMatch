# A class for a group of networks:
# Functions: 1. initialize the group of networks
#            2. forward the group of networks
#            3. update the group of networks

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from utils.ema import EMA
from models.model import TextClassifier


class NetGroup(nn.Module):
    def __init__(self, net_arch, num_nets, n_classes, device, lr, lr_linear=1e-3):
        super(NetGroup, self).__init__()
        # parameters
        self.net_arch = net_arch
        self.num_nets = num_nets
        self.n_classes = n_classes
        self.device = device
        self.lr = lr

        # initialize the group of networks
        self.nets = {}
        for i in range(num_nets):
            self.nets[i] = self.init_net(net_arch)

        # initialize optimizers for the group of networks
        self.optimizers = {}
        for i in range(num_nets):
            self.optimizers[i] = self.init_optimizer(self.nets[i], lr, lr_linear)

    # initialize one network
    def init_net(self, net_arch):
        if net_arch == 'bert-base-uncased':
            net = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = self.n_classes)
        elif net_arch == 'bert-base-uncased-2':
            net = TextClassifier(num_labels = self.n_classes)
        net.to(self.device)
        return net

    # initialize one optimizer
    def init_optimizer(self, net, lr, lr_linear):
        if self.net_arch == 'bert-base-uncased':
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('net_arch: ', self.net_arch, 'lr: ', lr)
        elif self.net_arch == 'bert-base-uncased-2':
            optimizer_net = AdamW([{"params": net.bert.parameters(), "lr": lr},
                                   {"params": net.linear.parameters(), "lr": lr_linear}])
            print('net_arch: ', self.net_arch, 'lr: ', lr, 'lr_linear: ', lr_linear)  
        return optimizer_net
    
    # EMA initialization
    def init_ema(self, ema_momentum):
        self.emas = {}
        for i in range(self.num_nets):
            self.emas[i] = EMA(self.nets[i], ema_momentum)
            self.emas[i].register()

    # switch to eval mode with EMA
    def eval_ema(self):
        for i in range(self.num_nets):
            self.emas[i].apply_shadow()

    # switch to train mode with EMA
    def train_ema(self):
        for i in range(self.num_nets):
            self.emas[i].restore()

    # switch to train mode
    def train(self):
        for i in range(self.num_nets):
            self.nets[i].train()

    # switch to eval mode
    def eval(self):
        for i in range(self.num_nets):
            self.nets[i].eval()

    # forward one network
    def forward_net(self, net, x, y=None):
        if self.net_arch == 'bert-base-uncased':
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits
        elif self.net_arch == 'bert-base-uncased-2':
            x.to(self.device)
            outs = net(x)
        return outs
    
    # forward the group of networks from the same batch input
    def forward(self, x, y=None):
        outs = []
        for i in range(self.num_nets):
            outs.append(self.forward_net(self.nets[i], x, y))
        return outs
    
    # update one network
    def update_net(self, net, optimizer, loss):
        # always clear any previously calculated gradients before performing backward pass
        net.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    # update the group of networks
    def update(self, losses):
        for i in range(self.num_nets):
            self.update_net(self.nets[i], self.optimizers[i], losses[i])

    # update the group of networks with EMA
    def update_ema(self):
        for i in range(self.num_nets):
            self.emas[i].update()

    # save & load
    # save the group of models
    def save_model(self, path, name, ema_mode=False):
        # use ema model for evaluation
        ema_model = {}
        for i in range(self.num_nets):
            filename = os.path.join(path, name + '_net' + str(i) + '.pth')
            # switch to eval mode with EMA
            self.nets[i].eval()
            if ema_mode:
                    self.emas[i].apply_shadow()
            ema_model[i] = self.nets[i].state_dict()
            # restore training mode
            if ema_mode:
                self.emas[i].restore()
            self.nets[i].train()

            # save model
            torch.save({'model': self.nets[i].state_dict(),
                       'optimizer': self.optimizers[i].state_dict(),
                       'ema_model': ema_model[i]},
                       filename)
        print('Save model to {}'.format(path))

    # load the group of models
    def load_model(self, path, name, ema_mode=False):
        ema_model = {}
        for i in range(self.num_nets):
            filename = os.path.join(path, name + '_net' + str(i) + '.pth')
            checkpoint = torch.load(filename)
            self.nets[i].load_state_dict(checkpoint['model'])
            self.optimizers[i].load_state_dict(checkpoint['optimizer'])
            if ema_mode:
                ema_model[i] = deepcopy(self.nets[i])
                ema_model[i].load_state_dict(checkpoint['ema_model'])
                self.emas[i].load(ema_model[i])
            print('Load model from {}'.format(filename))


















