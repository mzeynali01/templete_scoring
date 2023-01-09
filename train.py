from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from test import *
from sam import SAM
from torch.utils.data.sampler import WeightedRandomSampler
from utils import *




if __name__ == '__main__':
    
    np.random.seed(72)
    torch.manual_seed(72)
    num_categories = 4

    opt = Config()
    device = torch.device("cpu")
    
    writer = SummaryWriter("./logs")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    
    weight_train = make_weights_for_balanced_classes(train_dataset.imgs.tolist(), num_categories)
    weight_train = torch.tensor(weight_train, dtype=torch.float32)
    sampler = WeightedRandomSampler(weight_train, len(weight_train))
    
    train_loader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  sampler=sampler,
                                  num_workers=opt.num_workers)

    identity_list = get_images_list(opt.images_test_list)
    img_paths = [os.path.join(opt.images_root, each) for each in identity_list]
    
    print('{} train iters per epoch:'.format(len(train_loader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    summary(model, (1, 128, 128))
    model.to(device)
    # model = DataParallel(model)
    metric_fc.to(device)
    # metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sam':
        base_optimizer = torch.optim.SGD
        optimizer = SAM([{'params': model.parameters()}, {'params': metric_fc.parameters()}], 
                        base_optimizer, lr=opt.lr, 
                        momentum=0.9)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        scheduler.step()

        model.train()
        for ii, data in enumerate(train_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            
            if opt.optimizer == 'sam':
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                feature = model(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            iters = i * len(train_loader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                
                writer.add_scalar("train_loss", loss.item(), iters)
                writer.add_scalar("train_acc", acc, iters)

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        acc = images_test(model, img_paths, identity_list, opt.images_test_list, opt.test_batch_size)
        writer.add_scalar("test_acc", acc, iters)
