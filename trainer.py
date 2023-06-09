import logging
import os
import random
import sys
import torch
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from predict import make_mask_as_h5

import numpy as np

from datasets.landslide_dataset import LandslideDataSet

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            #  worker_init_fn=worker_init_fn)
    # main 于此处修改batch_size
    batch_size, num_steps_stop, num_workers = 24, 5000, 4
    # main max_iters参数可去
    trainloader = DataLoader(
        #FIXME: dataset路径
        LandslideDataSet("./datasets/", "./datasets/train.txt", set='labeled',max_iters=num_steps_stop*batch_size),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    all_epoch_loss = []
    for epoch_num in iterator:
        # main 渐进式标签细化
        if epoch_num % 20 == 0 and epoch_num > 0:
            # 生成伪类标签
            make_mask_as_h5()
            logging.info('Generating mask..........')
            # 重新加载数据集
            # main max_iters参数可去
            trainloader = DataLoader(
            LandslideDataSet("./datasets/", "./datasets/new_train.txt", set='labeled',max_iters=num_steps_stop*batch_size),batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        total_loss = 0
        for i_batch, sampled_batch in enumerate(tqdm(trainloader)):

            # image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch, _, _ = sampled_batch
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            
            # print(model)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch.long(), softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                # writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/GroundTruth', labs, iter_num)
            total_loss += loss.item()

        logging.info('Epoch %d : loss : %f' % (epoch_num, loss.item()))
        # 保存epoch loss
        all_epoch_loss.append(loss.item())
        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        save_mode_path = os.path.join(snapshot_path, 'epoch_tmp.pth')
        model.cpu()
        torch.save(model, save_mode_path)
        model.cuda()
        logging.info("save model to {}".format(save_mode_path))
        
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            model.cpu()
            torch.save(model, save_mode_path)
            model.cuda()
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        np.save("./logs_16/all_epoch_loss", np.array(all_epoch_loss))
    
    # writer.close()
    return "Training Finished!"