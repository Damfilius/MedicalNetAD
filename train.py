'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset, ADNIDataset
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
from split_adni import split_dataset
import wandb
from dataset_utils import AverageMeter, calculate_accuracy
# import tqdm as tqdm
# import test

os.environ['WANDB_API_KEY'] = '3dcea36a6508b37c26d8c73c01243a435a165578'

def test(data_loader, model, loss_fn):
    model.eval() # for testing 
    device = next(model.parameters()).device

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # disable gradient calculation
    with torch.no_grad():
        # test_loop = tqdm(data_loader)
        test_start_time = time.time()
        for batch_id, batch_data in enumerate(data_loader):
            # update the time
            batch_start_time = time.time()

            # forward
            [volumes, labels] = batch_data
            volumes = torch.tensor(volumes,dtype=torch.float32).to(device)
            labels = torch.tensor(labels).to(device)

            # calculate the loss and the accuracy
            out_labels = model(volumes)
            loss = loss_fn(out_labels, labels)
            acc = calculate_accuracy(out_labels, labels)

            # update the average meters
            losses.update(loss.item(), volumes.size(0))
            accuracies.update(acc, volumes.size(0))
            batch_time.update(time.time() - batch_start_time)

        # log the results
        total_test_time = time.time() - test_start_time
        wandb.log({'test': {
                'loss': losses.avg,
                'acc': accuracies.avg,
                'avg_batch_time': batch_time.avg,
                'test_time': total_test_time}})

def train(data_loader, test_loader, model, optimizer, scheduler, total_epochs,
          save_interval, save_folder, sets):
    # settings
    model.train()
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    print("Current setting is:")
    print(sets)
    print("\n\n")
        
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        scheduler.step()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        # train_loop = tqdm(data_loader)
        epoch_start_time = time.time()
        for batch_id, batch_data in enumerate(data_loader):
            batch_start_time = time.time()

            # getting data batch
            [volumes, labels] = batch_data
            volumes = torch.tensor(volumes,dtype=torch.float32).to(device)
            labels = torch.tensor(labels).to(device)

            # calculating loss and accuracy
            out_labels = model(volumes)
            print(f"Target: {labels} - Prediction: {out_labels}")
            loss = loss_fn(out_labels, labels)
            acc = calculate_accuracy(out_labels, labels)

            # update the average meter
            losses.update(loss.item(), volumes.size(0))
            accuracies.update(acc, volumes.size(0))

            # update the weights
            optimizer.zero_grad()
            loss.backward()                
            optimizer.step()

            # log the data
            wandb.log({'train_batch': {
                    'loss': losses.val,
                    'acc': accuracies.val}})

            # update the average time
            batch_time.update(time.time() - batch_start_time)

            # log to terminal
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  batch_id + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))
          
        # log train results after an epoch
        total_epoch_time = time.time() - epoch_start_time
        wandb.log({'train_epoch': {
                'loss': losses.avg,
                'acc': accuracies.avg,
                'epoch_time': total_epoch_time,
                'avg_batch_time': batch_time.avg,
                'lr': optimizer.param_groups[0]['lr']}})

        # run test every 20 epochs
        if epoch % 10 == 0:
            test(test_loader, model, loss_fn)

        # save model
        if epoch % save_interval == 0:
            model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            
            # log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
            wandb.log({"checkpoint": {"epoch": epoch}})
            torch.save({
                        'ecpoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)
                            
    print('Finished training')

def load_train_test_set(sets):
    # prepare the file names of the train and test sets
    train_file = os.path.join(sets.data_root,"train.txt")
    test_file = os.path.join(sets.data_root,"test.txt")
    train_set, test_set = [], []
    if os.path.getsize(train_file) > 0 and os.path.getsize(test_file) > 0:
        with open(train_file,"r+") as fp:
            train_set = fp.readlines()
        with open(test_file,"r+") as fp:
            test_set = fp.readlines()
    else:
        train_set, test_set = split_dataset(sets.data_root,sets.split_ratio)

    return train_set, test_set


if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt' 
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28
       
     
    
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    print (model)
    # optimizer
    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
                ]
    optimizer = torch.optim.Adam(parameters['new_parameters'], lr=sets.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10,20,30,40], gamma=0.5)
    
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    

    train_set, test_set = load_train_test_set(sets)

    # intitialize datasets
    training_dataset = ADNIDataset(train_set, sets.data_root, True)
    testing_dataset = ADNIDataset(test_set, sets.data_root, True)
    # training_dataset = BrainS18Dataset(sets.data_root, sets.img_list, sets)
    train_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    test_loader = DataLoader(testing_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # setup the logger
    wandb.init(
        entity="dakifile03-vrije-universiteit-brussel",
        project="AD Classification",
        name=f"{sets.model}_{time.time()}",
        config={
            "learning_rate": sets.learning_rate,
            "architecture": sets.model,
            "dataset": "ADNI",
            "epochs": sets.n_epochs,
        },
    )
    wandb.config.update(sets) # config is a variable that holds and saves hyper parameters and inputs

    # training
    train(train_loader, test_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs,
           save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
