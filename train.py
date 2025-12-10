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

os.environ['WANDB_API_KEY'] = '###################################'

def train(data_loader, model, optimizer, scheduler, total_epochs,
          save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_fn = loss_fn.cuda()
        
    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, labels = batch_data

            if not sets.no_cuda: 
                volumes = volumes.cuda()

            # calculating loss
            optimizer.zero_grad()
            out_labels = model(volumes)
            loss = loss_fn(out_labels, labels)
            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            wandb.log({"train": {"batch": f"{epoch}-{batch_id} ({batch_id_sp})", "loss": loss.item(), "avg_batch_time": avg_batch_time}})
            # log.info(
            #         'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}'\
            #         .format(epoch, batch_id, batch_id_sp, loss.item(), avg_batch_time))
          
            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    
                    # log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                    wandb.log('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
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
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
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
        config={
            "learning_rate": sets.learning_rate,
            "architecture": sets.model,
            "dataset": "ADNI",
            "epochs": sets.n_epochs,
        },
    )
    wandb.config.update(sets) # config is a variable that holds and saves hyper parameters and inputs

    # training
    train(train_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs,
           save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
