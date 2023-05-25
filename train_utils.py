import time
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics
import sklearn, sklearn.model_selection
from tqdm import tqdm
import wandb

def checkpoint(model, best_loss, epoch, LR, cfg):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, f'results/{cfg.name}_checkpoint')

def train(model, dataloaders, cfg):
    print("Our config:")
    print(cfg) 
    if not torch.cuda.is_available():
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")
    else:
        device = 'cuda'
    print(f'Using device: {device}')

    model.to(device)
    
    # define criterion, optimizer for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
            filter(
            lambda p: p.requires_grad,
            model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay)


    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1
    LR = cfg.lr #LR initial

    # iterate over epochs
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            #t = tqdm(dataloaders[phase])
            #for data in enumerate(t):
            for inputs, labels in tqdm(dataloaders[phase]):
                i += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                # calculate pred
                _, pred = torch.max(outputs, 1)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data * inputs.shape[0]
                running_corrects += torch.sum(pred == labels.data)

            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)

            if phase == 'train':
                last_train_loss = epoch_loss
                last_train_acc = epoch_acc

            print(phase + ' epoch {}:loss {:.4f} and acc {:.4f} with data size {}'.format(
                epoch, epoch_loss, epoch_acc, len(dataloaders[phase].dataset)))
            
            '''
            # decay learning rate if no val loss improvement in this epoch
            if phase == 'valid' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / cfg.lr_decay) + " as not seeing improvement in val loss")
                LR = LR / cfg.lr_decay
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=cfg.weight_decay)
                print("created new optimizer with LR " + str(LR))
            '''

            # checkpoint model if has best val loss yet
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, cfg)

            # log training and validation loss over each epoch
            if phase == 'valid':
                with open(f"results/{cfg.name}_log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss, last_train_acc, epoch_acc])

            if cfg.wandb == True:
                if phase == 'train':
                    wandb.log({"epoch":epoch,
                                "train_loss":epoch_loss,
                                "train_acc":epoch_acc})
                else:
                    wandb.log({"valid_loss":epoch_loss,
                                "valid_acc":epoch_acc})


        '''
        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break
        '''

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load(f'results/{cfg.name}_checkpoint')
    model = checkpoint_best['model']


    return model, best_epoch

