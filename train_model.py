import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import train_test_split
import train_utils
import test_utils
import wandb

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="densenet121_pretrain_freeze_ft2")
    parser.add_argument('--model', type=str, default="densenet121_pretrain_freeze")
    parser.add_argument('--dataset_path', type=str, default="archive/chest_xray_resized/")
    parser.add_argument('--num_epochs', type=int, default=10, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--lr', type=float, default=0.00005, help='')
    #parser.add_argument('--lr_decay', type=float, default=10, help='')#default 10 -> lr/10
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='')
    parser.add_argument('--wandb', type=bool, default=True, help='')

    cfg = parser.parse_args()
    print(cfg) 

    random_seed = 2023
    torch.manual_seed(random_seed)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    data_dir = "archive/chest_xray_resized/"

    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomAffine(15, 
                                        translate=(0.05, 0.05), 
                                        scale=(1.0-0.1, 1.0+0.1)),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ]), 
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ]),    
            }
    # Load the datasets with ImageFolder
    datasets_train_val = datasets.ImageFolder(data_dir+"train/" ,data_transforms['train'])
    data_sets={
            'test': datasets.ImageFolder(data_dir+"test/", data_transforms['test'])
            }
    data_sets['train'], data_sets['valid'] = train_test_split(datasets_train_val, test_size=0.2, random_state=random_seed)


    # Using the image datasets and the trainforms, define the dataloaders
    data_loaders={
            'train': torch.utils.data.DataLoader(data_sets['train'], batch_size=cfg.batch_size,shuffle=True),
            'test': torch.utils.data.DataLoader(data_sets['test'], batch_size=cfg.batch_size,shuffle=True),
            'valid': torch.utils.data.DataLoader(data_sets['valid'], batch_size=cfg.batch_size,shuffle=True)
            }

    #print(data_sets)

    # create models
    if "ft" in cfg.name:
        checkpoint_best = torch.load(f'results/{cfg.model}_checkpoint')
        model = checkpoint_best['model']
        for param in model.parameters():
            param.requires_grad = True
        in_features = model.classifier.in_features

        model.classifier = torch.nn.Linear(in_features, 2)
    elif "densenet121" == cfg.model:
        model = models.densenet121(weights='DEFAULT')
        #num_ftrs = model.classifier.in_features
        #model.classifier = torch.nn.Sequential(
        #torch.nn.Linear(num_ftrs, 2), torch.nn.Sigmoid())
        for param in model.parameters():
            param.requires_grad = True
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, 2)

    elif "densenet121_pretrain_freeze" == cfg.model:
        model = models.densenet121(weights='DEFAULT')
        #num_ftrs = model.classifier.in_features
        #model.classifier = torch.nn.Sequential(
        #torch.nn.Linear(num_ftrs, 2), torch.nn.Sigmoid())
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, 2)



    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="CXR-pneumonia",
    name=cfg.name,
    # track hyperparameters and run metadata
    config=cfg
    )

    model, best_epoch = train_utils.train(model, data_loaders, cfg)
    accuracy, f1, auc = test_utils.test_model(model, data_loaders)
    print('In test set: accuracy {:.4f} f1 {:.4f} and auc {:.4f} with data size {}'.format(
                accuracy, f1, auc, len(data_loaders['test'].dataset)))
    wandb.log({"test_acc":accuracy,
                                "test_f1":f1,
                                "test_auc":auc})
    

    # 🐝 Close your wandb run 
    wandb.finish()
