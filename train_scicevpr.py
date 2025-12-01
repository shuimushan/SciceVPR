import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
import scicevpr
import super_cricavpr
from loss_distill_crica_stable import loss_function
from dataloaders.GSVCities import get_GSVCities
import time
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

args.features_dim = args.mix_in_dim*14
#### Initialize model
model = scicevpr.SciceVPR(
                                 backbone_arch=args.backbone_arch,
                                 pretrained=True,
                                 layer1=args.layer1, #意思是不训练dino
                                 use_cls=False,
                                 norm_descs=True,
                                 out_indices=args.out_indices,
                                 backbone_out_dim=args.backbone_out_dim,
                                 mix_in_dim=args.mix_in_dim,
                                 token_num=args.token_num,
                                 token_ratio=args.token_ratio)
model = model.to(args.device)
model = torch.nn.DataParallel(model)



model_crica = super_cricavpr.Super_CricaVPR(
                                 backbone_arch=args.backbone_arch,
                                 pretrained=True,
                                 layer1=args.layer1, #意思是不训练dino
                                 use_cls=False,
                                 norm_descs=True,
                                 out_indices=args.out_indices,
                                 backbone_out_dim=args.backbone_out_dim,
                                 mix_in_dim=args.mix_in_dim,
                                 token_num=args.token_num,
                                 token_ratio=args.token_ratio)
model_crica = model_crica.to(args.device)
model_crica = torch.nn.DataParallel(model_crica)
checkpoint = torch.load(args.crica_path)
model_crica.load_state_dict(checkpoint['model_state_dict'],strict = True)
model_crica = model_crica.eval()

model_dict_weight = model.state_dict()
state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
              'conv' in k}
model_dict_weight.update(state_dict)
model.load_state_dict(model_dict_weight)
backbone_state_dict = {}
for k, v in checkpoint['model_state_dict'].items():
    if k.startswith('module.backbone.'):
        new_key = k.replace('module.backbone.', '')
        backbone_state_dict[new_key] = v

# 加载到模型的backbone中
model.module.backbone.load_state_dict(backbone_state_dict, strict=True)

## Freeze parameters 
for name, param in model.module.backbone.named_parameters():
    param.requires_grad = False
    
for name, param in model.module.named_parameters():
    if 'conv' in name:
        param.requires_grad = False

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
else:
    best_r5 = start_epoch_num = not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Getting GSVCities
train_dataset = get_GSVCities()

train_loader_config = {
    'batch_size': args.train_batch_size,
    'num_workers': args.num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': False}


#### Training loop
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
          
    model = model.train()
    epoch_losses_global=[]
    epoch_losses_distill=[]
    for images, place_id in tqdm(ds):   
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)
        with torch.no_grad():
            descriptors_crica = model_crica(images.to(args.device))
        descriptors = model(images.to(args.device))
        descriptors = descriptors.cuda()
        descriptors_crica = descriptors_crica.cuda()
        loss_global, loss_distill = loss_function(descriptors, descriptors_crica, labels) # Call the loss_function we defined above     
        loss = loss_global+loss_distill
        del descriptors, descriptors_crica
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss_global = loss_global.item()
        batch_loss_distill = loss_distill.item()        
        epoch_losses_global = np.append(epoch_losses_global, batch_loss_global)#?
        epoch_losses_distill = np.append(epoch_losses_distill, batch_loss_distill)#?
        del loss
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch global loss = {epoch_losses_global.mean():.4f}, "
                 f"average epoch distill loss = {epoch_losses_distill.mean():.4f}" )
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    
    is_best = recalls[1] > best_r5
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {(recalls[1]):.1f}")
        best_r5 = (recalls[1])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {(recalls[1]):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break


logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
logging.info("Test *best* model on test set")
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)
recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

#### Test last model on test set
logging.info("Test *last* model on test set")
last_model_state_dict = torch.load(join(args.save_dir, "last_model.pth"))["model_state_dict"]
model.load_state_dict(last_model_state_dict)
recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

