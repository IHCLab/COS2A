from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from architecture.COS2A import COS2A
from data import get_training_set, get_validation_set
from torch.autograd import Variable
from psnr import MPSNR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch COS2A HSI Reconstruction')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--in_channels', type=int, default=12, help='input MSI channels')
parser.add_argument('--out_channels', type=int, default=172, help='output HSI channels')
parser.add_argument('--num_iterations', type=int, default=2, help='number of iterations in MidStage')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='./TrainedModel_COS2A/', help='Directory to keep training outputs.')
parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint file for continuing training or testing')
opt = parser.parse_args()

print(opt)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.patch_size)
val_set = get_validation_set()

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, pin_memory=True)

print('===> Building model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = COS2A(
    in_channels=opt.in_channels,
    out_channels=opt.out_channels
).to(device)

# Set number of iterations for MidStage
model.num_iterations = opt.num_iterations

print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters()))) #***#

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Initialize optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5)

# Load checkpoint if specified
start_epoch = 0
if opt.checkpoint:
    if os.path.isfile(opt.checkpoint):
        print(f"Loading checkpoint '{opt.checkpoint}'")
        checkpoint = torch.load(opt.checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['param'])
        optimizer.load_state_dict(checkpoint['adam'])
        print(f"Loaded checkpoint '{opt.checkpoint}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{opt.checkpoint}'")

# Loss function
criterion = nn.L1Loss()

# TensorBoard setup
current_time = datetime.now().strftime('%b%d_%H-%M-%S') 
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'cos2a_hsi' + CURRENT_DATETIME_HOSTNAME)
current_step = 0

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdir(opt.save_folder)

def train(epoch, optimizer, scheduler):
    global current_step
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader):
        msi, hsi = batch[0].to(device), batch[1].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(msi)
        
        # Calculate loss
        loss = criterion(output, hsi)
        epoch_loss += loss.item()
        
        # Log to TensorBoard
        tb_logger.add_scalar('train_loss', loss.item(), current_step)
        current_step += 1
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress
        if iteration % 10 == 0:
            print(f"===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}): Loss: {loss.item():.10f}")
    
    # Print epoch summary
    avg_loss = epoch_loss / len(training_data_loader)
    print(f"===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}")
    return avg_loss

def validate():
    avg_psnr = 0
    model.eval()
    with torch.no_grad():
        for batch in validation_data_loader:
            msi, hsi = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            output = model(msi)
            
            # Convert to numpy for PSNR calculation
            hsi_np = hsi.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Calculate PSNR
            psnr = MPSNR(output_np, hsi_np)
            avg_psnr += psnr
    
    avg_psnr /= len(validation_data_loader)
    print(f"===> Validation Avg. PSNR: {avg_psnr:.4f} dB")
    return avg_psnr

def checkpoint(epoch, best=False):
    model_out_path = os.path.join(opt.save_folder, f"model_epoch_{epoch}.pth")
    if best:
        model_out_path = os.path.join(opt.save_folder, "model_best.pth")
    
    save_dict = {
        'epoch': epoch,
        'param': model.state_dict(),
        'adam': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr']
    }
    
    torch.save(save_dict, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")

def main():
    best_psnr = 0
    
        for epoch in range(start_epoch + 1, opt.nEpochs + 1):
            # Train for one epoch
            train(epoch, optimizer, scheduler)
            
            # Validate
            val_psnr = validate()
            
            # Save model checkpoint
            if epoch % 1 == 0:
                checkpoint(epoch)
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                checkpoint(epoch, best=True)
                print(f"New best model with PSNR: {best_psnr:.4f}")
            
            # Update learning rate
            scheduler.step()
            
            # Log to TensorBoard
            tb_logger.add_scalar('val_psnr', val_psnr, epoch)
            tb_logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

if __name__ == '__main__':
    main()
