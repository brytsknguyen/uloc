import torch
import torch.nn as nn
from util.eval import Result
import matplotlib.pyplot as plt
from IPython import display
from torch.utils.tensorboard import SummaryWriter
import csv
import os

def train(train_loader, model, criterion, optimizer, epoch,device):
    
    model.train()  # switch to train mode
    
    for batch_idx, (x, y_gt) in enumerate(train_loader):
        # print(y_gt.shape)
        x = x.float().to(device)
        y_gt = y_gt.float().to(device)

        # compute pred
        y_pred = model(x)

        loss = criterion(y_pred, y_gt)
        optimizer.zero_grad()
        loss.backward()                 # compute gradient and do SGD step
        optimizer.step()

def validate(val_loader, model, epoch, check_mode,device,Yscaler,half=0):
    
    # Switch to validate mode
    model.eval()
    test_result = Result(half=half)
    
    for i, (x, y_gt) in enumerate(val_loader):
        
        x = x.float()
        y_gt = y_gt.float()
        
        x, y_gt = x.to(device), y_gt.to(device)
        
        with torch.no_grad():
            y_pred = model(x)

        # Unscale output, also only take one sample at the end of the sequence
        y_pred = y_pred[:, -1, :]
        y_gt = y_gt[:, -1, :]

        y_pred_unscaled = Yscaler.undo_scale(y_pred.data.cpu())
        y_gt_unscaled   = Yscaler.undo_scale(y_gt.data.cpu())

        # print(y_pred_unscaled[-1, :], y_gt_unscaled[-1, :])

        # Set result
        seq_result = Result(half=half)
        seq_result.evaluate(y_pred_unscaled, y_gt_unscaled)
        test_result = test_result + seq_result

    # Report the RMSE
    print(f'Epoch: {epoch:2d}. {i:2d} / {len(val_loader):d} | Mode: {check_mode:s}. RMSE: {test_result.rmse:.6f}. MEAN:  {test_result.mean:.6f}. MEDIAN: {test_result.median:.6f}')

    return test_result


def network_train(model, train_dataloader, test_dataloader, Yscaler, criterion, optimizer, num_epoch, device, lr, decay_rate, decay_step, vis=1, save_weights=0, save_path='weights', log_path='logs/',half=0,real=0):
    best_res = 1000
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE [m]')

    rmse = []
    rmse_line, = ax.plot(0, 0, 'r')

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_path)

    # Initialize CSV file and write header if not exists
    csv_file_path = os.path.join(log_path, 'rmse_log.csv')
    os.makedirs(log_path, exist_ok=True)  # Ensure log directory exists


    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'RMSE'])

    for epoch in range(num_epoch):

        # Adjust learning rate
        lrk = lr * (decay_rate ** (epoch // decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrk

        train(train_dataloader, model, criterion, optimizer, epoch, device)  # train for one epoch
        res = validate(test_dataloader, model, epoch, 'TEST', device, Yscaler,half=half)  # evaluate on test set
        
        # Get the rmse
        rmse.append(res.rmse)
        current_res = res.rmse

        # Log RMSE to TensorBoard
        if real:
            writer.add_scalar('GT/RMS', current_res, epoch)
        else:
            writer.add_scalar('SLAM_prior/RMS', current_res, epoch)


        # Log RMSE to CSV
        with open(csv_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch, current_res])

        if save_weights and current_res < best_res:
            torch.save(model, save_path)
            best_res = current_res

        if vis:
            # Plot the rmse
            ax.clear()
            ax.plot(range(epoch + 1), rmse, 'r')
            ax.text(epoch, rmse[-1] * 1.1, f'{rmse[-1]:.3f}')

            display.clear_output(wait=True)
            display.display(plt.gcf())

    # Close the TensorBoard writer
    writer.close()

    return rmse

