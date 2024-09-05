import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)

# evaluation args
parser.add_argument('--weight_file', type=str)
parser.add_argument('--result_file', type=str)

# cnn args
parser.add_argument('--kernel_size', type=int)

# rnn args
parser.add_argument('--pooling_method', type=str)

# multi-task args
parser.add_argument('--alpha', type=float)

# log file name
parser.add_argument('--log_file', type=str)

args = parser.parse_args()

config = json.load(open('./config.json', 'r'))

def train(model, elogger, train_set, eval_set):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    # Set model to training mode
    model.train()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        print('Training on epoch {}'.format(epoch))
        for input_file in train_set:
            print('Train on file {}'.format(input_file))

            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                # Set model to training mode again (if it was set to eval mode elsewhere)
                model.train()

                # Forward pass and loss calculation
                _, loss = model.eval_on_batch(attr, traj, config)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()  # Ensure model is in training mode during backward pass
                optimizer.step()

                running_loss += loss.data.item()
                print('\rProgress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0)), end='')
            print()
            elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))

        # Evaluate the model after each epoch
        evaluate(model, elogger, eval_set, save_result=False)

        # Save the weight file after each epoch
        weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
        elogger.log('Save weight file {}'.format(weight_name))

        if not os.path.exists('./saved_weights'):
            os.makedirs('./saved_weights')

        torch.save(model.state_dict(), './saved_weights/' + weight_name)

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def evaluate_metrics(y_true, y_pred):
    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE as a percentage
    
  
    print(f'RMSE: {rmse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.4f}%')
    
    # Return the calculated metrics
    return mse, rmse, r2, mae, mape
# def evaluate(model, elogger, files, save_result=False):
#     model.eval()
#     if save_result:
#         fs = open('%s' % args.result_file, 'w')

#     for input_file in files:
#         running_loss = 0.0
#         data_iter = data_loader.get_loader(input_file, args.batch_size)

#         for idx, (attr, traj) in enumerate(data_iter):
#             attr, traj = utils.to_var(attr), utils.to_var(traj)

#             pred_dict, loss = model.eval_on_batch(attr, traj, config)

#             if save_result:
#                 write_result(fs, pred_dict, attr)

#             running_loss += loss.data.item()

#         print('Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0)))
#         elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))

#     if save_result:
#         fs.close()
def evaluate(model, elogger, files, save_result=False):
    model.eval()
    all_preds = []
    all_labels = []

    if save_result:
        fs = open('%s' % args.result_file, 'w')

    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            # Evaluate model on batch
            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            # Get predictions and labels
            y_pred = pred_dict['pred'].data.cpu().numpy()  # Predictions
            y_true = pred_dict['label'].data.cpu().numpy()  # Ground Truth
            
            # Collect predictions and labels for calculating metrics
            all_preds.append(y_pred)
            all_labels.append(y_true)

            running_loss += loss.data.item()

            if save_result:
                write_result(fs, pred_dict, attr)

        print(f"Evaluate on file {input_file}, Loss: {running_loss / (idx + 1):.4f}")
        elogger.log(f"Evaluate on file {input_file}, Loss: {running_loss / (idx + 1):.4f}")

    if save_result:
        fs.close()

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute evaluation metrics, including MAPE
    mse, rmse, r2, mae, mape = evaluate_metrics(all_labels, all_preds)
    
    return mse, rmse, r2, mae, mape,y_pred,y_true


    

def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args  # Use `getfullargspec` for Python 3
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        log_dir = './saved_weights/'
        train(model, elogger, train_set=config['train_set'], eval_set=config['eval_set'])
        weight_name = f'{args.log_file}_final_weights.pth'
        weight_path = os.path.join(log_dir, weight_name)
        torch.save(model.state_dict(), weight_path)
        print(f"Weights saved at {weight_path}")

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result=True)

if __name__ == '__main__':
    run()
