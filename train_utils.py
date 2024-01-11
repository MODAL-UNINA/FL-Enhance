#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 23:04:40 2022

@author: modal
"""

# %%

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from .utils import is_interactive
from tqdm.auto import tqdm, trange

# %%


# Due to a bug with tensorflow gradient tape, we cannot loop multiple model
# training in the same python execution.
# See https://github.com/tensorflow/tensorflow/issues/27120 for the problem and
# https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844
# for the workaround.
def get_apply_grad_fn():
    @tf.function
    def apply_grad(model, loss, x, y_true, optimizer, mu=0, global_weights=None):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_dict = loss(y_true, y_pred)
            if global_weights is not None:
                loss_dict['loss_penalty'] = mu/2 * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(w-gw)) for w, gw in zip(model.trainable_variables, global_weights)])
                loss_dict['loss'] += loss_dict['loss_penalty']
        gradients = tape.gradient(loss_dict['loss'], model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_dict
    return apply_grad

@tf.function
def train_step(model, loss, x, y_true, optimizer, mu=0, global_weights=None):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    apply_grads = get_apply_grad_fn()
    loss_dict = apply_grads(model, loss, x, y_true, optimizer, mu, global_weights)
    return loss_dict


# Uncomment this when the bug is fixed
# @tf.function
# def train_step(model, x, y_true, optimizer):
#     """Executes one training step and returns the loss.

#     This function computes the loss and gradients, and uses the latter to
#     update the model's parameters.
#     """
#     with tf.GradientTape() as tape:
#         loss_dict = model.compute_loss(x, y_true)
#     gradients = tape.gradient(loss_dict['loss'], model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss_dict


# Optimizator choice
# def select_optimizator(cf, lr):
#     if cf.opt == 'Adam':
#         opt = tf.keras.optimizers.Adam(learning_rate=lr)
#     elif cf.opt == 'RMSprop':
#         opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
#     elif cf.opt == 'Adadelta':
#         opt = tf.keras.optimizers.Adadelta(learning_rate=lr)
#     return opt

# local_model.fit(clients_train[client], clients_label[client], 
#         epochs=epochs, verbose=0, batch_size=batch_size)

def train(X, y, model, loss, opt, cf, global_weights=None, val_data=None):
    # assert isinstance(cf.lr, collections.abc.Iterable), "cf.lr is not iterable"
    # lr_iter = iter(cf.lr); lr = next(lr_iter)
    # epoch_change = []
    # opt = select_optimizator(cf, lr)
    start = time.time()
    best_loss = tf.constant(np.inf)
    patience = cf.patience
    wait = 0
    flag_epoch = False
    verbose = cf.verbose

    history = {}
    
    keys = None
    nrows = 0
    ncols = 0

    data = tf.data.Dataset.from_tensor_slices((X, y))
    data = data.shuffle(len(X), seed=cf.seed)
    data = data.batch(cf.batch_size)

    if verbose:
        range_epochs = trange(1,cf.epochs+1, desc='Epoch')
    else:
        range_epochs = range(1,cf.epochs+1)

    for epoch in range_epochs:
        start_epoch = time.time()
        loss_dict = {}

        for X_batch, y_batch in data:
            loss_batch = train_step(
                model, loss, X_batch, y_batch, opt, cf.mu, global_weights if cf.algorithm == 'fedprox' else None)
            for key, elem in loss_batch.items():
                loss_dict.setdefault(key, [])
                loss_dict[key].append(elem.numpy())

        if val_data is not None:
            for X_batch, y_batch in val_data:
                loss_batch = loss(X_batch, y_batch)
                for key, elem in loss_batch.items():
                    val_key = 'val_' + key
                    # history.setdefault(val_key, [])
                    loss_dict.setdefault(val_key, [])
                    loss_dict[val_key].append(elem.numpy())

        history.setdefault('epoch', [])
        history['epoch'].append(epoch)
        for key, elem in loss_dict.items():
            history.setdefault(key, [])
            if key.endswith('maxe'): history[key].append(np.max(elem))
            else: history[key].append(np.mean(elem))

        if val_data is not None:
            loss_value = history['val_loss'][-1]
        else:
            loss_value = history['loss'][-1]

        if loss_value<best_loss:
            if verbose:
                if flag_epoch:
                    # print('\n\n')
                    print('\n', end='')
                    flag_epoch = False
                mega_str = f'Best {epoch}: '

                for key in loss_dict.keys():
                    if key != 'epoch':
                        mega_str += f'{key} = {history[key][-1]:10.4e}  '
                
                mega_str += f'Epoch time {time.time()-start_epoch:.2f} secs'
                print('\r'+mega_str, end= '')
                            
            best_loss = loss_value
            best_epoch = epoch
            wait = 0
            best_weights = model.get_weights()
        elif patience is not None and wait>=patience:
            # lr = next(lr_iter, None)
            model.set_weights(best_weights)
            # if lr is not None:
            #     print(f'\n\nChanging learning rate to {lr}')
            #     opt = select_optimizator(cf, lr)
            #     wait = 0
            #     flag_epoch = True
            #     epoch_change.append(epoch)
            # else:
            if True:
                if verbose:
                    print('\nStop the train phase')
                break
        else:
            if verbose:
                if time.time()-start_epoch>cf.epoch_time or epoch % cf.plot_epochs[0] == 0:
                    flag_epoch = True
                    if wait == 0:
                        epoch_str = f'\n\nEpoch {epoch}: '
                    else:
                        epoch_str = f'\n\nEpoch {epoch}: '
                    for key in loss_dict.keys():
                        if key == 'epoch': continue
                        epoch_str += f'{key} = {history[key][-1]:10.4e}, '
                    epoch_str += f'Epoch time {time.time()-start_epoch:.2f} secs'
                    print(epoch_str, end='\n')
            wait +=1

        # if epoch % cf.plot_epochs[1] == 0 and is_interactive():
        #     if epoch == cf.plot_epochs[1]:
        #         keys = [k for k in history.keys() if not k.startswith('val_') and k not in ['epoch']]
        #         nrows = int(np.ceil(np.sqrt(len(keys))))
        #         ncols = nrows

        #     print('\nPlotting...')
        #     fig = plt.figure(figsize=(10*ncols, 5*nrows))
        #     plt.suptitle(f'Last {cf.plot_epochs[1]} epochs')

        #     for i, key in enumerate(keys):
        #         fig.add_subplot(nrows, ncols, i+1)
        #         plt.title(f'Metric: {key}')
        #         plt.plot(history['epoch'][-cf.plot_epochs[1]:], 
        #                 history[key][-cf.plot_epochs[1]:], label='train')
        #         if f'val_{key}' in history.keys():
        #             plt.plot(history['epoch'][-cf.plot_epochs[1]:], 
        #                     history[f'val_{key}'][-cf.plot_epochs[1]:], label='val')
        #             plt.legend()
        #         plt.yscale('log')
        #     plt.show()

    if verbose:
        print('\nComputation time: {} seconds'.format(time.time()-start))
    history = {k:v for k,v in history.items() if len(v)>=1}
    if patience is not None and epoch == cf.epochs and best_epoch != epoch:
        if verbose:
            print('\nRestore the best weights')
        model.set_weights(best_weights)

    return history, model#, {'epoch_change': epoch_change}
