import datetime
import time
import numpy as np
import math
import tensorflow as tf
from .log_helpers import progress_eta, log_scalar, setup_tensoroard

def train_model(model, train_generator, val_generator, training_size, epochs, batch_size,
                metric_names, best_model_metric, smooth_window=25, weights_dir='./', log_dir='./log'):
    
    assert best_model_metric in metric_names
    
    data = {
        'current_score': 0,
        'best_score': 0,
        'start': time.clock(),
        'end': 0,
        'mean': 0,
        'prev_time': 0,
        'num_iterations': math.ceil(training_size / batch_size),
        'window': smooth_window
    }
    
    # Setup Metrics and Logging
    init_metrics = lambda: dict([(name,[]) for name in metric_names])
    
    print('Start training...')
    print("Number of iterations per epoch is: " + str(data['num_iterations']))
    print()

    # Tensorboard setup
    global_step = setup_tensoroard(log_dir)
    data['global_step'] = global_step

    # Training Loop
    for epoch in range(epochs):
        # Init metrics to log
        metrics = init_metrics()

        # Start Training Epoch
        train_epoch(model, train_generator, epoch, metrics, data)
        
        # Validate Last Epoch
        val_epoch(model, val_generator, epoch, metric_names, 
                  best_model_metric, weights_dir, data)


def log_batch(metrics, i, epoch, data):
    if i > data['window']:
        metrics_string = 'Epoch: {}'.format(epoch)

        # Tensorboard add step
        data['global_step'].assign_add(1)

        for m_name, m_lst in  metrics.items():
            metrics[m_name] = metrics[m_name][:data['window']]
            log_scalar(m_name, np.mean(m_lst))
            metrics_string += '\t{0}: {1:.2}'.format(m_name, np.mean(m_lst))

        # print progress
        data['mean'] = progress_eta(i + 1 - data['window'], 
                                data['num_iterations']-data['window'],
                                data['prev_time'], 
                                data['c_time'],
                                data['mean'], 
                                metrics_string)
        data['prev_time'] = data['c_time']


def train_epoch(model, train_generator, epoch, metrics, data):
    for iteration, args in enumerate(train_generator):
        data['prev_time'] = time.time()
        # Do a train step on a single batch
        logs = model.train_on_batch(*args)
        data['c_time'] = time.time()

        for metric, (_, lst) in zip(logs, metrics.items()):
            lst.insert(0, metric)
        
        log_batch(metrics, iteration, epoch, data)
        

def val_epoch(model, val_generator, epoch, metrics, best_model_metric, weights_dir, data):
    total_metrics = dict([('val_' + metric, []) for metric in metrics])
    mean_metrics = {}

    # Compute validation in batches
    for args in val_generator:
        metrics_ = model.evaluate(*args, verbose=0)

        for i, metric in enumerate(metrics_):
            total_metrics['val' + metrics[i]].append(metric)

    # Average results & Log on Tensorboard
    for key, total_metric in total_metrics.items():
        mean_metrics[key] = np.mean(total_metric)
        log_scalar(key, mean_metrics[key])

    # Check best score and swap if better
    data['current_score'] = mean_metrics['val' + best_model_metric]

    print()

    # Check for improvement and save the best model
    if data['current_score'] > data['best_score']:
        model.save_weights("{0}weights.{1}-{2}.hdf5"
                           .format(weights_dir, str(epoch), str(data['current_score'])))
        data['best_score'] = data['current_score']
        print("Saved. ")

    print("Validation Accuracy in is {0:.6f} at epoch {1}"\
          .format(np.mean(mean_metrics['val_acc']), epoch))
    print("Validation Top K Accuracy is {0:.6f} at epoch {1}"\
          .format(np.mean(mean_metrics['val_top_k']), epoch))