import datetime
import time
import numpy as np


def progress_eta(count, total, prev_time, c_time, prev_mean, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = min(100, round(100.0 * count / float(total), 1))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    current_batch_time = c_time-prev_time
    mean = (1-1/count) * prev_mean + (1/count)*current_batch_time
    eta = int((total - count) * mean)
    eta_str = str(datetime.timedelta(seconds=eta))

    print('[{0}] {1}{2} \t{3}\tETA: {4}'.format(
        bar, percents, '%', status, eta_str), end='')
    print('\r', end='')
    return mean


def train_model(model, training_size, metric_func, val_names, train_generator,
                val_generator, epochs, base_filename='./', log_dir='./log'):



    current_score = 0
    best_score = 0
    j = 0
    smooth_window = 25
    mean = 0

    num_iterations = training_size
    print('Start training...')
    print("Number of iterations per epoch is: " + str(num_iterations))
    print()

    for epoch in range(epochs):
        # Init metrics to log
        metrics = metric_func()

        for iteration, (X, y) in enumerate(train_generator()):
            prev_time = time.time()
            # Do a train step on a single batch
            logs = model.train_on_batch(X, y)
            c_time = time.time()

            for metric_val, (_, lst) in zip(logs, metrics.items()):
                lst.insert(0, metric_val)

            if iteration > smooth_window:
                metrics_string = 'Epoch: {}'.format(epoch)

                for m_name, m_lst in metrics.items():
                    metrics[m_name] = metrics[m_name][:smooth_window]
                    tensorboard.on_epoch_end(j, {m_name: np.mean(m_lst)})
                    metrics_string += '\t{0}: {1:.2}'.format(
                        m_name, np.mean(m_lst))

                # print progress
                mean = progress_eta(iteration + 1 - smooth_window,
                                    num_iterations-smooth_window,
                                    prev_time,
                                    c_time,
                                    mean,
                                    metrics_string)
                prev_time = c_time
                j += 1

        total_metrics = {}
        mean_metrics = {}

        # Compute validation in batches
        for X, y in val_generator():
            metrics_ = model.evaluate(X, y, verbose=0)

            for i, metric in enumerate(metrics_):
                try:
                    total_metrics[val_names[i]].append(metric)
                except:
                    total_metrics[val_names[i]] = [metric]

        # Average results
        for key, total_metric in total_metrics.items():
            mean_metrics[key] = np.mean(total_metric)

        # Log on tensorboard
        tensorboard.on_epoch_end(epoch, mean_metrics)

        # Check best score and swap if better
        current_score = mean_metrics['val_acc']

        print()

        if current_score > best_score:
            model.save_weights(base_filename + 'weights.' +
                               str(epoch) + '-' + str(current_score) + '.hdf5')
            best_score = current_score
            print("Saved. ")

        print("Validation Accuracy in is {0:.6f} at epoch {1}"
              .format(np.mean(mean_metrics['val_acc']), epoch))
        print("Validation Top K Accuracy is {0:.6f} at epoch {1}"
              .format(np.mean(mean_metrics['val_top_k']), epoch))

    tensorboard.on_train_end(None)
