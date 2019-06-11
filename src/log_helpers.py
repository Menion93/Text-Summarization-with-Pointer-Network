import tensorflow as tf
import datetime

def log_scalar(name, value):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar(name, value)


def setup_tensoroard(log_dir):
    summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
    summary_writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()
    return global_step

def progress_eta(count, total, prev_time, c_time, prev_mean, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = min(100, round(100.0 * count / float(total), 1))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    
    current_batch_time = c_time-prev_time
    mean = (1-1/count) * prev_mean + (1/count)*current_batch_time
    eta = int((total - count) * mean)
    eta_str = str(datetime.timedelta(seconds=eta))

    print('[{0}] {1}{2} \t{3}\tETA: {4}'.format(bar, percents, '%', status, eta_str), end='')
    print('\r', end='')
    return mean