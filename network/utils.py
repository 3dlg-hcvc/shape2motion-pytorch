import os
import random
import torch
import numpy as np
from datetime import datetime

from tools.utils import io


def set_random_seed(seed):
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


def duration_in_hours(duration):
    t_m, t_s = divmod(duration, 60)
    t_h, t_m = divmod(t_m, 60)
    duration_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    return duration_time

def get_latest_file_with_datetime(path, folder_prefix, subdir, ext, datetime_pattern='%Y-%m-%d_%H-%M-%S'):
    folders = os.listdir(path)
    folder_pattern = folder_prefix + datetime_pattern
    matched_folders = np.asarray([fd for fd in folders if fd.startswith(folder_prefix)
                                  if len(io.get_file_list(os.path.join(path, fd, subdir), ext))])
    if len(matched_folders) == 0:
        return '', ''
    timestamps = np.asarray([int(datetime.strptime(fd, folder_pattern).timestamp() * 1000) for fd in matched_folders])
    sort_idx = np.argsort(timestamps)
    matched_folders = matched_folders[sort_idx]
    latest_folder = matched_folders[-1]
    files = io.alphanum_ordered_file_list(os.path.join(path, latest_folder, subdir), ext=ext)
    latest_file = files[-1]
    return latest_folder, latest_file


class AvgRecorder(object):
    """
    Average and current value recorder
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
