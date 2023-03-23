from collections import OrderedDict
from time import time
import numpy as np
import logging

class OurTimer(object):
    def __init__(self):
        self.t = time()
        self.durations_log = OrderedDict()

    def time_and_clear(self, log_str='', only_seconds=False):
        duration = self._get_duration_and_reset()
        if log_str:
            if log_str in self.durations_log:
                raise ValueError('log_str {} already in log {}'.format(
                    log_str, self.durations_log))
            self.durations_log[log_str] = duration
        if only_seconds:
            rtn = duration
        else:
            rtn = format_seconds(duration)
        logging.info(log_str, '\t\t', rtn)
        return rtn

    def start_timing(self):
        self.t = time()

    def print_durations_log(self):
        logging.info('Timer log', '*' * 50)
        rtn = []
        tot_duration = sum([sec for sec in self.durations_log.values()])
        logging.info('Total duration:', format_seconds(tot_duration))
        lss = np.max([len(s) for s in self.durations_log.keys()])
        for log_str, duration in self.durations_log.items():
            s = '{0}{1} : {2} ({3:.2%})'.format(
                log_str, ' ' * (lss - len(log_str)), format_seconds(duration),
                         duration / tot_duration)
            rtn.append(s)
            logging.info(s)
        logging.info('Timer log', '*' * 50)
        self.durations_log = OrderedDict() # reset
        return rtn

    def _get_duration_and_reset(self):
        now = time()
        duration = now - self.t
        self.t = now
        return duration

    def get_duration(self):
        now = time()
        duration = now - self.t
        return duration

    def reset(self):
        self.t = time()

def format_seconds(seconds):
    """
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('min', 60),
        ('sec', 1)
    ]

    if seconds <= 1:
        return '{:.3f} msecs'.format(seconds * 1000)

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            if period_name == 'sec':
                period_value = seconds
                has_s = 's'
            else:
                period_value, seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 else ''
            strings.append('{:.3f} {}{}'.format(period_value, period_name, has_s))

    return ', '.join(strings)
