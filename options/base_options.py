import argparse
import os

class BaseOptions():
    opt = None
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.initialized = True
        self.parser.add_argument('--log_level','-ll', type=str, default='info',
                                 help='Specify the logging level to show. Can be "debug", "info", "warning", "error" or "critical".')
        self.parser.add_argument('--log_file','-lf', type=str, default=None,
                                 help='Specify the file to log to. If not specified, the log will be saved to file.')
        self.parser.add_argument('--log_stdout','-ls', type=bool, default=True,
                                 help='Specify whether to log to stdout.')
        self.parser.add_argument('--export_computed_pairs', '-ecp', type=bool, default=False,
                                 help='If true, save the edge lists of all analyzed pairs to file')

    def parse(self):
        if not self.initialized:
            self.initialize()
            
        self.opt = self.parser.parse_args()
                
        return self.opt
