from collections import OrderedDict
import datetime, pytz, sys, pickle, klepto
from os.path import dirname, abspath, exists, join
from os import makedirs
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mstats

tstamp = None

def create_dir_if_not_exists(dir):
    if not exists(dir):
        makedirs(dir)
        
def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime('%Y-%m-%dT%H-%M-%S.%f')
     
def get_ts():
    global tstamp
    if not tstamp:
        tstamp = get_current_ts()
    return tstamp

def save(obj, filepath, print_msg=True, use_klepto=True):
    if type(obj) is not dict and type(obj) is not OrderedDict:
        raise ValueError('Can only save a dict or OrderedDict NOT {}'.format(type(obj)))
    fp = proc_filepath(filepath, ext='.klepto' if use_klepto else '.pickle')
    if use_klepto:
        create_dir_if_not_exists(dirname(filepath))
        save_klepto(obj, fp, print_msg)
    else:
        save_pickle(obj, fp, print_msg)
        
def proc_filepath(filepath, ext='.klepto'):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    return append_ext_to_filepath(ext, filepath)

def append_ext_to_filepath(ext, fp):
    if not fp.endswith(ext):
        fp += ext
    return fp

def save_klepto(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    klepto.archives.file_archive(filepath, dict=dic).dump()
    
def save_pickle(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    with open(filepath, 'wb') as handle:
        if sys.version_info.major < 3:  # python 2
            pickle.dump(dic, handle)
        elif sys.version_info >= (3, 4):  # qilin & feilong --> 3.4
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError()
        
def get_model_info_as_str(opt):
    rtn = []
    d = vars(opt)
    for k in d.keys():
        v = str(d[k])
        if k == 'dataset_list':
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
        else:
            vsplit = v.split(',')
            assert len(vsplit) >= 1
            for i, vs in enumerate(vsplit):
                if i == 0:
                    ks = k
                else:
                    ks = ''
                if i != len(vsplit) - 1:
                    vs = vs + ','
                s = '{0:26} : {1}'.format(ks, vs)
                rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)

def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn

def get_model_info_as_command(opt):
    rtn = []
    d = vars(opt)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '--{}={}'.format(k, v)
        rtn.append(s)
    return 'python {} {}'.format(join(get_our_dir(), 'main.py'), '  '.join(rtn))
    
def plot_scatter_line(data_dict, label, save_dir):
    fn = f'scatter_{label}_iterations.png'
    ss = ['rs-','b^-','g^-','c^-','m^-','ko-','yo-']
    cs = [s[0] for s in ss]
    plt.figure()
    i = 0

    # min_size = min([len(x['incumbent_data']) for x in data_dict.values()])
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li, y_li = [], []

        # min_len = float('inf')
        # for x in data_dict_elt['incumbent_data']:
        #     if x[1] < min_len:
        #         min_len = x[1]

        for x in data_dict_elt['incumbent_data']:
            # if x[1] > FLAGS.recursion_threshold:
            #     break
            x_li.append(x[1])
            y_li.append(x[0])
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()

    plt.figure()
    fn = f'scatter_{label}_time.png'
    i = 0
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li = [x[2] for x in data_dict_elt['incumbent_data']]
        y_li = [x[0] for x in data_dict_elt['incumbent_data']]
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    # plt.close()

def plot_dist(data, label, save_dir, saver=None, analyze_dist=True, bins=None):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    fn = f'distribution_{label}.png'
    plt.figure()
    sns.set()
    ax = sns.distplot(data, bins=bins, axlabel=label)
    plt.xlabel(label)
    ax.figure.savefig(join(save_dir, fn))
    plt.close()
    
def get_our_dir():
    return dirname(dirname(abspath(__file__)))

def _analyze_dist(saver, label, data):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Analyzing distribution of {label} (len={len(data)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    quantiles = mstats.mquantiles(data, prob=probs)
    func(f'{label} {len(data)}')
    s = '\t'.join([str(x) for x in probs])
    func(f'\tprob     \t {s}')
    s = '\t'.join(['{:.2f}'.format(x) for x in quantiles])
    func(f'\tquantiles\t {s}')
    func(f'\tnp.min(data)\t {np.min(data)}')
    func(f'\tnp.max(data)\t {np.max(data)}')
    func(f'\tnp.mean(data)\t {np.mean(data)}')
    func(f'\tnp.std(data)\t {np.std(data)}')