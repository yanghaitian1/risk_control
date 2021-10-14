import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import re
import datetime as dt
import importlib
import logging
import numpy as np
import os
import pandas as pd
import sys
from glob import glob
import yaml
from os.path import join, split, splitext, basename, exists, abspath, dirname
from ..esmm.tools import read_file, split_df, train_model_3t, load_tf_model, get_df_pred_3t
from ..feature_proc import feature_proc as fp
from ..esmm.esmm_test_3label import ESMM


def split_data(data, save_path, split_cfg={'train_tf':0.6, 'valid_tf':0.4}):

    value_list = list(split_cfg.values())
    key_list = list(split_cfg.keys())

    out = split_df(data, split_size=value_list)
    if save_path is not None:
        for i in range(len(split_cfg)):
            out[i].to_pickle(join(save_path, key_list[i]+'.pkl'))
    else:
        return out

def feas2int(data_list, cols, target, dict_out=None):

    if dict_out is None:
        feas_cfg = {f: 'feas2int' for f in cols}
        dict_out = fp.str2int(data_list[0], feas_cfg, target=target, replace=False)[1]

    i=0
    while i<=len(data_list)-1:
        data = data_list[0]
        data = data.replace(dict_out)
        tmp = data_list.pop(0)
        data_list.append(data)
        i = i+1

    return dict_out, data_list