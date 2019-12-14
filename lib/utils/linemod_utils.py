from lib.utils.base_utils import read_pickle
import numpy as np

def read_anns(ann_files):
    anns = []
    for ann_file in ann_files:
        anns += read_pickle(ann_file)
    return anns

def read_pose(rot_path, tra_path):
    rot = np.loadtxt(rot_path, skiprows=1)
    tra = np.loadtxt(tra_path, skiprows=1) / 100.
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

