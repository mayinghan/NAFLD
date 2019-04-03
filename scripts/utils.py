import pickle as pkl
import pandas as pd
from collections.abc import Iterable
from nafld_config import MODEL_DIR, RESULTS_DIR, IMG
import os
import json
from sklearn.tree import export_graphviz
import subprocess
import shlex


def pkl_process(data=None, file=None, mode="dump"):
    f_mode = "wb" if mode == "dump" else "rb"
    load_data = None
    
    with open(file, f_mode) as f:
        if mode == "dump":
            pkl.dump(data, f)
        else:
            load_data = pkl.load(f)
    
    return load_data


def pkl_load(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data
  

def pkl_save(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f)


def load_csv(file, exclude_cols=None):
    ex_col = []
    if isinstance(exclude_cols, Iterable):
        ex_col = exclude_cols

    df = pd.read_csv(
            file,
            usecols=lambda col: col not in ex_col
        )

    return df


def generate_seeds(n=10):
    seeds = []
    while n > 0:
        seed = 2*n+1
        seeds.append(seed)
        n -= 1
    return seeds


def __check_file_exist(filename):
    flag = True
    if os.path.exists(filename):
        decision = input(f"{filename} is existed. Do you want to re-write? (1->yes/0->no)")
        if decision == '0':
            flag = False
    return flag


def save_models(model, name):
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    model_file = f"{MODEL_DIR}/{name}.pkl"
    if __check_file_exist(model_file):
        pkl_save(model, model_file)


def results2json(d, filename):
    if not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    fname = f"{RESULTS_DIR}/{filename}"
    if __check_file_exist(fname):
        with open(fname, "w") as f:
            json.dump(d, f, indent=2)


def load_json(filename):
    with open(filename, "r") as f:
        d = json.load(f)
    return d


def write_results_to_table(d, filename):
    pass


def draw_decision_tree(clf, class_names=None, feature_names=None, dot_file=None, png_file=None):
    if not os.path.isdir(IMG):
        os.mkdir(IMG)

    dot_file = f"{IMG}/{dot_file}"
    png_file = f"{IMG}/{png_file}"
    eps_file = png_file.split('.')[0] + ".eps"

    dot = export_graphviz(clf, out_file=dot_file, class_names=class_names, feature_names=feature_names, filled=True)
    p = subprocess.Popen(shlex.split(f"dot -Tpng {dot_file} -o {png_file} -Gdpi=600"),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o = p.communicate()
    print(o[0].decode('utf8'))
    print(o[1].decode('utf8'))

    p = subprocess.Popen(shlex.split(f"dot -Teps {dot_file} -o {eps_file}"), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    o = p.communicate()
    print(o[0].decode('utf8'))
    print(o[1].decode('utf8'))

    os.remove(dot_file)
