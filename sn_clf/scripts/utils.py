import numpy as np
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import json
import onnxruntime as rt
import requests
import matplotlib.pyplot as plt

akb_sn_tags = set(['SNIa', 'SN', 'SLSN', 'CCSN'])
def get_sn_label_from_akb(filepath):
    file = open(filepath)
    obj_list = json.load(file)
    file.close()

    oids = []
    tags = []
    for data in obj_list:
        oids.append(data['oid'])
        tags.append(data['tags'])

    targets = [] # 1-SN,  0-non SN
    for tag_list in tags:
        if set(tag_list).intersection(akb_sn_tags):
            targets.append(1)
        else:
            targets.append(0)
    
    return np.array(oids), np.array(targets)


def get_art_label_from_akb(filepath):
    file = open(filepath)
    obj_list = json.load(file)
    file.close()

    oids = []
    tags = []
    for data in obj_list:
        oids.append(data['oid'])
        tags.append(data['tags'])

    # 0-artefact,  1-transient
    result = {}
    for oid, tag_list in zip(oids, tags):
        tag_list = set(tag_list)
        if tag_list == set(['Galaxy']):
            continue
        if tag_list == set(['Galaxy', 'uncertain']):
            continue
        if tag_list == set(['interaction']):
            continue
        if tag_list == set(['1-point']):
            continue
        if tag_list == set(['LSB', 'uncertain']):
            continue
        if tag_list == set(['M_dwarf_flare', 'uncertain']):
            continue
        if tag_list == set(['VAR', 'uncertain']):
            continue
        if 'STAR' in tag_list:
            continue
        if 'Asteroid' in tag_list:
            continue
        if 'Comet' in tag_list:
            continue
        if 'artefact' in tag_list and 'uncertain' in tag_list:
            continue

        
        if 'artefact' in tag_list:
            result[oid] = 0
        
        else:
            result[oid] = 1
    
    return np.array(list(result.keys())), np.array(list(result.values()))


def load_features(oid_filename, feature_filename):
    oid     = np.memmap(oid_filename, mode='c', dtype=np.uint64)
    feature = np.memmap(feature_filename, mode='c', dtype=np.float32).reshape(oid.shape[0], -1)
    return oid, feature


def download_akb_json(filename):
    url = f'https://akb.ztf.snad.space/objects/'
    with requests.get(url) as response:
        response.raise_for_status()
        open(f'../data/{filename}', 'wb').write(response.content)


def convert_to_onnx(model, input_shape, name):    
    initial_type = [('float_input', FloatTensorType([None, input_shape]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(f'../models/{name}.onnx', "wb") as f:
        f.write(onx.SerializeToString())


def load_snmodel(model_name, num_threads=None):
    """Load ONNX model with memory optimization"""
    sess_options = rt.SessionOptions()
    sess_options.enable_mem_pattern = False  # Disable memory pattern for large models
    
    if num_threads:
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
    
    sess = rt.InferenceSession(
        f'../models/{model_name}.onnx',
        sess_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    prob_name = sess.get_outputs()[1].name
    return sess, input_name, prob_name

def process_chunk(model, chunk_data, concat=False):
    """Process a single chunk of data"""
    sess, input_name, prob_name = model
    pred_proba = sess.run([prob_name], {input_name: chunk_data.astype(np.float32)})[0]
    proba = np.float32([pred[1] for pred in pred_proba])
    
    if concat:
        return np.hstack((chunk_data, proba.reshape((-1, 1))))
    return proba



def plot_config():
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["mathtext.fontset"] = 'dejavuserif'
    plt.rcParams["font.size"] = 22
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.2

    xtick_param = {'direction': 'in',
         'major.size': 8,
         'major.width': 2,
         'minor.size': 5,
         'minor.width': 1.5}
    ytick_param = {'direction': 'in',
         'major.size': 8,
         'major.width': 2,
         'minor.size': 5,
         'minor.width': 1.5}
    plt.rc('xtick', **xtick_param)
    plt.rc('ytick', **ytick_param)

    grid_param = {'linestyle': '--', 'alpha': 0.5}
    plt.rc('grid', **grid_param)

